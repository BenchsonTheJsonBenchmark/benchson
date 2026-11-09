"""Builds the description-grounded benchmark from a schema corpus.

A single minted `reference_json` feeds up to three ground-truth tasks:

- **create** (2a): a natural-language `description` that encodes all of the
  object's values, so a model can reconstruct the JSON from the description.
- **fix** (2b): the object with a single schema-violating error injected, to be
  repaired back toward the `reference_json`.
- **modify** (2c): a free-text change instruction plus the correctly modified
  object (a programmatic, schema-valid edit), to be reproduced by the model.

Schemas are read from one or more local datasets (e.g. Benchson's `schemas` and the
imported `jsb_schemas`); each instance records its `source`/`subset` so results can
be sliced by origin and difficulty tier.

Optional "generation" config block (see data_generation/generation_mix.py): each task's
count is split into "standard" examples (the original behaviour) and "complex" ones
(several edits / errors, richer references). Without that block nothing changes.
"""

import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import re
import threading

from data_generation import chunking
from data_generation.anchor import ANCHOR_FILENAME, AnchorError, load_anchor, select_pool
from data_generation.complexity import (fix_metrics, meets_create_thresholds, modify_metrics,
                                        reference_metrics, schema_can_support, single_fix_metrics)
from data_generation.fix_errors import inject_errors, verify_fix
from data_generation.generation_mix import MODES, GenerationSettings
from data_generation.published import canon_hash, load_reference_index, load_reserved, reference_hash, schema_hash
from data_generation.reuse import DuplicateIndex, ReuseSettings, SchemaLedger
from data_generation.tasks.base_task import BaseDataTask
from data_generation.tasks.error_correction import _inject_error_info
from data_generation.modifications import (apply_change, apply_changes,
                                           instruction_covers_operations, verify_modification)
from evaluations import metrics

_JSB_ATTRIBUTION = (
    "Schemas with source 'jsonschemabench' are derived from "
    "epfl-dlab/JSONSchemaBench (MIT, arXiv:2501.10868)."
)


def _looks_truncated(text):
    """Heuristic for a cut-off model answer: JSON-looking text that never closes, or an
    unclosed <think> block. Natural-language answers cannot be judged and return False."""
    body = (text or "").strip()
    if "<think" in body.lower() and "</think>" not in body.lower():
        return True
    body = body.lstrip("`").removeprefix("json").lstrip()
    return body[:1] in "{[" and body[:1] != "" and not body.rstrip("` \n").endswith(("}", "]"))


def _is_context_overflow(error):
    return bool(getattr(error, "is_context_overflow", False))


def _overflow_info(error):
    return {"prompt_tokens": error.prompt_tokens, "max_tokens": error.max_tokens,
            "max_model_len": error.max_model_len, "messages": error.n_messages}


class _CountingLLM:
    """Delegates to a provider, counts the calls made through it, and reports problems.

    `sink(kind, detail)` receives provider errors (the exception is still raised), empty
    answers and suspected truncations, so they end up in the run's event log.
    """

    def __init__(self, inner, sink=None, call_params=None):
        self.inner = inner
        self.calls = 0
        self.sink = sink
        self.call_params = call_params   # e.g. {"seed": 123}: extra provider parameters for every call

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.call_params:
            args = list(args)
            if len(args) > 1:
                args[1] = {**(args[1] or {}), **self.call_params}
            else:
                kwargs["parameters"] = {**(kwargs.get("parameters") or {}), **self.call_params}
        try:
            result = self.inner.generate(*args, **kwargs)
        except Exception as e:
            if self.sink:
                # A request refused BEFORE sending because it cannot fit the context window is an expected capacity
                # exclusion; every other provider error stays a (technical) provider_error.
                kind = "context_overflow" if _is_context_overflow(e) else "provider_error"
                self.sink(kind, f"{type(e).__name__}: {str(e)[:300]}")
            raise
        if self.sink:
            text = result[0] if isinstance(result, tuple) else result
            if not (text or "").strip():
                self.sink("empty_response", None)
            elif _looks_truncated(text):
                self.sink("suspected_truncation", f"{len(text)} chars, ends with {text.strip()[-40:]!r}")
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


class BenchmarkBuilder:

    _TASK_LABELS = {
        "create": "create-by-description",
        "fix": "fix-invalid",
        "modify": "modify-by-instruction",
    }

    def __init__(self, config):
        self.sources = config["sources"]
        self.max_retries = config.get("max_retries", 3)
        self.seed = config.get("seed", 0)
        self.concurrency = config.get("concurrency", 1)
        self.system_prompt_prefix = config.get("system_prompt_prefix", "")
        # An optional separate model verifies the round-trip gate (defaults to the
        # builder model). Its prompt prefix is configured independently, since the
        # builder's prefix may be model-specific (e.g. gpt-oss "Reasoning: low").
        self.validator_system_prompt_prefix = config.get("validator_system_prompt_prefix", "")
        self.test_ratio = config.get("split", {}).get("test_ratio", 0.15)

        # Per-split counts + gates. New shape:
        #   counts: {test: {...}, train: {...}}, validation: {test: {...}, train: {...}}
        # Old flat shape (counts/validation without test/train) is treated as test-only.
        counts_cfg = config.get("counts", {"create": 30, "fix": 30, "modify": 30})
        val_cfg = config.get("validation", {})
        if "test" in counts_cfg or "train" in counts_cfg:
            self.splits = [s for s in ("test", "train") if s in counts_cfg]
            self.counts_by_split = {s: counts_cfg[s] for s in self.splits}
            self.gates_by_split = {s: self._parse_gates(val_cfg.get(s, val_cfg)) for s in self.splits}
        else:
            self.splits = ["test"]
            self.counts_by_split = {"test": counts_cfg}
            self.gates_by_split = {"test": self._parse_gates(val_cfg)}

        # `output_root` redirects where instances are written (used by preview runs so they
        # never touch data/). Schemas are still read from data/ as before.
        data_root = os.path.abspath(config["output_root"]) if config.get("output_root") \
            else os.path.join(os.getcwd(), "data")
        self.base = {
            "create": os.path.join(data_root, config.get("create_dataset", "benchmark_create")),
            "fix": os.path.join(data_root, config.get("fix_dataset", "benchmark_fix")),
            "modify": os.path.join(data_root, config.get("modify_dataset", "benchmark_modify")),
        }

        self.data_root = data_root
        # Schemas already used by a published dataset (train or test) that must not be reused.
        self.reserved_ids, self.reserved_hashes, self.reserved_info = set(), set(), None
        exclusions = config.get("exclusions") or {}
        if exclusions.get("file"):
            self.reserved_ids, self.reserved_hashes, self.reserved_info = load_reserved(exclusions["file"])
        self.reuse = ReuseSettings.from_config(config)
        # Chunked full run (see data_generation/chunking.py): {"index", "n_chunks", "totals", "parent_dir", "config_sha256"}
        self.chunk = config.get("chunk")
        if self.chunk and not self.reuse:
            raise ValueError("chunked generation needs the reuse block")
        self._published_refs, self._published_keys = {}, None
        if self.reuse:
            if not self.reserved_info:
                raise ValueError("reuse needs exclusions.file (the published test split must be excluded)")
            self._published_refs, self._published_keys = load_reference_index(exclusions["file"])
        self._reuse_report = None
        self._pool_stats = None
        self._events = []
        self._events_lock = threading.Lock()
        self.settings = GenerationSettings.from_config(config)
        if self.reuse:
            if self.splits != ["train"]:
                raise ValueError("reuse is for a train-only build (counts must contain only 'train')")
            if not self.settings.enabled or self.settings.complex_ratio != 0 or \
                    self.settings.standard_modify_strategy != "original":
                raise ValueError("reuse requires a generation block that is 100% standard with "
                                 "standard_modify_strategy 'original'")
        self.last_summary = None
        self._split_reports = {}
        self._run_started = datetime.datetime.now()
        self._run_completed = False

        self.task = BaseDataTask(system_prompt_prefix=self.system_prompt_prefix)
        self._partition_sizes = {}

    @staticmethod
    def _parse_gates(v):
        return {
            "deterministic_gates": v.get("deterministic_gates", True),
            "min_populated_fields": v.get("min_populated_fields", 2),
            "grounding_threshold": v.get("grounding_threshold", 0.9),
            "round_trip": v.get("round_trip", True),
            "round_trip_threshold": v.get("round_trip_threshold", 0.9),
            "max_build_attempts": v.get("max_build_attempts", 3),
            "naturalize_instruction": v.get("naturalize_instruction", True),
            "naturalize_attempts": v.get("naturalize_attempts", 2),
        }

    # ------------------------------------------------------------------ loading

    def _load_schema_records(self, apply_exclusions=True):
        """Returns a shuffled list of (schema_dict, source, subset, origin_name)."""
        loader = BaseDataTask()
        records = []
        for src in self.sources:
            name = src["name"]
            paths = loader.load_instances(
                source=name,
                filter_kw=src.get("filter"),
                files=src.get("files"),
                max_schema_kb=src.get("max_schema_kb", 10),
            )
            for path in paths:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(loaded, dict) and "schema" in loaded:
                    schema = loaded["schema"]
                    source = loaded.get("source", name)
                    subset = loaded.get("subset")
                    origin = loaded.get("name") or os.path.basename(path)
                else:
                    schema, source, subset = loaded, name, None
                    origin = os.path.splitext(os.path.basename(path))[0]
                if isinstance(schema, dict):
                    records.append((schema, source, subset, origin))

        loaded = len(records)
        by_id = by_hash = 0
        if apply_exclusions and (self.reserved_ids or self.reserved_hashes):
            kept = []
            for rec in records:
                if self._safe_id(rec[1], rec[2], rec[3]) in self.reserved_ids:
                    by_id += 1
                elif schema_hash(rec[0]) in self.reserved_hashes:
                    by_hash += 1
                else:
                    kept.append(rec)
            records = kept
        origins = {}
        for _, source, subset, _ in records:
            key = "/".join(p for p in (source, subset) if p) or "unknown"
            origins[key] = origins.get(key, 0) + 1
        self._pool_stats = {"schemas_loaded": loaded, "excluded_already_published_by_id": by_id,
                            "excluded_already_published_by_schema_hash": by_hash,
                            "usable_schemas": len(records), "usable_by_source": dict(sorted(origins.items()))}

        random.Random(self.seed).shuffle(records)
        return records

    def _partition(self, records):
        """Splits records into disjoint per-split schema pools.

        Stratified by (source, subset) so each split covers every origin/tier, and
        deterministic + stable (sorted by id, then a seeded shuffle) so a schema
        always lands on the same side across reruns. With a single split, all
        records go to it (no partition).
        """
        if self.splits == ["test"]:
            return {"test": records}

        groups = {}
        for rec in records:
            _, source, subset, origin = rec
            groups.setdefault((source, subset), []).append(rec)

        # Both pools are always computed, even when only one split is built, so a train-only
        # build still excludes exactly the schemas that belong to test (no leakage).
        pools = {"test": [], "train": []}
        rng = random.Random(self.seed)
        for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
            grp = sorted(groups[key], key=lambda r: self._safe_id(r[1], r[2], r[3]))
            rng.shuffle(grp)
            n_test = round(len(grp) * self.test_ratio)
            pools["test"].extend(grp[:n_test])
            pools["train"].extend(grp[n_test:])
        return pools

    # ------------------------------------------------------------------ build

    def build(self, llm_provider, validator_provider=None):
        # The validator runs the round-trip gate; falls back to the builder model.
        self._validator = validator_provider or llm_provider
        self._run_started = datetime.datetime.now()
        self._run_completed = False
        for base in self.base.values():
            os.makedirs(os.path.join(base, "test"), exist_ok=True)
            os.makedirs(os.path.join(base, "train"), exist_ok=True)

        try:
            if self.reuse:
                pools = {"train": self._reuse_pool()}
            else:
                pools = self._partition(self._load_schema_records())
            self._partition_sizes = {s: len(pools[s]) for s in self.splits}

            for split in self.splits:
                if self.reuse:
                    self._build_split_reuse(split, pools[split], llm_provider)
                else:
                    self._build_split(split, pools[split], llm_provider)

            for k in self.base:
                self._write_manifest(self.base[k], self._TASK_LABELS[k], llm_provider)
            self._run_completed = True
        finally:
            # Written even when the run is stopped early (budget / provider failure), so the
            # partial result is still described. Legacy configs (no "generation" block) get none.
            if self.settings.enabled:
                self._write_report(llm_provider)

    # ------------------------------------------------------------ metadata (sidecar) files

    def _meta_path(self, task, split, instance_id):
        return os.path.join(self.base[task], "_metadata", split, instance_id + ".json")

    @staticmethod
    def _note_content(seen, instance_id, task, record):
        """Remembers an example's content; returns the id of an earlier example with the same
        content (ignoring the schema), or None. Used to report duplicates across schemas."""
        parts = {"create": ("reference_json", "description"),
                 "fix": ("valid_json", "erroneous_json"),
                 "modify": ("data", "instructions", "ground_truth")}[task]
        digest = hashlib.sha1(json.dumps([record.get(k) for k in parts], sort_keys=True,
                                         default=str).encode("utf-8")).hexdigest()
        first = seen.get(digest)
        if first is None:
            seen[digest] = instance_id
        return first

    def _log_event(self, split, schema_id, kind, detail):
        with self._events_lock:
            self._events.append({"split": split, "schema_id": schema_id, "kind": kind, "detail": detail})

    def pool_report(self):
        """Schema supply check (no model needed): how many usable schemas exist per split."""
        pools = self._partition(self._load_schema_records())
        return {**self._pool_stats,
                "schemas_per_split": {sp: len(pools[sp]) for sp in self.splits},
                "targets_per_task": {sp: dict(self.counts_by_split[sp]) for sp in self.splits},
                "reserved": self.reserved_info}

    def _stored_mode(self, task, split, instance_id):
        """Mode an existing example was generated in ("standard" when it has no sidecar)."""
        try:
            with open(self._meta_path(task, split, instance_id), "r", encoding="utf-8") as f:
                return json.load(f).get("mode", "standard")
        except (OSError, json.JSONDecodeError):
            return "standard"

    def _build_split(self, split, pool, llm_provider):
        want = {k: self.counts_by_split[split].get(k, 0) for k in self.base}
        gates = self.gates_by_split[split]
        quota = self.settings.mix(want)   # {task: {"standard": n, "complex": n}}
        split_dir = {k: os.path.join(self.base[k], split) for k in self.base}
        for d in split_dir.values():
            os.makedirs(d, exist_ok=True)

        have = {k: len([f for f in os.listdir(split_dir[k]) if f.endswith(".json")]) for k in self.base}
        have_mode = {k: {m: 0 for m in MODES} for k in self.base}
        for k in self.base:
            for f in os.listdir(split_dir[k]):
                if f.endswith(".json"):
                    have_mode[k][self._stored_mode(k, split, f[:-5])] += 1
            if have[k]:
                print(f"[{split}/{k}] resuming from {have[k]}/{want[k]}")
        had_before = {k: dict(have_mode[k]) for k in self.base}

        reserved = {k: {m: 0 for m in MODES} for k in self.base}   # slots held by in-flight schemas
        attempts_failed = {k: {m: 0 for m in MODES} for k in self.base}
        closed = {k: {m: False for m in MODES} for k in self.base}   # complex cell out of failure budget
        failures = []
        metas = []
        llm_calls = 0
        # Registered up front and filled in place, so the report still describes a run that
        # is stopped early (call budget / provider failure) before this split finishes.
        seen = {k: {} for k in self.base}        # content fingerprint -> first instance id
        duplicates = []
        report = {"quota": quota, "had_before": had_before, "have_mode": have_mode,
                  "failed_attempts": attempts_failed, "closed": closed, "failures": failures,
                  "metas": metas, "llm_calls": 0, "duplicates": duplicates}
        self._split_reports[split] = report
        fail_budget = self.settings.complex.get("max_failures_per_example", 10)

        lock = threading.Lock()
        records = iter(pool)
        submitted = 0

        def open_slots(task):
            slots = {}
            for m in MODES:
                free = quota[task][m] - have_mode[task][m] - reserved[task][m]
                if free > 0 and not closed[task][m]:
                    slots[m] = free
            return slots

        def needed_for(instance_id, mode_rng):
            needed = {}
            for k in self.base:
                if os.path.exists(os.path.join(split_dir[k], instance_id + ".json")):
                    continue
                slots = open_slots(k)
                if not slots:
                    continue
                if len(slots) == 1:
                    needed[k] = next(iter(slots))
                else:  # both modes still open: pick one, weighted by how many slots each has left
                    needed[k] = mode_rng.choices(list(slots), weights=list(slots.values()))[0]
            return needed

        def targets_met():
            return all(have_mode[k][m] >= quota[k][m] or closed[k][m]
                       for k in self.base for m in MODES)

        def submit_next(executor):
            nonlocal submitted
            for schema, source, subset, origin in records:
                instance_id = self._safe_id(source, subset, origin)
                with lock:
                    needed = needed_for(instance_id, random.Random(f"{self.seed}-{split}-{submitted}-mode"))
                    for k, m in needed.items():
                        reserved[k][m] += 1
                if not needed:
                    continue
                seed_label = f"{self.seed}-{split}-{submitted}"
                rng = random.Random(seed_label)
                submitted += 1
                fut = executor.submit(self._process_schema, schema, source, subset,
                                      origin, needed, llm_provider, rng, gates, split, seed_label)
                inflight[fut] = (needed, instance_id)
                return fut
            return None

        inflight = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            pending = set()
            while len(pending) < self.concurrency:
                fut = submit_next(executor)
                if fut is None:
                    break
                pending.add(fut)

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    needed, instance_id = inflight.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as e:  # one bad schema must not kill the run
                        print(f"[{split}] worker error (skipped): {e}")
                        result = {"instance_id": instance_id, "produced": {}, "meta": {}, "llm_calls": 0,
                                  "failures": [{"task": t, "mode": m, "schema_id": instance_id,
                                                "stage": "worker_error", "detail": str(e)}
                                               for t, m in needed.items()]}
                    with lock:
                        for k, m in needed.items():
                            reserved[k][m] -= 1
                        llm_calls += result.get("llm_calls", 0)
                        report["llm_calls"] = llm_calls
                        for task, record in result["produced"].items():
                            mode = needed[task]
                            path = os.path.join(split_dir[task], instance_id + ".json")
                            if have_mode[task][mode] < quota[task][mode] and not os.path.exists(path):
                                self._write(path, record)
                                have[task] += 1
                                have_mode[task][mode] += 1
                                meta = result["meta"].get(task)
                                first = self._note_content(seen[task], instance_id, task, record)
                                if first is not None:   # detection only: reported, not rejected
                                    duplicates.append({"task": task, "mode": mode, "split": split,
                                                       "schema_id": instance_id, "duplicate_of": first})
                                if meta is not None and self.settings.enabled:
                                    meta["split"] = split
                                    meta["duplicate_of"] = first
                                    os.makedirs(os.path.dirname(self._meta_path(task, split, instance_id)),
                                                exist_ok=True)
                                    self._write(self._meta_path(task, split, instance_id), meta)
                                    metas.append(meta)
                                tag = f" {mode}" if self.settings.enabled else ""
                                print(f"[{split}/{task}] {have[task]}/{want[task]}{tag} ({instance_id})")
                        for fail in result["failures"]:
                            fail = {**fail, "split": split}
                            failures.append(fail)
                            attempts_failed[fail["task"]][fail["mode"]] += 1
                            budget = fail_budget * max(quota[fail["task"]][fail["mode"]], 1)
                            if fail["mode"] == "complex" and \
                                    attempts_failed[fail["task"]]["complex"] >= budget:
                                closed[fail["task"]]["complex"] = True
                if targets_met():
                    break
                while len(pending) < self.concurrency:
                    fut = submit_next(executor)
                    if fut is None:
                        break
                    pending.add(fut)

            for fut in pending:
                fut.cancel()

        print(f"[{split}] done — " + ", ".join(f"{k}: {have[k]}/{want[k]}" for k in self.base))

    # ------------------------------------------------- controlled schema reuse (scale-only)

    def _reuse_pool(self):
        """The ORIGINAL eligible training pool: the seed/ratio partition of all eligible schemas, train side,
        minus every schema whose id or content hash appears in the published TEST split, minus content
        duplicates inside the pool. Published train schemas stay in (they may be reused)."""
        if self.reuse.pool_source == "published_train":
            return self._anchored_pool()
        records = self._load_schema_records(apply_exclusions=False)
        partition = self._partition(records)
        train = sorted(partition["train"], key=lambda r: self._safe_id(r[1], r[2], r[3]))
        sid = lambda r: self._safe_id(r[1], r[2], r[3])
        with open(self.reserved_info["file"], "r", encoding="utf-8") as f:
            res = json.load(f)
        test_ids, test_hashes = set(res["ids"].get("test", [])), set(res["schema_hashes"].get("test", []))
        train_ids = set(res["ids"].get("train", []))
        in_pool = {sid(r) for r in train}
        found = len(train_ids & in_pool)
        fraction = found / len(train_ids) if train_ids else 1.0
        stats = {"eligible_schemas": len(records), "original_test_pool": len(partition["test"]),
                 "original_train_pool": len(train),
                 "published_train_ids": len(train_ids), "published_train_ids_found_in_pool": found,
                 "published_test_ids_found_in_train_pool": len(test_ids & in_pool)}
        if train_ids and fraction < self.reuse.pool_sanity_min_fraction:
            raise ValueError(
                f"Only {found}/{len(train_ids)} published train schemas are in the reconstructed training pool "
                f"({fraction:.0%}). The schema corpus (data/schemas, data/jsb_schemas), seed or test_ratio do not "
                "match the original build; refusing to continue.")
        kept, by_id, by_hash = [], 0, 0
        for r in train:
            if sid(r) in test_ids:
                by_id += 1
            elif schema_hash(r[0]) in test_hashes:
                by_hash += 1
            else:
                kept.append(r)
        stats.update(removed_published_test_by_id=by_id, removed_published_test_by_content_hash=by_hash)
        dup = 0
        if self.reuse.dedupe_pool_by_content:
            seen, unique = set(), []
            for r in kept:
                h = schema_hash(r[0])
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                unique.append(r)
            kept = unique
        stats.update(removed_content_duplicates_in_pool=dup, usable_schemas=len(kept))
        by_source = {}
        for r in kept:
            key = "/".join(p for p in (r[1], r[2]) if p) or "unknown"
            by_source[key] = by_source.get(key, 0) + 1
        stats["usable_by_source"] = dict(sorted(by_source.items()))
        self._pool_stats = {**(self._pool_stats or {}), **stats}
        return kept

    def _anchor_path(self):
        return self.reuse.anchor_file or os.path.join(os.path.dirname(os.path.abspath(self.reserved_info["file"])),
                                                       ANCHOR_FILENAME)

    def _anchored_pool(self):
        """pool_source 'published_train': the schemas of the published TRAINING rows (ids, content and hashes as
        published), minus those that overlap the published test set by id or content hash, minus the explicit size
        restriction, minus content duplicates. Replaces only the reconstructed-pool compatibility check: instead of
        "most published ids are in a rebuilt pool", the anchor must agree EXACTLY with reserved_schemas.json."""
        try:
            anchor = load_anchor(self._anchor_path(), self.reserved_info["file"])
        except AnchorError as e:
            raise ValueError(f"published_train pool refused: {e}") from None
        kept, stats = select_pool(anchor, self.reuse.anchor_max_schema_kb, self.reuse.dedupe_pool_by_content)
        stats["pool_source"] = "published_train"
        stats["anchor_file"] = self._anchor_path()
        self._pool_stats = {**(self._pool_stats or {}), **stats}
        return [(e["schema"], e["source"], e["subset"], e["name"]) for e in kept]

    def reuse_capacity_report(self, assumed_acceptance=0.7):
        """Can the targets be met from the original training pool under the retry policy? (no model needed)"""
        pool = self._reuse_pool()
        P, rs = len(pool), self.reuse
        with open(self.reserved_info["file"], "r", encoding="utf-8") as f:
            res = json.load(f)
        pool_ids = {self._safe_id(r[1], r[2], r[3]) for r in pool}
        targets = dict(self.counts_by_split["train"])
        max_refs = P * rs.max_new_references_per_schema
        published = {t: len(pool_ids & set(res.get("ids_by_task", {}).get(t, {}).get("train", []))) for t in targets}
        per_task = {}
        for t, target in targets.items():
            per_task[t] = {"target": target, "upper_bound_if_every_attempt_succeeds": max_refs,
                           "required_average_acceptance": round(target / max_refs, 3) if max_refs else None,
                           "pool_schemas_with_a_published_example_for_this_task": published[t],
                           "feasible_at_assumed_acceptance":
                               int(max_refs * assumed_acceptance) >= target}
        refs_needed = max((int(-(-t // assumed_acceptance)) for t in targets.values()), default=0)
        refs_needed = min(refs_needed, max_refs)
        low, extra = divmod(refs_needed, P) if P else (0, 0)
        pub_counts = {}
        for sid in pool_ids:
            n = sum(1 for t in targets if sid in set(res.get("ids_by_task", {}).get(t, {}).get("train", [])))
            pub_counts[n] = pub_counts.get(n, 0) + 1
        return {
            "pool": self._pool_stats, "settings": rs.as_dict(), "assumed_acceptance_per_attempt": assumed_acceptance,
            "max_new_references": max_refs, "per_task": per_task,
            "references_to_mint_at_assumed_acceptance": refs_needed,
            "expected_new_references_per_schema": {
                "schemas_with_%d" % (low + 1): extra, "schemas_with_%d" % low: P - extra},
            "expected_new_examples_total": {t: min(targets[t], int(refs_needed * assumed_acceptance)) for t in targets},
            "published_train_examples_per_pool_schema_histogram": dict(sorted(pub_counts.items())),
            "note": "Balanced scheduling gives every schema its k-th new reference before any schema gets its (k+1)-th, "
                    "schemas with fewer published+new references first. The real acceptance rate of repeated attempts "
                    "is unknown until a smoke test; 'required_average_acceptance' is the assumption-free number."}

    def _chunk_restore(self, ledger, dup, by_id):
        """Continues the run from the verified parent chunk: the ledger (all per-schema limits) and the duplicate index."""
        ch = self.chunk
        problems = []
        if ch["index"] < 1 or ch["index"] > ch["n_chunks"]:
            problems.append("chunk index outside 1..n_chunks")
        if not ch.get("parent_dir"):
            if ch["index"] != 1:
                problems.append("only chunk 1 may start without a parent chunk")
            if problems:
                raise ValueError("chunk refused: " + "; ".join(problems))
            return
        chain = chunking.load_chain(ch["parent_dir"])
        first = chain[0][1]
        if chain[-1][1]["chunk_index"] != ch["index"] - 1:
            problems.append(f"parent is chunk {chain[-1][1]['chunk_index']}, expected {ch['index'] - 1}")
        expect = {"anchor_sha256": (self._pool_stats or {}).get("anchor_sha256"), "pool_fingerprint": self._chunk_pool_fp,
                  "config_sha256": ch["config_sha256"], "totals": ch["totals"], "n_chunks": ch["n_chunks"], "seed": self.seed}
        for key, value in expect.items():
            if first[key] != value:
                problems.append(f"{key} differs from the parent chain")
        if problems:
            raise ValueError("chunk refused: " + "; ".join(problems))
        chunking.restore_ledger(ledger, chain)
        registered = 0
        for cdir, state in chain:    # every earlier example is a known reference/content for duplicate control
            for task in chunking.TASKS:
                for name in state["files"][task]:
                    with open(os.path.join(cdir, f"benchmark_{task}", "train", name), "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    sid = name[:-5].rpartition("__a")[0]
                    dup.register_reference(by_id[sid][4], reference_hash(task, rec), sid)
                    dup.register_task(task, rec, sid)
                    registered += 1
        self._chunk_carried = {"parent_dir": ch["parent_dir"], "examples_registered": registered,
                               "chain": [d for d, _ in chain]}

    def chunk_state(self, out_dir, parent_state_sha=None, parent_dir_name=None):
        """The state to checkpoint after a chunk that met its quota (called by build_benchmark.py)."""
        st = getattr(self, "_reuse_state", None)
        if not self.chunk or not st:
            raise ValueError("no chunked reuse run to checkpoint")
        ch = self.chunk
        files = chunking.collect_files(out_dir)
        return {
            "chunk_index": ch["index"], "n_chunks": ch["n_chunks"], "totals": ch["totals"],
            "quota": {t: self.counts_by_split["train"].get(t, 0) for t in self.base},
            "anchor_sha256": (self._pool_stats or {}).get("anchor_sha256"),
            "pool_fingerprint": self._chunk_pool_fp, "config_sha256": ch["config_sha256"], "seed": self.seed,
            "parent": ({"dir": parent_dir_name, "state_sha256": parent_state_sha} if parent_dir_name else None),
            "passes_end": st["passes"], "ledger": {sid: st["ledger"].entries[sid] for sid in st["ids"]},
            "events": st["ledger"].events, "files": files,
            "counts": {t: len(files[t]) for t in files}}

    def _reuse_scan_existing(self, split_dir, ledger, dup, by_id):
        """Resume: registers examples already on disk (references, contents, per-schema counts)."""
        have = {k: 0 for k in self.base}
        attempts_seen = {}
        for task in self.base:
            for name in sorted(os.listdir(split_dir[task])):
                if not name.endswith(".json"):
                    continue
                stem = name[:-5]
                sid, _, att = stem.rpartition("__a")
                if sid not in by_id or not att.isdigit():
                    print(f"[reuse] ignoring unexpected file {task}/{name}")
                    continue
                with open(os.path.join(split_dir[task], name), "r", encoding="utf-8") as f:
                    rec = json.load(f)
                shash = by_id[sid][4]
                dup.register_reference(shash, reference_hash(task, rec), sid)
                dup.register_task(task, rec, sid)
                have[task] += 1
                attempts_seen.setdefault(sid, {}).setdefault(int(att), []).append(task)
        for sid, atts in attempts_seen.items():
            e = ledger.entries[sid]
            e["attempts"] = max(atts)
            e["accepted_references"] = len(atts)
            for tasks in atts.values():
                for t in tasks:
                    e["examples"][t] += 1
        return have

    def _build_split_reuse(self, split, pool, llm_provider):
        rs = self.reuse
        want = {k: self.counts_by_split[split].get(k, 0) for k in self.base}
        gates = self.gates_by_split[split]
        split_dir = {k: os.path.join(self.base[k], split) for k in self.base}
        for d in split_dir.values():
            os.makedirs(d, exist_ok=True)

        by_id = {}
        for schema, source, subset, origin in pool:
            by_id[self._safe_id(source, subset, origin)] = (schema, source, subset, origin, schema_hash(schema))
        ids = sorted(by_id)
        ledger = SchemaLedger(rs, self._published_refs)
        for sid in ids:
            ledger.entry(sid, by_id[sid][4])
        dup = DuplicateIndex(self._published_refs, self._published_keys)
        self._chunk_pool_fp = chunking.pool_fingerprint(by_id)
        if self.chunk:
            self._chunk_restore(ledger, dup, by_id)
        have = self._reuse_scan_existing(split_dir, ledger, dup, by_id)
        had_before = dict(have)
        for k in self.base:
            if have[k]:
                print(f"[{split}/{k}] resuming from {have[k]}/{want[k]}")

        failures, metas, cross_dups = [], [], []
        failed_attempts = {k: {"standard": 0, "complex": 0} for k in self.base}
        report = {"quota": {k: {"standard": want[k], "complex": 0} for k in self.base},
                  "had_before": {k: {"standard": had_before[k], "complex": 0} for k in self.base},
                  "have_mode": {k: {"standard": have[k], "complex": 0} for k in self.base},
                  "failed_attempts": failed_attempts,
                  "closed": {k: {"standard": False, "complex": False} for k in self.base},
                  "failures": failures, "metas": metas, "llm_calls": 0, "duplicates": cross_dups}
        self._split_reports[split] = report
        self._reuse_state = {"ledger": ledger, "dup": dup, "ids": ids, "split": split, "want": want, "passes": 0}

        lock = threading.Lock()
        reserved = {k: 0 for k in self.base}
        state = self._reuse_state

        def schedule():
            pass_no = 0
            while ledger.live(ids):
                pass_no += 1
                state["passes"] = pass_no
                order = ledger.order_for_pass(ids, pass_no, random.Random(f"{self.seed}-{split}-pass{pass_no}"
                                                                                  + (f"-chunk{self.chunk['index']}" if self.chunk else "")))
                for sid in order:
                    yield pass_no, sid

        sched = schedule()

        def open_tasks():
            return {k: "standard" for k in self.base if want[k] - have[k] - reserved[k] > 0}

        def targets_met():
            return all(have[k] >= want[k] for k in self.base)

        inflight = {}

        def submit_next(executor):
            for pass_no, sid in sched:
                if not ledger.eligible(sid, pass_no):
                    continue
                with lock:
                    needed = open_tasks()
                    if not needed:
                        return None
                    for k in needed:
                        reserved[k] += 1
                schema, source, subset, origin, shash = by_id[sid]
                attempt = ledger.begin_attempt(sid)
                full_id = f"{sid}__a{attempt}"
                seed_label = f"{self.seed}-{split}-{sid}-a{attempt}"
                call_params = None
                if rs.sampling_seed_param:
                    call_params = {rs.sampling_seed_param: int(hashlib.sha1(seed_label.encode()).hexdigest()[:8], 16)}

                def reject(reference, _shash=shash):
                    kind = dup.reference_kind(_shash, canon_hash(reference))
                    if kind:
                        dup.note_rejected_reference(kind)
                    return kind

                fut = executor.submit(self._process_schema, schema, source, subset, origin, needed,
                                      llm_provider, random.Random(seed_label), gates, split, seed_label,
                                      instance_id=full_id, reject_reference=reject, call_params=call_params)
                inflight[fut] = (needed, sid, attempt, pass_no, full_id)
                return fut
            return None

        llm_calls = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            pending = set()
            while len(pending) < self.concurrency:
                fut = submit_next(executor)
                if fut is None:
                    break
                pending.add(fut)
            while pending:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    needed, sid, attempt, pass_no, full_id = inflight.pop(fut)
                    shash = by_id[sid][4]
                    try:
                        result = fut.result()
                    except Exception as e:   # one bad schema must not stop the run
                        print(f"[{split}] worker error (skipped): {e}")
                        result = {"instance_id": full_id, "produced": {}, "meta": {}, "llm_calls": 0,
                                  "reference": None,
                                  "failures": [{"task": t, "mode": m, "schema_id": full_id, "stage": "worker_error",
                                                "detail": str(e)} for t, m in needed.items()]}
                    with lock:
                        for k in needed:
                            reserved[k] -= 1
                        llm_calls += result.get("llm_calls", 0)
                        report["llm_calls"] = llm_calls
                        accepted, quota_full = [], 0
                        local_fails = [dict(f) for f in result["failures"]]
                        reference = result.get("reference")
                        rh = canon_hash(reference) if reference is not None else None
                        produced = result["produced"]
                        if produced and not dup.register_reference(shash, rh, sid):
                            for task in produced:   # a concurrent attempt on this schema made the same reference
                                local_fails.append({"task": task, "mode": "standard", "schema_id": full_id,
                                                    "stage": "duplicate_reference_at_commit", "detail": None})
                            produced = {}
                        for task, record in produced.items():
                            if have[task] >= want[task]:
                                quota_full += 1
                                continue
                            verdict = dup.check_task(task, record, rs.reject_cross_schema_duplicates)
                            if verdict:
                                local_fails.append({"task": task, "mode": "standard", "schema_id": full_id,
                                                    "stage": f"duplicate_{verdict}", "detail": None})
                                continue
                            path = os.path.join(split_dir[task], full_id + ".json")
                            self._write(path, record)
                            dup.register_task(task, record, sid)
                            have[task] += 1
                            report["have_mode"][task]["standard"] = have[task]
                            accepted.append(task)
                            meta = result["meta"].get(task)
                            if meta is not None:
                                meta.update(split=split, schema_id=sid, attempt=attempt, reference_hash=rh,
                                            schema_hash=shash, reuse=True)
                                os.makedirs(os.path.dirname(self._meta_path(task, split, full_id)), exist_ok=True)
                                self._write(self._meta_path(task, split, full_id), meta)
                                metas.append(meta)
                            print(f"[{split}/{task}] {have[task]}/{want[task]} ({full_id})")
                        for fail in local_fails:
                            fail = {**fail, "split": split, "schema": sid, "attempt": attempt}
                            failures.append(fail)
                            failed_attempts[fail["task"]]["standard"] += 1
                        if accepted or not quota_full:
                            why = None
                            if not accepted:
                                stages = sorted({f["stage"] for f in local_fails})
                                ref = result.get("reference_attempts") or {}
                                why = ",".join(stages + list(dict.fromkeys(ref.get("reasons") or []))[:3])
                            event = ledger.finish_attempt(sid, accepted, pass_no, why)
                            if result.get("schema_ineligible") and not accepted:
                                # its own prompt cannot fit the window: no later attempt can differ
                                event = ledger.retire(sid, "context_overflow: the schema's own prompt does not fit the "
                                                           "model context window", pass_no) or event
                            if event:
                                print(f"[{split}] schema {sid}: {event['event']}"
                                      + (f" ({event.get('reason')})" if event.get("reason") else ""))
                if targets_met():
                    break
                while len(pending) < self.concurrency:
                    fut = submit_next(executor)
                    if fut is None:
                        break
                    pending.add(fut)
            for fut in pending:
                fut.cancel()
        print(f"[{split}] done - " + ", ".join(f"{k}: {have[k]}/{want[k]}" for k in self.base))

    def _reuse_summary(self):
        """Per-schema / per-task distribution, skipped schemas and duplicate statistics."""
        st = getattr(self, "_reuse_state", None)
        if not st:
            return None
        ledger, dup = st["ledger"], st["dup"]
        rows = [ledger.entries[i] for i in st["ids"]]
        used = [r for r in rows if r["accepted_references"] > 0]

        def hist(values):
            out = {}
            for v in values:
                out[v] = out.get(v, 0) + 1
            return dict(sorted(out.items()))

        status = {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
        totals = {t: sum(r["examples"][t] for r in rows) for t in self.base}
        per_schema_total = [sum(r["examples"].values()) for r in rows]
        return {
            "settings": self.reuse.as_dict(), "passes_started": st["passes"],
            "schemas_in_pool": len(rows), "schemas_used": len(used),
            "schema_status": status,
            "new_references_per_schema_histogram": hist(r["accepted_references"] for r in rows),
            "new_examples_per_schema_histogram": hist(per_schema_total),
            "new_examples_per_task": totals,
            "new_examples_per_task_and_schema_max": {t: max((r["examples"][t] for r in rows), default=0) for t in self.base},
            "attempts_total": sum(r["attempts"] for r in rows),
            "failed_attempts_total": sum(r["failed_attempts"] for r in rows),
            "duplicates": dup.stats,
            "skipped_or_capped_events": ledger.events,
            "schemas_with_published_examples_reused": sum(1 for r in used if r["published_references"] > 0),
            "mode": "standard (original-style) only"}

    def _process_schema(self, schema, source, subset, origin, needed, llm_provider, rng, gates,
                        split=None, seed_label=None, instance_id=None, reject_reference=None,
                        call_params=None):
        """Builds the needed task records for one schema.

        `needed` maps task -> "standard"/"complex" (a plain collection of task names means
        all-standard, the legacy shape). Returns a dict:
        {"instance_id", "produced": {task: record}, "meta": {task: sidecar}, "failures": [...],
         "llm_calls"}. A task that is neither produced nor failed explicitly never happens:
        every needed task ends up in exactly one of the two.
        """
        if not isinstance(needed, dict):
            needed = {task: "standard" for task in needed}
        instance_id = instance_id or self._safe_id(source, subset, origin)
        seed_label = seed_label or "0"
        llm = _CountingLLM(llm_provider, sink=lambda kind, detail: self._log_event(
            split, instance_id, kind, detail), call_params=call_params)
        tags = {"name": origin, "source": source, "subset": subset, "schema": schema}
        cfg = self.settings.complex
        produced, meta, failures = {}, {}, []
        refs, ref_info, descs = {}, {}, {}
        active = dict(needed)

        def fail(task, stage, detail=None):
            failures.append({"task": task, "mode": needed[task], "schema_id": instance_id,
                             "stage": stage, "detail": detail})

        # Complex Create needs a schema that can hold a rich instance; check before spending calls.
        create_thresholds = None
        if active.get("create") == "complex":
            ok, why = schema_can_support(schema, cfg["create"])
            if ok:
                create_thresholds = cfg["create"]
            else:
                fail("create", "schema_cannot_support_complexity", why)
                del active["create"]

        def reference_for(mode):
            if mode not in refs:
                info = {}
                if mode == "standard":
                    refs[mode] = self._mint_reference(schema, llm, gates, diag=info, reject=reject_reference)
                else:
                    refs[mode] = self._mint_reference(schema, llm, gates, complexity_hint=True,
                                                      min_complexity=create_thresholds, diag=info)
                ref_info[mode] = info
            return refs[mode]

        ineligible = []

        def ref_fail(task, mode):
            info = ref_info[mode]
            if info.get("stage") == "context_overflow":
                if info.get("permanent_context_overflow"):
                    ineligible.append("context_overflow")
                fail(task, "context_overflow", info)
            else:
                fail(task, "reference_failed", info)

        def base_meta(task, mode):
            inner = getattr(llm_provider, "inner", llm_provider)
            return {"task": task, "mode": mode, "schema_id": instance_id, "seed": seed_label,
                    "model": getattr(llm_provider, "model", None), "provider": type(inner).__name__,
                    "attempts": {"reference": ref_info.get(mode)}, "failure_reason": None}

        if "create" in active:
            mode = active["create"]
            reference = reference_for(mode)
            if reference is None:
                ref_fail("create", mode)
            else:
                diag = {}
                try:
                    description = self._make_description(schema, reference, llm, gates, diag=diag)
                except Exception as e:
                    if not _is_context_overflow(e):
                        raise
                    description = None
                    diag.update(stage="context_overflow", context_overflow=_overflow_info(e))
                descs[mode] = description
                if description:
                    produced["create"] = {**tags, "description": description,
                                          "reference_json": reference}
                    m = base_meta("create", mode)
                    m["attempts"]["description"] = diag
                    m["complexity"] = {"reference": reference_metrics(reference, schema)}
                    if mode == "complex":
                        m["complexity"]["thresholds"] = cfg["create"]
                    ratio = _grounding_ratio(reference, description)
                    m["validation"] = {
                        "reference_schema_valid": metrics.schema_compliance(reference, schema)[0] == 1,
                        "grounding_ratio": round(ratio, 4),
                        "grounding_threshold": gates["grounding_threshold"],
                        "grounding_passed": ratio >= gates["grounding_threshold"],
                        "round_trip_enabled": gates["round_trip"],
                    }
                    m["requires_manual_review"] = [
                        "description conveys every value faithfully (grounding only checks literal "
                        "string overlap)"]
                    meta["create"] = m
                else:
                    fail("create", "context_overflow" if diag.get("stage") == "context_overflow" else "description_failed", diag)

        if "fix" in active:
            mode = active["fix"]
            reference = reference_for(mode)
            if reference is None:
                ref_fail("fix", mode)
            else:
                fix_rng = random.Random(f"{seed_label}-fix")
                erroneous, complexity_info, validation = None, None, None
                if mode == "standard":
                    candidate, info = _inject_error_info(reference, schema, fix_rng)
                    if candidate is None:
                        fail("fix", "no_applicable_mutation")
                    elif metrics.schema_compliance(candidate, schema)[0] != 0:
                        # Keep only mutations that genuinely break schema compliance.
                        fail("fix", "mutation_did_not_break_schema", info)
                    else:
                        erroneous = candidate
                        complexity_info = single_fix_metrics(reference, erroneous, info)
                        validation = {"original_valid": True, "corrupted_invalid": True}
                else:
                    fcfg = cfg["fix"]
                    found = inject_errors(reference, schema, fix_rng,
                                          n_errors=(fcfg["min_errors"], fcfg["max_errors"]),
                                          attempts=fcfg["attempts"])
                    if found is None:
                        fail("fix", "no_compatible_error_combination")
                    else:
                        candidate, errors = found
                        check = verify_fix(reference, candidate, errors, schema)
                        if not check["ok"]:
                            fail("fix", "verification_failed", check["problems"])
                        else:
                            erroneous = candidate
                            complexity_info = fix_metrics(reference, erroneous, errors)
                            complexity_info["errors"] = errors
                            validation = check
                overflow = False
                if erroneous is not None:
                    try:
                        fix_desc = descs.get(mode) or self._describe(schema, reference, llm)
                    except Exception as e:
                        if not _is_context_overflow(e):
                            raise
                        fail("fix", "context_overflow", _overflow_info(e))
                        overflow, fix_desc = True, None
                if erroneous is not None and not overflow:
                    record = {**tags, "erroneous_json": erroneous, "valid_json": reference}
                    if fix_desc:
                        record["description"] = fix_desc
                    produced["fix"] = record
                    m = base_meta("fix", mode)
                    m["complexity"] = {"errors": complexity_info,
                                       "reference": reference_metrics(reference, schema)}
                    m["validation"] = validation
                    m["requires_manual_review"] = ["the injected errors are realistic for the domain"]
                    meta["fix"] = m

        if "modify" in active:
            mode = active["modify"]
            reference = reference_for(mode)
            if reference is None:
                ref_fail("modify", mode)
            else:
                # Standard keeps the original rng stream; complex gets its own derived stream.
                mod_rng = rng if mode == "standard" else random.Random(f"{seed_label}-modify")
                diag = {}
                try:
                    mod = self._make_modification(schema, reference, llm, mod_rng, gates,
                                                  mode=mode, diag=diag)
                except Exception as e:
                    if not _is_context_overflow(e):
                        raise
                    mod = None
                    diag.update(stage="context_overflow", context_overflow=_overflow_info(e))
                if mod is None:
                    fail("modify", diag.get("stage", "modification_failed"),
                         {k: v for k, v in diag.items() if k != "stage"})
                else:
                    modified, instruction, spec = mod
                    produced["modify"] = {**tags, "data": reference, "instructions": instruction,
                                          "ground_truth": modified, "modification": spec}
                    m = base_meta("modify", mode)
                    m["complexity"] = {"operations": modify_metrics(reference, modified, spec),
                                       "reference": reference_metrics(reference, schema)}
                    m["validation"] = diag.get("verification")
                    m["attempts"]["instruction"] = {k: v for k, v in diag.items() if k != "verification"}
                    m["requires_manual_review"] = diag.get("verification", {}).get("manual_review", [])
                    meta["modify"] = m

        for m in meta.values():
            m["builder_llm_calls_for_schema"] = llm.calls
        return {"instance_id": instance_id, "produced": produced, "meta": meta,
                "failures": failures, "llm_calls": llm.calls, "reference": refs.get("standard"),
                "reference_attempts": ref_info.get("standard"),
                "schema_ineligible": ineligible[0] if ineligible else None}

    # --------------------------------------------------------------- run report

    def _event_summary(self):
        counts = {}
        for event in self._events:
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
        return {"counts": dict(sorted(counts.items())),
                "note": "provider_error = the model call raised; suspected_truncation = JSON-looking "
                        "answer that never closed or an unclosed <think> block (heuristic); "
                        "empty_response = blank answer. Details in generation_events.jsonl."}

    def _write_report(self, llm_provider):
        """Writes generation_summary.json and generation_failures.jsonl for this run."""
        report_dir = self.settings.report_dir or os.path.join(
            self.data_root, "_generation_reports", self._run_started.strftime("%Y%m%d_%H%M%S"))
        report_dir = os.path.abspath(report_dir)
        os.makedirs(report_dir, exist_ok=True)

        requested, on_disk, new_this_run, failed = {}, {}, {}, {}
        duplicates = []
        shortfall, failure_stages, check_failures = [], {}, {}
        complexity_rows = {}
        failures_all, llm_calls = [], 0
        for split, rep in self._split_reports.items():
            llm_calls += rep["llm_calls"]
            duplicates += rep["duplicates"]
            requested[split] = rep["quota"]
            on_disk[split] = rep["have_mode"]
            new_this_run[split] = {t: {m: rep["have_mode"][t][m] - rep["had_before"][t][m]
                                       for m in MODES} for t in rep["have_mode"]}
            failed[split] = rep["failed_attempts"]
            for task, modes in rep["quota"].items():
                for mode, want in modes.items():
                    got = rep["have_mode"][task][mode]
                    if got < want:
                        shortfall.append({
                            "split": split, "task": task, "mode": mode, "requested": want,
                            "produced": got,
                            "reason": "failure budget exhausted" if rep["closed"][task][mode]
                            else "schema pool exhausted or run stopped early"})
            for fail in rep["failures"]:
                key = f"{fail['task']}/{fail['mode']}/{fail['stage']}"
                failure_stages[key] = failure_stages.get(key, 0) + 1
                failures_all.append(fail)
            for meta in rep["metas"]:
                for name, ok in (meta.get("validation") or {}).get("checks", {}).items():
                    if ok is False:
                        key = f"{meta['task']}/{meta['mode']}/{name}"
                        check_failures[key] = check_failures.get(key, 0) + 1
                complexity_rows.setdefault((meta["task"], meta["mode"]), []).append(meta["complexity"])

        def stats(values):
            values = [v for v in values if isinstance(v, (int, float))]
            if not values:
                return None
            return {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 2)}

        complexity = {}
        for (task, mode), rows in sorted(complexity_rows.items()):
            ref = [r["reference"] for r in rows]
            entry = {"examples": len(rows),
                     "reference_depth": stats([r["depth"] for r in ref]),
                     "reference_leaves": stats([r["leaf_count"] for r in ref])}
            ops = [r.get("operations") or r.get("errors") for r in rows]
            if ops[0] is not None:
                entry.update(operations_or_errors=stats([o["count"] for o in ops]),
                             max_depth=stats([o["max_depth"] for o in ops]),
                             affected_fields=stats([o["affected_fields"] for o in ops]),
                             different_branches=sum(1 for o in ops if o["different_branches"]),
                             inside_array_items=sum(1 for o in ops if o["inside_array_items"]))
            complexity[f"{task}/{mode}"] = entry

        def total(table, mode=None):
            return sum(n for split in table.values() for modes in split.values()
                       for m, n in modes.items() if mode in (None, m))

        summary = {
            "completed": self._run_completed,
            "started_at": self._run_started.isoformat(timespec="seconds"),
            "seed": self.seed,
            "model": getattr(llm_provider, "model", None),
            "validator_model": getattr(getattr(self, "_validator", None), "model", None),
            "generation": {
                "standard_ratio": float(self.settings.standard_ratio),
                "complex_ratio": float(self.settings.complex_ratio),
                "standard_modify_strategy": self.settings.standard_modify_strategy,
                "complex": self.settings.complex,
            },
            "requested": requested,
            "produced_on_disk": on_disk,
            "produced_this_run": new_this_run,
            "failed_attempts": failed,
            "totals": {
                "requested": total(requested), "requested_standard": total(requested, "standard"),
                "requested_complex": total(requested, "complex"),
                "produced_on_disk": total(on_disk), "produced_standard": total(on_disk, "standard"),
                "produced_complex": total(on_disk, "complex"),
                "failed_attempts": total(failed),
            },
            "shortfall": shortfall,
            "schema_pool": {**(self._pool_stats or {}), "partition_sizes": self._partition_sizes,
                            "reserved_from": self.reserved_info},
            "llm_events": self._event_summary(),
            "schema_reuse": self._reuse_summary(),
            "duplicates": {"count": len(duplicates), "examples": duplicates[:20],
                           "note": "same content as an earlier example of this run (different "
                                   "schema); detection only - nothing is removed"},
            "failure_reasons": dict(sorted(failure_stages.items())),
            "capacity_exclusions": {
                "context_overflow": [
                    {"schema_id": f.get("schema_id"), "task": f.get("task"),
                     "detail": ((f.get("detail") or {}).get("context_overflow") if isinstance(f.get("detail"), dict)
                                else f.get("detail"))}
                    for f in failures_all if f.get("stage") == "context_overflow"],
                "note": "requests that cannot fit the model context window; never sent, never retried, not provider "
                        "errors (see src/llm/context_guard.py)"},
            "validation_check_failures": dict(sorted(check_failures.items())),
            "complexity": complexity,
            "builder_llm_calls": llm_calls,
            "builder_llm_calls_note": "calls made by the builder model while processing schemas; "
                                      "validator-model (round-trip) calls are not included",
            "outputs": {
                "datasets": {task: self.base[task] for task in self.base},
                "metadata_sidecars": {task: os.path.join(self.base[task], "_metadata") for task in self.base},
                "report_dir": report_dir,
                "failures_file": os.path.join(report_dir, "generation_failures.jsonl"),
                "events_file": os.path.join(report_dir, "generation_events.jsonl"),
            },
        }
        with open(os.path.join(report_dir, "generation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(os.path.join(report_dir, "generation_failures.jsonl"), "w", encoding="utf-8") as f:
            for fail in failures_all:
                f.write(json.dumps(fail, default=str) + "\n")
        with open(os.path.join(report_dir, "generation_events.jsonl"), "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event, default=str) + "\n")
        reuse_state = getattr(self, "_reuse_state", None)
        if reuse_state:   # per-schema ledger and the schemas that were suspended / retired / capped
            with open(os.path.join(report_dir, "schema_ledger.jsonl"), "w", encoding="utf-8") as f:
                for sid in reuse_state["ids"]:
                    f.write(json.dumps(reuse_state["ledger"].entries[sid], default=str) + "\n")
            with open(os.path.join(report_dir, "skipped_schemas.jsonl"), "w", encoding="utf-8") as f:
                for event in reuse_state["ledger"].events:
                    f.write(json.dumps(event, default=str) + "\n")
        self.last_summary = summary
        print(f"[report] summary written to {report_dir}")

    def _write_manifest(self, base, task_label, llm_provider):
        """Writes a provenance manifest reflecting the dataset's on-disk state.

        Recounts the files per split (rather than trusting in-loop counters) so the
        manifest stays accurate across resumed/topped-up builds.
        """
        splits_info = {}
        for split in ("test", "train"):
            split_dir = os.path.join(base, split)
            if not os.path.isdir(split_dir):
                continue
            total = 0
            by_origin = {}
            for fn in os.listdir(split_dir):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(split_dir, fn), "r", encoding="utf-8") as f:
                        rec = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                total += 1
                key = "/".join(p for p in (rec.get("source"), rec.get("subset")) if p) or "unknown"
                by_origin[key] = by_origin.get(key, 0) + 1
            if total:
                splits_info[split] = {"total": total, "by_source": dict(sorted(by_origin.items()))}

        manifest = {
            "task": task_label,
            "splits": splits_info,
            "schema_partition": {"test_ratio": self.test_ratio, "schemas_per_split": self._partition_sizes},
            "builder_model": getattr(llm_provider, "model", None),
            "validator_model": getattr(getattr(self, "_validator", None), "model", None),
            "schema_sources": self.sources,
            "seed": self.seed,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": (
                "Generated benchmark ground truth. test/ and train/ are drawn from "
                "disjoint schema pools (no contamination). The build is not "
                "reproducible (LLM-authored); this frozen dataset is the artifact."
            ),
            "attribution": _JSB_ATTRIBUTION,
        }
        with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        counts = {s: i["total"] for s, i in splits_info.items()}
        print(f"[{task_label}] manifest written ({counts})")

    # ---------------------------------------------------------- reference minting

    def _mint_reference(self, schema, llm_provider, gates, complexity_hint=False,
                        min_complexity=None, diag=None, reject=None):
        """Generates a schema-valid, non-trivial reference object, or None.

        Shared starting point for all three tasks. Retries up to
        `max_build_attempts` times to clear the non-triviality gate.

        Complex mode (off by default - the standard prompt and gates are untouched):
        `complexity_hint` asks the model for a richer object, and `min_complexity`
        (thresholds dict) additionally requires the result to be rich enough. `diag`, when
        given, receives {"attempts", "reasons"} describing why attempts were rejected.
        """
        reasons = []
        for _ in range(gates["max_build_attempts"]):
            initial = self.task._build_schema_creation_prompt(schema, complexity_hint=True) \
                if complexity_hint else self.task._build_schema_creation_prompt(schema)
            try:
                result = self.task._run_fix_loop(schema, initial, llm_provider, self.max_retries)
            except Exception as e:
                if not _is_context_overflow(e):
                    raise
                # The request was never sent. If it is the schema's own first prompt, every attempt would be
                # identical, so stop at once and mark the schema as unusable; otherwise (a repair prompt that has
                # grown too long) the next build attempt starts again from the original prompt.
                reasons.append("context_overflow")
                if diag is not None:
                    diag["context_overflow"] = _overflow_info(e)
                if e.n_messages == len(initial):
                    if diag is not None:
                        diag["permanent_context_overflow"] = True
                    break
                continue
            if result is None:
                reasons.append("invalid_json_after_retries")
                continue
            try:
                reference_json = json.loads(result[0])
            except json.JSONDecodeError:
                reasons.append("unparseable")
                continue
            if gates["deterministic_gates"] and \
                    _count_nontrivial(reference_json) < gates["min_populated_fields"]:
                reasons.append("trivial")
                continue
            if reject is not None:
                kind = reject(reference_json)   # reuse mode: same reference already known for this schema
                if kind:
                    reasons.append(f"duplicate_reference_{kind}")
                    continue
            if min_complexity is not None:
                ok, why = meets_create_thresholds(reference_json, schema, min_complexity)
                if not ok:
                    reasons.append("below_complexity: " + "; ".join(why))
                    continue
            if diag is not None:
                diag.update(attempts=len(reasons) + 1, reasons=reasons)
            return reference_json
        if diag is not None:
            diag.update(attempts=len(reasons), reasons=reasons, stage=reasons[-1] if reasons else None)
        return None

    # ----------------------------------------------------------------- create 2a

    def _prefixed(self, messages, prefix=None):
        """Prepends a reasoning/system prefix (builder's by default)."""
        prefix = self.system_prompt_prefix if prefix is None else prefix
        if not prefix:
            return messages
        if messages and messages[0]["role"] == "system":
            head = {**messages[0], "content": prefix + "\n" + messages[0]["content"]}
            return [head] + messages[1:]
        return [{"role": "system", "content": prefix}] + messages

    def _describe(self, schema, reference_json, llm_provider):
        """One raw natural-language description of the reference object."""
        prompt = self._prefixed(self.task._build_source_doc_prompt(schema, reference_json))
        description, _ = llm_provider.generate(prompt)
        return (description or "").strip()

    def _make_description(self, schema, reference_json, llm_provider, gates, diag=None):
        """A description that clears the grounding + round-trip gates, or None.

        `diag`, when given, receives {"attempts", "reasons", "groundings"} - why each
        attempt was rejected. The gates themselves are unchanged.
        """
        reasons, groundings = [], []
        for _ in range(gates["max_build_attempts"]):
            description = self._describe(schema, reference_json, llm_provider)
            if not description:
                reasons.append("empty_description")
                continue
            if gates["deterministic_gates"]:
                ratio = _grounding_ratio(reference_json, description)
                groundings.append(round(ratio, 4))
                if ratio < gates["grounding_threshold"]:
                    reasons.append("grounding_failed")
                    continue
            if gates["round_trip"]:
                extraction = self._prefixed(self.task._build_extraction_prompt(schema, description),
                                            self.validator_system_prompt_prefix)
                raw, _ = self._validator.generate(extraction)
                recovered = _loads_lenient(raw or "")
                if recovered is None or \
                        metrics.semantic_fidelity(reference_json, recovered) < gates["round_trip_threshold"]:
                    reasons.append("round_trip_failed")
                    continue
            if diag is not None:
                diag.update(attempts=len(reasons) + 1, reasons=reasons, groundings=groundings)
            return description
        if diag is not None:
            diag.update(attempts=len(reasons), reasons=reasons, groundings=groundings,
                        stage=reasons[-1] if reasons else None)
        return None

    # ----------------------------------------------------------------- modify 2c

    def _make_modification(self, schema, reference_json, llm_provider, rng, gates,
                           mode="standard", diag=None):
        """Returns (modified_json, instruction, change_spec) or None.

        The edit is applied programmatically (correct by construction); the
        instruction is optionally naturalized and round-trip gated so that a fresh
        model can reproduce the modified object from it.

        mode="standard": one edit, chosen with the configured standard strategy.
        mode="complex":  2-3 compatible edits, verified by replay, and a rewritten
        instruction is only kept if it still mentions every edit.
        `diag` receives the failure stage / verification details.
        """
        diag = diag if diag is not None else {}
        complex_mode = mode == "complex"
        if complex_mode:
            mcfg = self.settings.complex["modify"]
            change = apply_changes(reference_json, schema, rng,
                                   n_ops=(mcfg["min_operations"], mcfg["max_operations"]),
                                   attempts=mcfg["attempts"])
            if change is None:
                diag["stage"] = "no_compatible_operations"
                return None
        else:
            change = apply_change(reference_json, schema, rng, strategy=self.settings.modify_strategy)
            if change is None:
                diag["stage"] = "no_valid_edit"
                return None
        modified, spec = change

        diag["verification"] = verify_modification(reference_json, modified, spec, schema)
        if complex_mode and not diag["verification"]["ok"]:
            diag["stage"] = "verification_failed"
            diag["problems"] = diag["verification"]["problems"]
            return None

        instructions = []
        if gates["naturalize_instruction"]:
            for _ in range(gates["naturalize_attempts"]):
                nat = self._naturalize(spec["precise_instruction"], llm_provider, multi=complex_mode)
                if not nat:
                    continue
                if complex_mode:
                    covered, missing = instruction_covers_operations(nat, spec["operations"])
                    if not covered:
                        diag.setdefault("rejected_rewrites", []).append(missing)
                        continue
                instructions.append((nat, "naturalized"))
        instructions.append((spec["precise_instruction"], "precise_template"))  # fallback

        for instruction, source in instructions:
            if not gates["round_trip"] or \
                    self._modify_round_trip_ok(schema, reference_json, instruction, modified,
                                               gates["round_trip_threshold"]):
                diag["instruction_source"] = source
                diag["round_trip_checked"] = gates["round_trip"]
                return modified, instruction, spec
        diag["stage"] = "round_trip_failed"
        return None

    def _naturalize(self, precise_instruction, llm_provider, multi=False):
        many = ("It lists several changes: keep EVERY change, and do not add, drop, merge or "
                "alter any of them. Preserve the exact field names, values, numbers and list "
                "positions. " if multi else "")
        prompt = [{
            "role": "user",
            "content": (
                "Rewrite the following data-edit instruction as a natural, conversational "
                "request a user might type. " + many + "Keep it unambiguous and preserve the "
                "exact field names, values, and numbers. Output only the rewritten instruction."
                f"\n\nInstruction: {precise_instruction}"
            ),
        }]
        text, _ = llm_provider.generate(self._prefixed(prompt))
        return (text or "").strip()

    def _modify_round_trip_ok(self, schema, original, instruction, modified, threshold):
        prompt = self._prefixed(_modify_prompt(schema, original, instruction),
                                self.validator_system_prompt_prefix)
        raw, _ = self._validator.generate(prompt)
        recovered = _loads_lenient(raw or "")
        if recovered is None:
            return False
        return metrics.semantic_fidelity(modified, recovered) >= threshold

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _safe_id(source, subset, origin):
        raw = "__".join(p for p in (source, subset, origin) if p)
        return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)[:150]

    @staticmethod
    def _write(path, record):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def _loads_lenient(text):
    """Parses JSON, tolerating prose/thinking traces around it.

    The round-trip validator may be a reasoning model that wraps its output. We
    only need to recover its intended object to score fidelity, so fall back to
    decoding the first balanced JSON value found in the text. (The evaluations
    themselves stay strict — this leniency applies only to gate verification.)
    """
    obj, _ = metrics.parse_json(text)
    if obj is not None:
        return obj
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return None


def _modify_prompt(schema, original, instruction):
    """Round-trip prompt; mirrors ModifyJson.format_for_llm for consistency."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that modifies JSON objects based on given "
                "instructions. Output only the modified JSON with no other text or explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Given the following JSON:\n```json\n{json.dumps(original, indent=2)}\n```\n"
                f"Modify it as instructed: {instruction}\n\n"
                f"The result must still conform to this schema:\n{json.dumps(schema, indent=2)}"
            ),
        },
        {"role": "assistant", "content": ""},
    ]


def _count_nontrivial(obj):
    """Counts leaf values that are not empty/default (stub-detection)."""
    count = 0
    for _, value in metrics._iter_leaves(obj):
        if isinstance(value, bool):
            count += 1
        elif value is None:
            continue
        elif isinstance(value, str) and value.strip() == "":
            continue
        elif isinstance(value, (list, dict)) and len(value) == 0:
            continue
        else:
            count += 1
    return count


def _grounding_ratio(reference_json, description):
    """Fraction of distinctive string values that appear in the description.

    Only checks alphabetic strings of reasonable length (names, ids, emails, enum
    labels) — values that natural prose rarely rephrases. Numbers, dates, booleans
    and nulls are skipped, since descriptions legitimately reword them ("January
    15" for "2024-01-15", "active" for true). Returns 1.0 when nothing is checkable.
    """
    desc = description.lower()
    checked = grounded = 0
    for _, value in metrics._iter_leaves(reference_json):
        if not isinstance(value, str):
            continue
        token = value.strip()
        if len(token) < 4 or not any(c.isalpha() for c in token):
            continue
        checked += 1
        if token.lower() in desc:
            grounded += 1
    return 1.0 if checked == 0 else grounded / checked
