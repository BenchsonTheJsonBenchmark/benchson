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
"""

import concurrent.futures
import datetime
import json
import os
import random
import re
import threading

from data_generation.tasks.base_task import BaseDataTask
from data_generation.tasks.error_correction import _inject_error
from data_generation.modifications import apply_change
from evaluations import metrics

_JSB_ATTRIBUTION = (
    "Schemas with source 'jsonschemabench' are derived from "
    "epfl-dlab/JSONSchemaBench (MIT, arXiv:2501.10868)."
)


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

    def _load_schema_records(self):
        """Returns a shuffled list of (schema_dict, source, subset, origin_name)."""
        loader = BaseDataTask()
        records = []
        for src in self.sources:
            name = src["name"]
            paths = loader.load_instances(
                source=name,
                filter_kw=src.get("filter"),
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

        pools = {s: [] for s in self.splits}
        rng = random.Random(self.seed)
        for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
            grp = sorted(groups[key], key=lambda r: self._safe_id(r[1], r[2], r[3]))
            rng.shuffle(grp)
            n_test = round(len(grp) * self.test_ratio)
            pools["test"].extend(grp[:n_test])
            if "train" in pools:
                pools["train"].extend(grp[n_test:])
        return pools

    # ------------------------------------------------------------------ build

    def build(self, llm_provider, validator_provider=None):
        # The validator runs the round-trip gate; falls back to the builder model.
        self._validator = validator_provider or llm_provider
        for base in self.base.values():
            os.makedirs(os.path.join(base, "test"), exist_ok=True)
            os.makedirs(os.path.join(base, "train"), exist_ok=True)

        pools = self._partition(self._load_schema_records())
        self._partition_sizes = {s: len(pools[s]) for s in self.splits}

        for split in self.splits:
            self._build_split(split, pools[split], llm_provider)

        for k in self.base:
            self._write_manifest(self.base[k], self._TASK_LABELS[k], llm_provider)

    def _build_split(self, split, pool, llm_provider):
        want = {k: self.counts_by_split[split].get(k, 0) for k in self.base}
        gates = self.gates_by_split[split]
        split_dir = {k: os.path.join(self.base[k], split) for k in self.base}
        for d in split_dir.values():
            os.makedirs(d, exist_ok=True)

        have = {k: len([f for f in os.listdir(split_dir[k]) if f.endswith(".json")]) for k in self.base}
        for k in self.base:
            if have[k]:
                print(f"[{split}/{k}] resuming from {have[k]}/{want[k]}")

        lock = threading.Lock()
        records = iter(pool)
        submitted = 0

        def needed_for(instance_id):
            return {k for k in self.base
                    if have[k] < want[k]
                    and not os.path.exists(os.path.join(split_dir[k], instance_id + ".json"))}

        def targets_met():
            return all(have[k] >= want[k] for k in self.base)

        def submit_next(executor):
            nonlocal submitted
            for schema, source, subset, origin in records:
                instance_id = self._safe_id(source, subset, origin)
                with lock:
                    needed = needed_for(instance_id)
                if not needed:
                    continue
                rng = random.Random(f"{self.seed}-{split}-{submitted}")
                submitted += 1
                return executor.submit(self._process_schema, schema, source, subset,
                                       origin, needed, llm_provider, rng, gates)
            return None

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
                    try:
                        result = fut.result()
                    except Exception as e:  # one bad schema must not kill the run
                        print(f"[{split}] worker error (skipped): {e}")
                        result = None
                    if result is not None:
                        instance_id, produced = result
                        with lock:
                            for task, record in produced.items():
                                path = os.path.join(split_dir[task], instance_id + ".json")
                                if have[task] < want[task] and not os.path.exists(path):
                                    self._write(path, record)
                                    have[task] += 1
                                    print(f"[{split}/{task}] {have[task]}/{want[task]} ({instance_id})")
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

    def _process_schema(self, schema, source, subset, origin, needed, llm_provider, rng, gates):
        """Builds the needed task records for one schema. Returns (instance_id, {task: record})."""
        reference_json = self._mint_reference(schema, llm_provider, gates)
        if reference_json is None:
            return None

        instance_id = self._safe_id(source, subset, origin)
        tags = {"name": origin, "source": source, "subset": subset, "schema": schema}
        produced = {}

        description = None  # gated description, reused by fix when available
        if "create" in needed:
            description = self._make_description(schema, reference_json, llm_provider, gates)
            if description:
                produced["create"] = {**tags, "description": description,
                                      "reference_json": reference_json}

        if "fix" in needed:
            erroneous = _inject_error(reference_json, schema)
            # Keep only mutations that genuinely break schema compliance.
            if erroneous is not None and metrics.schema_compliance(erroneous, schema)[0] == 0:
                fix_desc = description or self._describe(schema, reference_json, llm_provider)
                record = {**tags, "erroneous_json": erroneous, "valid_json": reference_json}
                if fix_desc:
                    record["description"] = fix_desc
                produced["fix"] = record

        if "modify" in needed:
            mod = self._make_modification(schema, reference_json, llm_provider, rng, gates)
            if mod is not None:
                modified, instruction, spec = mod
                produced["modify"] = {**tags, "data": reference_json, "instructions": instruction,
                                      "ground_truth": modified, "modification": spec}

        return instance_id, produced

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

    def _mint_reference(self, schema, llm_provider, gates):
        """Generates a schema-valid, non-trivial reference object, or None.

        Shared starting point for all three tasks. Retries up to
        `max_build_attempts` times to clear the non-triviality gate.
        """
        for _ in range(gates["max_build_attempts"]):
            initial = self.task._build_schema_creation_prompt(schema)
            result = self.task._run_fix_loop(schema, initial, llm_provider, self.max_retries)
            if result is None:
                continue
            try:
                reference_json = json.loads(result[0])
            except json.JSONDecodeError:
                continue
            if not gates["deterministic_gates"] or \
                    _count_nontrivial(reference_json) >= gates["min_populated_fields"]:
                return reference_json
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

    def _make_description(self, schema, reference_json, llm_provider, gates):
        """A description that clears the grounding + round-trip gates, or None."""
        for _ in range(gates["max_build_attempts"]):
            description = self._describe(schema, reference_json, llm_provider)
            if not description:
                continue
            if gates["deterministic_gates"] and \
                    _grounding_ratio(reference_json, description) < gates["grounding_threshold"]:
                continue
            if gates["round_trip"]:
                extraction = self._prefixed(self.task._build_extraction_prompt(schema, description),
                                            self.validator_system_prompt_prefix)
                raw, _ = self._validator.generate(extraction)
                recovered = _loads_lenient(raw or "")
                if recovered is None or \
                        metrics.semantic_fidelity(reference_json, recovered) < gates["round_trip_threshold"]:
                    continue
            return description
        return None

    # ----------------------------------------------------------------- modify 2c

    def _make_modification(self, schema, reference_json, llm_provider, rng, gates):
        """Returns (modified_json, instruction, change_spec) or None.

        The edit is applied programmatically (correct by construction); the
        instruction is optionally naturalized and round-trip gated so that a fresh
        model can reproduce the modified object from it.
        """
        change = apply_change(reference_json, schema, rng)
        if change is None:
            return None
        modified, spec = change

        instructions = []
        if gates["naturalize_instruction"]:
            for _ in range(gates["naturalize_attempts"]):
                nat = self._naturalize(spec["precise_instruction"], llm_provider)
                if nat:
                    instructions.append(nat)
        instructions.append(spec["precise_instruction"])  # precise template fallback

        for instruction in instructions:
            if not gates["round_trip"]:
                return modified, instruction, spec
            if self._modify_round_trip_ok(schema, reference_json, instruction, modified,
                                          gates["round_trip_threshold"]):
                return modified, instruction, spec
        return None

    def _naturalize(self, precise_instruction, llm_provider):
        prompt = [{
            "role": "user",
            "content": (
                "Rewrite the following data-edit instruction as a natural, conversational "
                "request a user might type. Keep it unambiguous and preserve the exact field "
                "names, values, and numbers. Output only the rewritten instruction.\n\n"
                f"Instruction: {precise_instruction}"
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
