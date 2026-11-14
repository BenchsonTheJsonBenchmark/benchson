"""Read-only quality audit of the Scale Only dataset: original 3,000 vs newly generated 4,000 vs merged 7,000 (+ test integrity).

    python src/audit_scale_dataset.py --baseline data_published/7962b0947f7d --new data_scale_only/merged_4k_20261114 \
        --merged data_scale_7k_20261114 --jsonl hf_export_7k_20261114 --out-dir outputs/quality_audit/scale_only_7k_20261114

Nothing in the inputs is written. The output directory must not exist; it is built in a private staging directory and renamed
into place only when complete. No model is called and nothing is uploaded. Remote `$ref` documents are fetched read-only by the
existing bounded retriever (src/evaluations/schema_validation.py); use --offline to forbid network access (every example that
needs a remote document is then TECHNICAL: reported as unresolved, never as valid or invalid, and training readiness is then NOT READY).

Three separate verdicts: data integrity (are the files what they claim to be), training readiness (any blocking finding, or open
defects in the original dataset: genuine validation failures of the original examples are reported, never hidden, and nothing is
modified or excluded), scientific comparability (are original vs new comparable). Exit codes: 0 READY, 3 REVIEW REQUIRED, 2 NOT READY.
A failed provenance or chain check never aborts the audit: the remaining diagnostics are still produced and the dependent checks are
reported NOT_VERIFIED.

Every check says whether the audit verified it itself ("independent") or only compared with what the merge/chain manifests
claim ("inherited"). Validation is three-way (VALID / INVALID / TECHNICAL): TECHNICAL is never counted as invalid and never as valid.

Metric definitions are in the report (section "Metric definitions") and in METRIC_DEFINITIONS below.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_scale_dataset as msd
from data_generation import complexity
from data_generation.published import (REQUIRED_FIELDS, TASKS, encode_record, overlap_report, reference_hash, schema_hash,
                                       sha256_file, task_keys)
from evaluations import metrics
from evaluations import schema_validation as sv

FORMAT = "benchson-quality-audit/1"
PRIMARY = {"create": "reference_json", "fix": "valid_json", "modify": "data"}
TEXT_FIELD = {"create": "description", "modify": "instructions"}
SUBSCHEMA_MAPS = ("properties", "patternProperties", "definitions", "$defs", "dependentSchemas")
SUBSCHEMA_ONE = ("items", "additionalProperties", "additionalItems", "contains", "not", "if", "then", "else", "propertyNames",
                 "unevaluatedProperties", "unevaluatedItems")
SUBSCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")
ANNOTATION_KEYS = {"$ref", "definitions", "$defs", "$schema", "$id", "id", "title", "description", "default", "examples", "$comment",
                   "readOnly", "writeOnly", "deprecated"}
CONSTRAINT_KEYS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
                   "format", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties")

METRIC_DEFINITIONS = {
    "schema_bytes": "UTF-8 bytes of the compact JSON of the schema (separators (',',':'), keys in file order). Tokens are NOT computed (they need the serving tokenizer).",
    "schema_properties": "Number of keys inside every `properties` object, at any depth (definitions included, $ref not followed).",
    "schema_required": "Number of names listed in every `required` array of the schema.",
    "schema_depth": "Static nesting depth of schema nodes (root = 1) through properties/patternProperties/items/additionalProperties/combinators/definitions; $ref is NOT followed, so recursive schemas are not unrolled.",
    "schema_arrays": "Schema nodes with type 'array' or an `items`/`prefixItems` keyword.",
    "schema_objects": "Schema nodes with type 'object' or a `properties` keyword.",
    "schema_constraints": "Occurrences of minimum, maximum, exclusiveMinimum/Maximum, multipleOf, minLength, maxLength, pattern, format, minItems, maxItems, uniqueItems, minProperties, maxProperties on schema nodes.",
    "schema_enums": "Schema nodes with `enum` or `const`.",
    "schema_patterns": "`pattern` keywords plus patternProperties keys.",
    "schema_combinators": "Occurrences of allOf, anyOf, oneOf, not.",
    "schema_refs": "Occurrences of `$ref`.",
    "schema_defs": "Entries of `definitions` and `$defs`.",
    "schema_external_refs": "`$ref` values that do not start with '#'.",
    "schema_ref_siblings": "Schema nodes that have a `$ref` AND other keywords besides annotations/definitions (e.g. `required`). Drafts up to 7 ignore those siblings, 2019-09/2020-12 enforce them, so the verdict of such a schema depends on the jsonschema version (3.2.0 vs the pinned 4.26.0). A failing result on such a schema is reported as UNCONFIRMED, never as a genuine defect.",
    "json_leaves": "Leaf values of the primary instance (create: reference_json, fix: valid_json, modify: data); empty containers count as leaves (metrics._iter_leaves).",
    "json_depth": "Longest leaf path of the primary instance (a top-level key = 1; complexity.path_depth).",
    "json_fields": "Number of object keys at every depth of the primary instance.",
    "json_nested_objects": "Objects inside the primary instance, excluding the root.",
    "json_max_array_len": "Longest array in the primary instance (0 when there is none).",
    "json_max_object_size": "Largest number of keys of any object in the primary instance.",
    "json_chars": "Characters of the compact JSON of the primary instance.",
    "json_distinct_leaf_types": "Number of distinct leaf kinds (boolean, integer, number, string, null, empty container).",
    "grounding": "Create only: project metric benchmark_builder._grounding_ratio = share of distinctive strings (>= 4 characters, with a letter) of reference_json that appear in the description (case-insensitive substring). Lexical, not a semantic check.",
    "description_chars": "Create: characters of the description. `_words`: whitespace-separated words.",
    "instruction_chars": "Modify: characters of the instruction. `_words`: whitespace-separated words.",
    "changed_paths": "Fix: leaf paths that differ between valid_json and erroneous_json. Modify: between data and ground_truth (metrics.changed_paths: added, removed or changed leaf). Used as a proxy for 'number of change operations'; one operation may touch several leaves.",
    "changed_max_depth": "Largest path length among the changed leaf paths.",
    "changed_mean_depth": "Mean path length of the changed leaf paths.",
    "unrelated_changes": "Fix: changed leaf paths NOT covered by any schema violation of erroneous_json (a violation path that is a prefix of the changed path, or that the changed path is a prefix of). Structural and approximate: a violation the validator reports at a parent counts as covering its children; combinators can hide coverage.",
    "instruction_path_mentions": "Modify: share of changed leaf paths whose last key name occurs (case-insensitive) in the instruction. Lexical and approximate; it does NOT show that the instruction was carried out correctly.",
}


# ------------------------------------------------------------------ pure helpers

def _stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}

    def pct(p):
        k = (len(v) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "min": v[0], "max": v[-1],
            "p25": pct(.25), "p75": pct(.75), "p90": pct(.90), "p95": pct(.95)}


def _cluster_bootstrap(a, b, reps, seed):
    """95% percentile interval of mean(b)-mean(a) when whole schemas (clusters) are resampled, not single examples.
    a, b: {cluster: [values]}. Examples that share a schema are not independent, so the interval is wider than a plain one."""
    if not a or not b or reps <= 0:
        return None
    rng = random.Random(seed)

    def pack(groups):
        return [(sum(v), len(v)) for v in groups.values()]
    pa, pb = pack(a), pack(b)

    def mean_of(packed):
        picks = rng.choices(packed, k=len(packed))
        n = sum(c for _, c in picks)
        return sum(s for s, _ in picks) / n if n else None
    diffs = sorted(d for d in (mean_of(pb) - mean_of(pa) for _ in range(reps)) if d is not None)
    if not diffs:
        return None
    return [diffs[int(0.025 * (len(diffs) - 1))], diffs[int(0.975 * (len(diffs) - 1))]]


def _words(text):
    return len(text.split()) if isinstance(text, str) else None


def _compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def schema_metrics(schema):
    """Static, $ref-free structural counts of a schema (see METRIC_DEFINITIONS). Never raises."""
    c = Counter()
    external, max_depth, nodes = set(), 0, [0]

    def node(s, depth):
        nonlocal max_depth
        if not isinstance(s, dict) or depth > 200 or nodes[0] > 200000:
            return
        nodes[0] += 1
        max_depth = max(max_depth, depth)
        t = s.get("type")
        if t == "object" or isinstance(s.get("properties"), dict):
            c["schema_objects"] += 1
        if t == "array" or "items" in s or "prefixItems" in s:
            c["schema_arrays"] += 1
        if isinstance(s.get("properties"), dict):
            c["schema_properties"] += len(s["properties"])
        if isinstance(s.get("required"), list):
            c["schema_required"] += len(s["required"])
        for k in CONSTRAINT_KEYS:
            if k in s:
                c["schema_constraints"] += 1
        if "enum" in s or "const" in s:
            c["schema_enums"] += 1
        if "pattern" in s:
            c["schema_patterns"] += 1
        if isinstance(s.get("patternProperties"), dict):
            c["schema_patterns"] += len(s["patternProperties"])
        for k in ("allOf", "anyOf", "oneOf", "not"):
            if k in s:
                c["schema_combinators"] += 1
        for k in ("definitions", "$defs"):
            if isinstance(s.get(k), dict):
                c["schema_defs"] += len(s[k])
        ref = s.get("$ref")
        if isinstance(ref, str):
            c["schema_refs"] += 1
            if set(s) - ANNOTATION_KEYS:
                c["schema_ref_siblings"] += 1
            if not ref.startswith("#"):
                c["schema_external_refs"] += 1
                external.add(ref.split("#", 1)[0])
        for k in SUBSCHEMA_MAPS:
            if isinstance(s.get(k), dict):
                for sub in s[k].values():
                    node(sub, depth + 1)
        for k in SUBSCHEMA_ONE:
            if isinstance(s.get(k), dict):
                node(s[k], depth + 1)
        for k in SUBSCHEMA_LISTS:
            if isinstance(s.get(k), list):
                for sub in s[k]:
                    node(sub, depth + 1)
        if isinstance(s.get("items"), list):
            for sub in s["items"]:
                node(sub, depth + 1)

    node(schema, 1)
    out = {k: c.get(k, 0) for k in ("schema_properties", "schema_required", "schema_arrays", "schema_objects", "schema_constraints",
                                    "schema_enums", "schema_patterns", "schema_combinators", "schema_refs", "schema_defs",
                                    "schema_external_refs", "schema_ref_siblings")}
    out["schema_depth"] = max_depth
    out["schema_bytes"] = len(_compact(schema).encode("utf-8"))
    return out, sorted(external)


def instance_metrics(value):
    ref = complexity.reference_metrics(value)          # leaf_count, depth, distinct_leaf_types, ...
    objs = arrays = fields = 0
    max_arr = max_obj = 0
    stack = [(value, True)]
    while stack:
        v, root = stack.pop()
        if isinstance(v, dict):
            objs += 0 if root else 1
            fields += len(v)
            max_obj = max(max_obj, len(v))
            stack.extend((x, False) for x in v.values())
        elif isinstance(v, list):
            arrays += 1
            max_arr = max(max_arr, len(v))
            stack.extend((x, False) for x in v)
    return {"json_leaves": ref["leaf_count"], "json_depth": ref["depth"], "json_fields": fields, "json_nested_objects": objs,
            "json_max_array_len": max_arr, "json_max_object_size": max_obj, "json_chars": len(_compact(value)),
            "json_distinct_leaf_types": ref["distinct_leaf_types"]}


def _change_kind(before, after, path):
    b_found, b = metrics._navigate(before, path)
    a_found, a = metrics._navigate(after, path)
    if b_found and not a_found:
        return "removed"
    if a_found and not b_found:
        return "added"
    return "type_change" if _kind(b) != _kind(a) else "value_change"


def _kind(v):
    return "boolean" if isinstance(v, bool) else "null" if v is None else "number" if isinstance(v, (int, float)) else \
        "string" if isinstance(v, str) else "array" if isinstance(v, list) else "object"


def change_summary(before, after):
    paths = sorted(metrics.changed_paths(before, after), key=lambda p: [str(s) for s in p])
    kinds = Counter(_change_kind(before, after, p) for p in paths)
    depths = [complexity.path_depth(p) for p in paths]
    return paths, kinds, depths


def _covered(path, violation_paths):
    for v in violation_paths:
        n = min(len(v), len(path))
        if tuple(v[:n]) == tuple(path[:n]):
            return True
    return False


def violation_info(schema, instance):
    """(keywords, paths) of the schema violations of `instance`; ([], []) if this cannot be computed."""
    try:
        errors = list(sv._validator(schema).iter_errors(instance))
        return sorted({e.validator for e in errors}), [tuple(e.absolute_path) for e in errors]
    except Exception:
        return [], []


def classify_technical(detail):
    if detail.startswith("malformed schema"):
        return "malformed_schema"
    if detail.startswith("external reference"):
        return "unresolvable_external_ref"
    return "validator_error"


def _remote_url(detail):
    import re
    m = re.search(r"https?://[^\s'\"]+", detail)
    return sv._display(m.group(0).rstrip(".,)")) if m else ""


# ------------------------------------------------------------------ loading

def read_split(root, task, split):
    """{name: (record|None, sha256, error|None)}. Every directory entry is listed, so stray files cannot hide."""
    out = {}
    d = os.path.join(root, f"benchmark_{task}", split)
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        path = os.path.join(d, name)
        if not os.path.isfile(path):
            out[name] = (None, "", "not a regular file")
            continue
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8"))
            out[name] = (data, digest, None) if isinstance(data, dict) else (None, digest, "valid JSON but not an object")
        except (ValueError, UnicodeDecodeError) as e:
            out[name] = (None, digest, f"unparseable: {str(e)[:100]}")
    return out


def dup_summary(group_sizes):
    """One definition of 'duplicate' for every table. A group = records with the same key (size k >= 2).
    duplicate_groups: number of groups; records_in_duplicate_groups: all records that belong to a group (sum k);
    redundant_records: records beyond the first of each group (sum k-1 = what removing duplicates would delete);
    duplicate_pairs: unordered pairs inside groups (sum k(k-1)/2). Counting never depends on origin or task."""
    g = [k for k in group_sizes if k > 1]
    return {"duplicate_groups": len(g), "records_in_duplicate_groups": sum(g), "redundant_records": sum(k - 1 for k in g),
            "duplicate_pairs": sum(k * (k - 1) // 2 for k in g)}


def schema_id(name):
    stem = name[:-5] if name.endswith(".json") else name
    sid, _, att = stem.rpartition("__a")
    return (sid, att) if sid and att.isdigit() else (stem, "")


# ------------------------------------------------------------------ the audit

class Audit:
    def __init__(self, args):
        self.missing_inputs = []
        self.args = args
        self.checks = []          # {id, title, status, basis, critical, detail}
        self.rows = {"complexity": [], "failures": [], "dups": []}
        self.summary = {}
        self.prov = {}
        # safe defaults, so a failure in one phase never prevents the later phases and the report
        self.baseline_ok = self.new_ok = False
        self.base_records = {t: {"train": {}, "test": {}} for t in TASKS}
        self.base_sha = {t: {"train": {}, "test": {}} for t in TASKS}
        self.new_sha = {t: {} for t in TASKS}
        self.merged = {t: {"train": {}, "test": {}} for t in TASKS}
        self.origin = {t: {} for t in TASKS}
        self.counts = {t: {"train": 0, "test": 0} for t in TASKS}
        self.new_manifest = {}
        self.test_hashes = set()
        self.reuse = {"references_per_schema": Counter(), "per_task_per_schema": Counter()}
        self.val = {(t, o): Counter() for t in TASKS for o in ("original", "new", "unknown")}
        self.m = {(t, o): defaultdict(list) for t in TASKS for o in ("original", "new", "unknown")}
        self.cluster = {(t, o): defaultdict(lambda: defaultdict(list)) for t in TASKS for o in ("original", "new", "unknown")}
        self.kinds = {(t, o, k): Counter() for t in TASKS for o in ("original", "new", "unknown") for k in ("change_kind", "violation_keyword")}
        self.texts = {(t, o): [] for t in TASKS for o in ("original", "new", "unknown")}
        self.records_seen = {(t, o): [] for t in TASKS for o in ("original", "new", "unknown")}
        self.external_refs = {"distinct_external_documents": 0, "documents": {}, "documents_fetched_by_this_process": {}}
        self.stat_rows = []
        self.dup_groups = {}

    # checks that compare the merged dataset with the baseline / the new run: they mean nothing unless both were verified first
    DEPENDENT = ("B2", "B3", "B6", "C1", "C2", "C2b", "C4")

    def check(self, cid, title, status, basis, detail="", critical=True):
        if status == "PASS" and cid in self.DEPENDENT and not (self.baseline_ok and self.new_ok):
            status, detail = "NOT_VERIFIED", "compared with a baseline / new dataset that is NOT verified (see A1, A2); " + detail
        self.checks.append({"id": cid, "title": title, "status": status, "basis": basis, "critical": critical, "detail": detail})

    def guard(self, cid, title, fn):
        """Runs one audit section. An unexpected error becomes a FAIL finding; the audit goes on and still writes its report."""
        try:
            fn()
        except Exception as e:
            self.check(cid, f"{title} could not be completed", "FAIL", "independent", f"{type(e).__name__}: {str(e)[:300]}")

    def run(self):
        a = self.args
        saved = (sv._opener, sv.REMOTE_ATTEMPTS)          # module settings are restored afterwards (the audit may run inside a larger process)
        if a.offline:
            def refuse(request, timeout):
                raise OSError("offline mode: no network access during the audit")
            sv._opener, sv.REMOTE_ATTEMPTS = refuse, 1
        sv.reset_caches()
        if self.missing_inputs:
            self.check("I0", "All inputs exist", "FAIL", "independent", "missing: " + ", ".join(self.missing_inputs))
        try:
            self.guard("P1", "Integrity and provenance checks", self.integrity)
            self.guard("P2", "Validation and measurements", self.validate_and_measure)
            self.guard("P3", "Diversity and duplicate analysis", self.diversity)
            self.guard("P4", "Statistics", self.statistics)
            self.finish()
        finally:
            sv._opener, sv.REMOTE_ATTEMPTS = saved

    # ---------------- A. integrity and provenance
    def integrity(self):
        a = self.args
        expected_base = {"train": a.expected_baseline_train, "test": a.expected_baseline_test}
        self.base_records = {t: {"train": {}, "test": {}} for t in TASKS}
        try:
            prov, self.base_records = msd.verify_baseline(a.baseline, a.expected_revision, expected_base)
            self.prov["baseline"] = prov
            self.check("A1", "Baseline equals the pinned published dataset (revision, counts, JSONL SHA256, git blob ids, every file = its published row)",
                       "PASS", "independent", f"revision {prov['revision']}")
            self.baseline_ok = True
        except Exception as e:            # MergeAbort (a finding) or anything unexpected: both are reported, neither is "verified"
            self.check("A1", "Baseline equals the pinned published dataset", "FAIL", "independent", str(e)[:1500])
            self.base_records = {t: {s: {n: r for n, (r, _, _) in read_split(a.baseline, t, s).items() if r is not None}
                                     for s in ("train", "test")} for t in TASKS}
        self.base_sha = {t: {s: {n: v[1] for n, v in read_split(a.baseline, t, s).items()} for s in ("train", "test")} for t in TASKS}

        errors = []
        try:
            prov, self.new_manifest, new_names = msd.verify_new(a.new, a.expected_new, a.expected_chunks, errors)
            self.prov["new"] = prov
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            self.new_manifest, new_names = {}, {t: [] for t in TASKS}
        if a.chunks_root:
            if not (a.anchor_file and a.reserved_file):
                errors.append("--chunks-root needs --anchor-file and --reserved-file")
            else:
                try:
                    chain = msd.verify_chain_dirs(self.new_manifest, a.chunks_root, a.anchor_file, a.reserved_file, errors)
                except Exception as e:
                    chain = None
                    errors.append(f"chain re-validation crashed: {e!r}")
                self.prov.setdefault("new", {})["chain_verification"] = chain or "FAILED"
        self.check("A2", "New dataset matches merged_chunks_manifest.json (files, SHA256, quotas, no test split, chain shape)",
                   "FAIL" if errors else "PASS", "independent (files) / inherited (chain shape)", "; ".join(errors[:8]))
        self.new_ok = not errors
        chain_basis = "independent" if a.chunks_root and not errors else "NOT_VERIFIED"
        self.check("A3", "Chunk chain (state files, seals) re-validated against the chunk directories",
                   "PASS" if chain_basis == "independent" else "NOT_VERIFIED", chain_basis,
                   "" if a.chunks_root else "run with --chunks-root/--anchor-file/--reserved-file to verify; otherwise provenance of the chain is only what the merged manifest claims",
                   critical=False)
        self.new_sha = {t: {} for t in TASKS}
        for t in TASKS:
            for n, v in read_split(a.new, t, "train").items():
                self.new_sha[t][n] = v[1]

        self.guard("P1b", "Merged dataset checks", self._merged_checks)
        self.guard("P1c", "JSONL export check", self.jsonl_check)
        self.guard("P1d", "Leakage and duplicate checks", self.leakage_and_duplicates)

    def _merged_checks(self):
        a = self.args
        # merged dataset
        self.merged = {t: {s: read_split(a.merged, t, s) for s in ("train", "test")} for t in TASKS}
        counts, problems = {}, []
        want_train = {t: a.expected_baseline_train + a.expected_new[t] for t in TASKS}
        for t in TASKS:
            counts[t] = {s: len(self.merged[t][s]) for s in ("train", "test")}
            if counts[t]["train"] != want_train[t]:
                problems.append(f"{t}: {counts[t]['train']} train files, expected {want_train[t]}")
            if counts[t]["test"] != a.expected_baseline_test:
                problems.append(f"{t}: {counts[t]['test']} test files, expected {a.expected_baseline_test}")
        self.check("B1", "Merged dataset has exactly the expected counts per task and split", "FAIL" if problems else "PASS", "independent",
                   "; ".join(problems) or json.dumps(counts))
        self.counts = counts
        extra_top = sorted(set(os.listdir(a.merged)) - {f"benchmark_{t}" for t in TASKS} - {"merge_manifest.json"}) if os.path.isdir(a.merged) else ["(merged directory missing)"]

        self.origin = {t: {} for t in TASKS}
        miss, extra, changed = [], [], []
        for t in TASKS:
            for n, (_, sha, _) in self.merged[t]["train"].items():
                in_base, in_new = n in self.base_sha[t]["train"], n in self.new_sha[t]
                if in_base and self.base_sha[t]["train"][n] == sha:
                    self.origin[t][n] = "original"
                elif in_new and self.new_sha[t][n] == sha:
                    self.origin[t][n] = "new"
                elif in_base != in_new:           # the name identifies the source, but the bytes differ: still measured, and B2 fails
                    self.origin[t][n] = "original" if in_base else "new"
                    changed.append(f"{t}/{n}")
                else:
                    self.origin[t][n] = "unknown"
                    (changed if in_base else extra).append(f"{t}/{n}")
            for n in set(self.base_sha[t]["train"]) | set(self.new_sha[t]):
                if n not in self.merged[t]["train"]:
                    miss.append(f"{t}/{n}")
            both = set(self.base_sha[t]["train"]) & set(self.new_sha[t])
            if both:
                changed.append(f"{t}: {len(both)} names exist in both the baseline and the new run")
            tmiss = set(self.base_sha[t]["test"]) - set(self.merged[t]["test"])
            textra = set(self.merged[t]["test"]) - set(self.base_sha[t]["test"])
            tdiff = [n for n, (_, s, _) in self.merged[t]["test"].items() if n in self.base_sha[t]["test"] and s != self.base_sha[t]["test"][n]]
            if tmiss or textra or tdiff:
                changed.append(f"{t} test: missing {len(tmiss)}, extra {len(textra)}, changed {len(tdiff)}")
        self.check("B2", "Every merged train file is byte-identical (SHA256) to exactly one baseline or new source file; none missing, extra or overwritten",
                   "FAIL" if (miss or extra or changed or extra_top) else "PASS", "independent",
                   f"missing {len(miss)}, extra {len(extra)}, changed/overwritten {len(changed)}, unexpected top-level {extra_top}; "
                   + "; ".join((miss + extra + changed)[:6]))
        tbad = [f"{t}" for t in TASKS if {n: s for n, (_, s, _) in self.merged[t]["test"].items()} != self.base_sha[t]["test"]]
        self.check("B3", "All 450 original test files are present and byte-identical to the baseline", "FAIL" if tbad else "PASS", "independent",
                   f"differs in: {tbad}" if tbad else "")
        unparse = [f"{t}/{s}/{n}: {e}" for t in TASKS for s in ("train", "test") for n, (_, _, e) in self.merged[t][s].items() if e]
        self.check("B4", "Every merged example file is parseable JSON", "FAIL" if unparse else "PASS", "independent", "; ".join(unparse[:5]))

        mm_path = os.path.join(a.merged, "merge_manifest.json")
        try:
            with open(mm_path, encoding="utf-8") as f:
                mm = json.load(f)
            issues = []
            if mm.get("baseline_provenance", {}).get("revision") != a.expected_revision:
                issues.append("manifest baseline revision differs")
            nmh = sha256_file(os.path.join(a.new, "merged_chunks_manifest.json")) if os.path.isfile(os.path.join(a.new, "merged_chunks_manifest.json")) else None
            if mm.get("new_provenance", {}).get("merged_chunks_manifest_sha256") != nmh:
                issues.append("manifest records a different merged_chunks_manifest.json hash than the one on disk")
            ofs = mm.get("output_file_sha256", {})
            for t in TASKS:
                for s in ("train", "test"):
                    rec = ofs.get(t, {}).get(s, {})
                    act = {n: v[1] for n, v in self.merged[t][s].items()}
                    if rec != act:
                        issues.append(f"{t}/{s}: recorded output hashes differ from the files on disk")
            self.prov["merge_manifest"] = {"sha256": sha256_file(mm_path), "format": mm.get("format"), "chain": mm.get("new_provenance", {}).get("chain"),
                                           "schema_reuse": mm.get("policy"), "limitations": mm.get("limitations")}
            self.check("B5", "merge_manifest.json is consistent with the files on disk (cross-check of inherited evidence)",
                       "FAIL" if issues else "PASS", "inherited, cross-checked", "; ".join(issues))
        except (OSError, ValueError) as e:
            self.check("B5", "merge_manifest.json is readable and consistent", "FAIL", "inherited", str(e)[:200])


    def jsonl_check(self):
        a = self.args
        if not a.jsonl:
            self.check("B6", "JSONL export consistent with the merged dataset", "NOT_VERIFIED", "-", "no --jsonl given", critical=False)
            return
        issues = []
        for t in TASKS:
            for s in ("train", "test"):
                path = os.path.join(a.jsonl, t, f"{s}.jsonl")
                if not os.path.isfile(path):
                    issues.append(f"{t}/{s}.jsonl missing")
                    continue
                try:
                    with open(path, encoding="utf-8") as f:
                        lines = [l.rstrip("\n") for l in f if l.strip()]
                    got = Counter(_compact_sorted(json.loads(l)) for l in lines)
                except ValueError as e:
                    issues.append(f"{t}/{s}.jsonl: unparseable line: {str(e)[:80]}")
                    continue
                want = Counter(_compact_sorted(encode_record(r)) for r, _, _ in self.merged[t][s].values() if r is not None)
                if len(lines) != len(self.merged[t][s]):
                    issues.append(f"{t}/{s}: {len(lines)} JSONL lines, {len(self.merged[t][s])} files")
                if got != want:
                    issues.append(f"{t}/{s}: JSONL content differs from the merged files")
                orig = os.path.join(a.baseline, "original_files", t, f"{s}.jsonl")
                if os.path.isfile(orig):
                    with open(orig, encoding="utf-8") as f:
                        published = {l.rstrip("\n") for l in f if l.strip()}
                    if not published <= set(lines):
                        issues.append(f"{t}/{s}: published rows are not reproduced byte for byte")
        extra = sorted(set(os.listdir(a.jsonl)) - set(TASKS))
        if extra:
            issues.append(f"unexpected entries: {extra}")
        self.check("B6", "JSONL export has the merged dataset's exact records (and baseline rows byte for byte)", "FAIL" if issues else "PASS",
                   "independent", "; ".join(issues[:6]))

    def leakage_and_duplicates(self):
        a = self.args
        test_hashes, test_exact, test_refs = set(), {t: set() for t in TASKS}, set()
        leak = []
        for t in TASKS:
            for r, _, _ in self.merged[t]["test"].values():
                if r is None:
                    continue
                try:
                    h = schema_hash(r["schema"])
                    test_hashes.add(h)
                    test_exact[t].add(task_keys(t, r)[0])
                    test_refs.add((h, reference_hash(t, r)))
                except Exception as err:
                    leak.append(f"{t} test file cannot be fingerprinted: {err!r}")
        self.test_hashes = test_hashes
        dup_test, groups = [], {t: defaultdict(list) for t in TASKS}      # groups: exact key -> [(name, origin)]
        base_dup_test = []          # original examples identical to a test example: baseline property, reported, never blocking
        ref_owner = {}
        for t in TASKS:
            for n, (r, _, e) in sorted(self.merged[t]["train"].items()):
                if r is None:
                    continue
                try:
                    h = schema_hash(r["schema"])
                    ex, _c = task_keys(t, r)
                    rh = (h, reference_hash(t, r))
                except Exception as err:
                    leak.append(f"{t}/{n}: cannot fingerprint: {err!r}")
                    continue
                org = self.origin[t].get(n, "unknown")
                if h in test_hashes and org == "new":
                    leak.append(f"{t}/{n}: schema of a NEW example is in the test set")
                if ex in test_exact[t]:
                    (dup_test if org != "original" else base_dup_test).append(f"{t}/{n} [{org}] identical to a test example")
                groups[t][ex].append((n, org))
                if org == "new" and rh in test_refs:
                    leak.append(f"{t}/{n}: same schema and reference as a test example")
        self.check("C1", "No leakage of new examples: schema not in the published test set, no identical or same-reference test example (existing project policy)",
                   "FAIL" if leak else "PASS", "independent", "; ".join(leak[:6]))
        self.dup_groups = groups
        involving_new, original_only = [], []
        for t in TASKS:
            for ex, members in groups[t].items():
                if len(members) > 1:
                    (involving_new if any(o != "original" for _, o in members) else original_only).append((t, members))

        def describe(gs):
            d = dup_summary(len(m) for _, m in gs)
            sample = "; ".join(f"{t}: " + ", ".join(f"{n} [{o}]" for n, o in m[:3]) for t, m in gs[:2])
            return (f"{d['duplicate_groups']} groups, {d['records_in_duplicate_groups']} records in groups, "
                    f"{d['redundant_records']} redundant records, {d['duplicate_pairs']} pairs" + (f" (e.g. {sample})" if gs else ""))
        self.check("C2", "No exact duplicate group involving a NEW example (new vs original, new vs new), and no new example identical to a test example",
                   "FAIL" if (dup_test or involving_new) else "PASS", "independent",
                   "; ".join(dup_test[:4]) + (" | " if dup_test and involving_new else "") + describe(involving_new))
        self.check("C2b", "Exact duplicate groups / test-identical examples that exist ONLY inside the ORIGINAL dataset (baseline property; reported, never altered)",
                   "WARN" if (original_only or base_dup_test) else "PASS", "independent",
                   describe(original_only) + f"; {len(base_dup_test)} original train examples identical to a test example", critical=False)
        # baseline's own train/test overlap (published property, reported not judged)
        orig_dir = os.path.join(a.baseline, "original_files")
        try:
            rep = overlap_report(orig_dir)
            self.prov["baseline_train_test_overlap"] = {"shared_schema_contents": rep["shared_schema_contents"], "per_task": rep["per_task"]}
            self.check("C3", "Train/test overlap inside the ORIGINAL published dataset (existing property of the baseline; reported, not changed)",
                       "INFO", "independent", f"{rep['shared_schema_contents']} schema contents occur in both original splits; new examples never use them (C1)", critical=False)
        except Exception as e:
            self.check("C3", "Train/test overlap inside the original dataset", "NOT_VERIFIED", "-", repr(e)[:150], critical=False)

        # reuse cap per schema id, from file names (same rule as chunked_run.validate_chunk)
        atts, per_task = defaultdict(set), Counter()
        for t in TASKS:
            for n in self.new_sha[t]:
                sid, att = schema_id(n)
                atts[sid].add(att)
                per_task[(t, sid)] += 1
        bad_names = [n for t in TASKS for n in self.new_sha[t] if not schema_id(n)[1]]
        over = [f"{sid}: {len(v)} references" for sid, v in atts.items() if len(v) > a.cap] + \
               [f"{t}/{sid}: {n} examples" for (t, sid), n in per_task.items() if n > a.cap]
        self.reuse = {"references_per_schema": Counter(len(v) for v in atts.values()), "per_task_per_schema": Counter(per_task.values())}
        self.check("C4", f"Controlled schema reuse: at most {a.cap} new references per schema id and per task (counted from file names like the chain validation)",
                   "FAIL" if (over or bad_names) else "PASS", "independent (from files); the ledger itself is inherited",
                   "; ".join((over + [f"unparseable name {n}" for n in bad_names])[:6]))

    # ---------------- B. validation + per-example measurements
    def validate_and_measure(self):
        a = self.args
        external = defaultdict(set)
        for t in TASKS:
            for name, (rec, _sha, err) in sorted(self.merged[t]["train"].items()):
                org = self.origin[t].get(name, "unknown")
                v = self.val[(t, org)]
                v["examples"] += 1
                if rec is None:
                    v["unparseable"] += 1
                    self.fail(t, org, name, "file", "unparseable", err or "")
                    continue
                v["parseable"] += 1
                try:
                    self.evaluate(t, org, name, rec, external)
                except Exception as e:      # one odd record must not stop the audit; it becomes a finding
                    v["audit_errors"] += 1
                    self.fail(t, org, name, "record", "audit_internal_error", f"{type(e).__name__}: {e}", "error")
        urls = sorted(external)
        self.external_refs = {"distinct_external_documents": len(urls), "documents": {u: sv.used_documents().get(u) for u in urls},
                              "documents_fetched_by_this_process": sv.used_documents()}

    def fail(self, task, origin, name, field, category, reason, status="invalid", url=""):
        self.rows["failures"].append({"task": task, "origin": origin, "name": name, "field": field, "status": status,
                                      "category": category, "reason": str(reason)[:300], "remote_url": url})

    def vstatus(self, t, org, name, rec, field, expect, external, blocking=True):
        """Runs the existing three-way validation; returns (state, violation keywords). expect: 'valid' | 'invalid' | None (informational).

        A result that contradicts the expectation is a GENUINE defect only when it does not depend on how the validator reads `$ref`
        siblings (drafts <= 7 ignore them, 2019-09/2020-12 enforce them). On such schemas it is recorded as UNCONFIRMED."""
        v = self.val[(t, org)]
        if not isinstance(rec["schema"], dict):
            v[f"{field}:technical"] += 1
            if blocking:
                self._tech_flag = True
            self.fail(t, org, name, field, "malformed_schema", "schema is not an object", "technical")
            return sv.TECHNICAL, []
        state, detail = sv.validate_instance(rec[field], rec["schema"])
        v[f"{field}:{state}"] += 1
        kinds = []
        ambiguous = self._refsib > 0
        if state == sv.TECHNICAL:
            if blocking:
                self._tech_flag = True
            cat = classify_technical(detail)
            v[f"technical:{cat}"] += 1
            url = _remote_url(detail)
            if url:
                external[url].add(name)
            self.fail(t, org, name, field, cat, detail, "technical", url)
        elif state == sv.INVALID:
            kinds, _paths = violation_info(rec["schema"], rec[field])
            for k in kinds:
                self.kinds[(t, org, "violation_keyword")][f"{field}:{k}"] += 1
            if expect == "valid":
                if ambiguous:
                    v[f"unconfirmed:{field}"] += 1
                    self.fail(t, org, name, field, "unconfirmed_depends_on_ref_sibling_semantics:" + ",".join(kinds), detail, "unresolved")
                else:
                    v[f"genuine:{field}"] += 1
                    for k in kinds or ["unknown"]:
                        v[f"violation:{field}:{k}"] += 1
                    self.fail(t, org, name, field, "genuine_violation:" + ",".join(kinds), detail)
        elif expect == "invalid":
            if ambiguous:
                v[f"unconfirmed:{field}"] += 1
                self.fail(t, org, name, field, "unconfirmed_valid_depends_on_ref_sibling_semantics",
                          "erroneous_json satisfies the schema, but the schema has $ref siblings (validator-version dependent)", "unresolved")
            else:
                v[f"genuine:{field}"] += 1
                self.fail(t, org, name, field, "expected_invalid_but_valid", "erroneous_json satisfies the schema", "valid")
        return state, kinds

    def evaluate(self, t, org, name, rec, external):
        v, m = self.val[(t, org)], self.m[(t, org)]
        missing = [f for f in REQUIRED_FIELDS[t] if f not in rec]
        if missing:
            v["missing_fields"] += 1
            self.fail(t, org, name, "record", "missing_required_fields", ",".join(sorted(missing)))
            return
        for f in ("schema", PRIMARY[t]) + (("erroneous_json",) if t == "fix" else ()) + (("ground_truth",) if t == "modify" else ()):
            if isinstance(rec[f], str):
                v["json_field_is_string"] += 1
                self.fail(t, org, name, f, "json_field_stored_as_string", "field holds a string where the file format has an object")
        sch_m, ext = schema_metrics(rec["schema"])
        self._refsib, self._tech_flag = sch_m["schema_ref_siblings"], False
        if self._refsib:
            v["examples_with_ref_siblings"] += 1
        if ext:
            v["examples_with_external_refs"] += 1
        for u in ext:
            external[u].add(name)
        primary = rec[PRIMARY[t]]
        row = {"task": t, "origin": org, "name": name, "schema_id": schema_id(name)[0], "schema_hash12": schema_hash(rec["schema"])[:12]}
        row.update(sch_m)
        row.update(instance_metrics(primary))
        cluster = schema_hash(rec["schema"])
        if t == "create":
            state, _ = self.vstatus(t, org, name, rec, "reference_json", "valid", external)
            text = rec["description"] if isinstance(rec["description"], str) else ""
            row["description_chars"], row["description_words"] = len(text), _words(text)
            if not text.strip():
                v["empty_text"] += 1
                self.fail(t, org, name, "description", "empty_description", "")
            ratio = _grounding(rec["reference_json"], text)
            row["grounding"] = round(ratio, 4)
            if ratio < self.args.grounding_threshold:
                v["below_grounding_threshold"] += 1
                self.fail(t, org, name, "description", "below_grounding_threshold", f"grounding {ratio:.3f} < {self.args.grounding_threshold}", "warning")
            self.texts[(t, org)].append(text)
        elif t == "fix":
            sv_state, _ = self.vstatus(t, org, name, rec, "valid_json", "valid", external)
            er_state, kinds = self.vstatus(t, org, name, rec, "erroneous_json", "invalid", external)
            same = rec["erroneous_json"] == rec["valid_json"]
            if same:
                v["identical_pair"] += 1
                self.fail(t, org, name, "pair", "erroneous_equals_valid", "")
            paths, kc, depths = change_summary(rec["valid_json"], rec["erroneous_json"])
            _k, vpaths = violation_info(rec["schema"], rec["erroneous_json"]) if er_state == sv.INVALID else ([], [])
            unrelated = sum(1 for p in paths if not _covered(p, vpaths)) if er_state == sv.INVALID else None
            for k, n in kc.items():
                self.kinds[(t, org, "change_kind")][k] += 1
            row.update(changed_paths=len(paths), changed_max_depth=max(depths, default=0),
                       changed_mean_depth=round(sum(depths) / len(depths), 3) if depths else 0,
                       unrelated_changes=unrelated, fix_error_keywords=";".join(sorted(set(kinds))),
                       fix_change_kinds=";".join(sorted(kc)))
        else:
            self.vstatus(t, org, name, rec, "data", "valid", external)
            gt_state, _ = self.vstatus(t, org, name, rec, "ground_truth", None, external, blocking=False)
            if rec["ground_truth"] == rec["data"]:
                v["identical_pair"] += 1
                self.fail(t, org, name, "pair", "ground_truth_equals_data", "")
            text = rec["instructions"] if isinstance(rec["instructions"], str) else ""
            if not text.strip():
                v["empty_text"] += 1
                self.fail(t, org, name, "instructions", "empty_instruction", "")
            row["instruction_chars"], row["instruction_words"] = len(text), _words(text)
            paths, kc, depths = change_summary(rec["data"], rec["ground_truth"])
            for k in kc:
                self.kinds[(t, org, "change_kind")][k] += 1
            low = text.lower()
            named = [p for p in paths if p and isinstance(p[-1], str)]
            row.update(changed_paths=len(paths), changed_max_depth=max(depths, default=0),
                       changed_mean_depth=round(sum(depths) / len(depths), 3) if depths else 0,
                       instruction_path_mentions=round(sum(1 for p in named if p[-1].lower() in low) / len(named), 4) if named else None,
                       modify_change_kinds=";".join(sorted(kc)), ground_truth_state=gt_state)
            self.texts[(t, org)].append(text)
        for k, val in row.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                m[k].append(val)
                self.cluster[(t, org)][k][cluster].append(val)
        if self._tech_flag:
            v["examples_with_technical_validation"] += 1
        row["has_ref_siblings"] = int(self._refsib > 0)
        self.rows["complexity"].append(row)
        self.records_seen[(t, org)].append((name, rec))

    # ---------------- C. diversity
    def diversity(self):
        D = []

        def add(section, task, origin, key, value):
            D.append({"section": section, "task": task, "origin": origin, "key": key, "value": value})

        def add_dups(section, task, origin, label, sizes):
            for k, val in dup_summary(sizes).items():       # the SAME four numbers, with the SAME definitions, for every row
                add(section, task, origin, f"{label}: {k}", val)
        for t in TASKS:
            exact_by_org = {}
            for org in ("original", "new"):
                recs = self.records_seen[(t, org)]
                hashes = [schema_hash(r["schema"]) for _, r in recs]
                sids = [schema_id(n)[0] for n, _ in recs]
                add("unique", t, org, "examples", len(recs))
                add("unique", t, org, "schema_content_hashes", len(set(hashes)))
                add("unique", t, org, "schema_ids", len(set(sids)))
                per_hash = Counter(hashes)
                for n_ex, n_sch in sorted(Counter(per_hash.values()).items()):
                    add("examples_per_schema_content", t, org, str(n_ex), n_sch)
                if org == "new":
                    for n_ref, n_sch in sorted(self.reuse["references_per_schema"].items()):
                        if t == "create":
                            add("new_references_per_schema_id", "all", org, str(n_ref), n_sch)
                    for sid, n in Counter(sids).most_common(10):
                        add("most_reused_schema_ids", t, org, sid[:80], n)
                keys = [task_keys(t, r) for _, r in recs]
                exact, content = Counter(k[0] for k in keys), Counter(k[1] for k in keys)
                refs = Counter((schema_hash(r["schema"]), reference_hash(t, r)) for _, r in recs)
                refs_any = Counter(reference_hash(t, r) for _, r in recs)
                exact_by_org[org] = set(exact)
                add_dups("duplicates_within", t, org, "exact (schema + task JSON)", exact.values())
                by_content = defaultdict(list)
                for (_e, c_key), (_n, r) in zip(keys, recs):
                    by_content[c_key].append(schema_hash(r["schema"]))
                add_dups("duplicates_within", t, org, "same task JSON under different schemas [exact on JSON; reported, not blocking]",
                         (len(h) for h in by_content.values() if len(set(h)) > 1))
                add_dups("duplicates_within", t, org, "same reference JSON on the same schema", refs.values())
                add_dups("duplicates_within", t, org, "same reference JSON on any schema", refs_any.values())
                if t in TEXT_FIELD:
                    texts = self.texts[(t, org)]
                    norm = Counter(" ".join(x.lower().split()) for x in texts)
                    add_dups("duplicates_within", t, org, f"identical {TEXT_FIELD[t]} (exact string)", Counter(texts).values())
                    add_dups("duplicates_within", t, org, f"identical {TEXT_FIELD[t]} after lower-casing/whitespace folding [structural]", norm.values())
                if t == "fix" or t == "modify":
                    for k, n in sorted(self.kinds[(t, org, "change_kind")].items()):
                        add("change_kinds", t, org, k, n)
                for k, n in sorted(self.kinds[(t, org, "violation_keyword")].items()):
                    add("fix_error_keywords" if t == "fix" else "violation_keywords", t, org, k, n)
            mixed = [m for m in self.dup_groups.get(t, {}).values() if len({o for _, o in m}) > 1 and "original" in {o for _, o in m} and "new" in {o for _, o in m}]
            o_n = [sum(1 for _, o in m if o == "original") for m in mixed]
            n_n = [sum(1 for _, o in m if o == "new") for m in mixed]
            add("duplicates_between", t, "original-vs-new", "exact: groups containing both origins", len(mixed))
            add("duplicates_between", t, "original-vs-new", "exact: original records in those groups", sum(o_n))
            add("duplicates_between", t, "original-vs-new", "exact: new records in those groups (new records that duplicate an original)", sum(n_n))
            add("duplicates_between", t, "original-vs-new", "exact: cross-origin pairs", sum(a_ * b_ for a_, b_ in zip(o_n, n_n)))
            add_dups("duplicates_merged", t, "merged train (original+new)", "exact (schema + task JSON)", (len(m) for m in self.dup_groups.get(t, {}).values()))
            ho = {schema_hash(r["schema"]) for _, r in self.records_seen[(t, "original")]}
            hn = {schema_hash(r["schema"]) for _, r in self.records_seen[(t, "new")]}
            add("reuse", t, "new", "new schema contents that already occur in original train", len(ho & hn))
            add("reuse", t, "new", "new schema contents not in original train", len(hn - ho))
            add("reuse", t, "new", "new examples on a schema that has original train examples",
                sum(1 for _, r in self.records_seen[(t, "new")] if schema_hash(r["schema"]) in ho))
            add("schema_overlap_with_test", t, "all", "train schema contents also in test (original, published property)",
                len({schema_hash(r["schema"]) for o in ("original",) for _, r in self.records_seen[(t, o)]} & self.test_hashes))
        for org in ("original", "new"):
            by_stem = defaultdict(dict)
            for t in TASKS:
                for n, r in self.records_seen[(t, org)]:
                    by_stem[n[:-5] if org == "original" else n[:-5]][t] = r
            groups = list(by_stem.values())
            full = [g for g in groups if len(g) == 3]
            add("cross_task", "all", org, "example keys (file stems) present in all three tasks", len(full))
            add("cross_task", "all", org, "example keys in fewer than three tasks", len(groups) - len(full))
            same = sum(1 for g in full if g["create"]["reference_json"] == g["fix"]["valid_json"] == g["modify"]["data"])
            add("cross_task", "all", org, "of those: create.reference_json == fix.valid_json == modify.data (exact)", same)
            sch = sum(1 for g in full if schema_hash(g["create"]["schema"]) == schema_hash(g["fix"]["schema"]) == schema_hash(g["modify"]["schema"]))
            add("cross_task", "all", org, "of those: identical schema in the three tasks", sch)
            hs = [{schema_hash(r["schema"]) for _, r in self.records_seen[(t, org)]} for t in TASKS]
            add("cross_task", "all", org, "schema contents used by all three tasks", len(hs[0] & hs[1] & hs[2]))
        self.rows["dups"] = D

    # ---------------- D. statistics
    def statistics(self):
        a = self.args
        self.stat_rows, self.stat_summary = [], {}
        for t in TASKS:
            names = sorted(set(self.m[(t, "original")]) | set(self.m[(t, "new")]))
            for k in names:
                so, sn = _stats(self.m[(t, "original")].get(k, [])), _stats(self.m[(t, "new")].get(k, []))
                row = {"task": t, "metric": k}
                for tag, s in (("original", so), ("new", sn)):
                    for f in ("n", "mean", "median", "sd", "min", "max", "p25", "p75", "p90", "p95"):
                        row[f"{tag}_{f}"] = round(s[f], 4) if isinstance(s.get(f), float) else s.get(f)
                if so.get("n") and sn.get("n"):
                    row["delta_mean"] = round(sn["mean"] - so["mean"], 4)
                    row["relative_delta_mean_pct"] = round(100 * (sn["mean"] - so["mean"]) / so["mean"], 2) if so["mean"] else None
                    row["delta_median"] = round(sn["median"] - so["median"], 4)
                    ci = _cluster_bootstrap(self.cluster[(t, "original")][k], self.cluster[(t, "new")][k], a.bootstrap, a.seed)
                    row["delta_mean_ci95_cluster_bootstrap"] = f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
                    row["original_schemas"], row["new_schemas"] = len(self.cluster[(t, "original")][k]), len(self.cluster[(t, "new")][k])
                self.stat_rows.append(row)

    # ---------------- E. verdict
    @staticmethod
    def stable(task, counter):
        """Every counter present (0 when it never happened), so consumers never have to guess which keys exist."""
        fields = {"create": ("reference_json",), "fix": ("valid_json", "erroneous_json"), "modify": ("data", "ground_truth")}[task]
        out = {k: 0 for k in ("examples", "parseable", "unparseable", "missing_fields", "empty_text", "identical_pair", "json_field_is_string",
                              "below_grounding_threshold", "technical:malformed_schema", "technical:unresolvable_external_ref", "technical:validator_error",
                              "examples_with_technical_validation", "examples_with_external_refs", "examples_with_ref_siblings", "audit_errors")}
        out.update({f"{f}:{st}": 0 for f in fields for st in (sv.VALID, sv.INVALID, sv.TECHNICAL)})
        out.update({f"{p}:{f}": 0 for f in fields for p in ("genuine", "unconfirmed")})
        out.update(counter)
        return out

    def validator_info(self):
        try:
            from importlib.metadata import version
            installed = version("jsonschema")
        except Exception:
            installed = None
        locked = None
        try:
            import re
            lock = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uv.lock")
            m = re.search(r'name = "jsonschema"\nversion = "([^"]+)"', open(lock, encoding="utf-8").read())
            locked = m.group(1) if m else None
        except OSError:
            pass
        try:
            import jsonschema
            dialect = jsonschema.validators.validator_for({}).__name__
        except Exception:
            dialect = None
        return {"jsonschema_installed": installed, "jsonschema_in_uv_lock": locked, "default_dialect_without_$schema": dialect}

    def finish(self):
        a = self.args
        info = self.validator_info()
        self.prov["validator"] = info
        same = info["jsonschema_installed"] is not None and info["jsonschema_installed"] == info["jsonschema_in_uv_lock"]
        self.check("E1", "Validator version equals the one pinned in uv.lock (schemas with `$ref` siblings are judged differently by jsonschema < 4.18)",
                   "INFO" if same else "WARN", "independent", json.dumps(info), critical=False)
        for t in TASKS:
            for org in ("new", "original"):
                c = self.val[(t, org)]
                genuine = sum(n for k, n in c.items() if k.startswith("genuine:"))
                unconfirmed = sum(n for k, n in c.items() if k.startswith("unconfirmed:"))
                struct = c["missing_fields"] + c["unparseable"] + c["empty_text"] + c["identical_pair"] + c["json_field_is_string"]
                errs = c["audit_errors"]
                detail = f"genuine {genuine}, structural {struct}, audit errors {errs}"
                if org == "new":
                    self.check(f"D-{t}-new", f"{t}: NEW examples have no genuine violation and no structural defect", "FAIL" if (genuine or struct or errs) else "PASS",
                               "independent", detail)
                else:
                    self.check(f"D-{t}-original", f"{t}: ORIGINAL examples: genuine violations / structural defects (reported as found, never altered, never treated as valid)",
                               "WARN" if (genuine or struct) else "PASS", "independent", detail, critical=False)
                    if errs:
                        self.check(f"D-{t}-original-errors", f"{t}: audit could not evaluate some ORIGINAL examples", "FAIL", "independent", f"{errs} audit errors")
                tech = c["examples_with_technical_validation"]
                self.check(f"U-{t}-{org}", f"{t}: {org} examples whose validity could not be established (technical failure or `$ref`-sibling dependent result)",
                           "UNRESOLVED" if (tech or unconfirmed) else "PASS", "independent",
                           f"{tech} examples with a technical validation failure, {unconfirmed} unconfirmed fields (never counted as valid or invalid)")
        unknown = [f"{t}/{n}" for t in TASKS for n, o in self.origin[t].items() if o == "unknown"]
        integrity_ids = ("I0", "A1", "A2", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C4")
        c_by = lambda pred: [c for c in self.checks if pred(c)]
        int_fail = c_by(lambda c: c["status"] == "FAIL" and (c["id"] in integrity_ids or c["id"].startswith("P")))
        int_incomplete = c_by(lambda c: c["status"] == "NOT_VERIFIED" and c["critical"])
        int_notes = c_by(lambda c: c["status"] == "NOT_VERIFIED" and not c["critical"])
        if int_fail:
            self.integrity_verdict = "NOT INTACT"
        elif int_incomplete:
            self.integrity_verdict = "INCOMPLETE (checks could not be completed)"
        elif int_notes:
            self.integrity_verdict = "INTACT (not independently verified: " + "; ".join(c["id"] for c in int_notes) + ")"
        else:
            self.integrity_verdict = "INTACT"
        self.integrity_reasons = [f"{c['id']}: {c['title']} -> {c['status']}" for c in int_fail + int_incomplete + int_notes]
        blocking = int_fail + int_incomplete + c_by(lambda c: c["critical"] and c["status"] == "FAIL" and c["id"] not in integrity_ids and not c["id"].startswith("P")) \
            + c_by(lambda c: c["status"] == "UNRESOLVED")
        seen = set()
        blocking = [c for c in blocking if not (c["id"] in seen or seen.add(c["id"]))]
        open_defects = c_by(lambda c: c["status"] == "WARN" and c["id"] != "E1")
        if blocking:
            self.readiness = "NOT READY FOR TRAINING"
        elif open_defects:
            self.readiness = "REVIEW REQUIRED (open defects in the ORIGINAL dataset; nothing was changed or excluded)"
        else:
            self.readiness = "READY FOR TRAINING (no blocking finding; see limitations)"
        self.readiness_reasons = [f"{c['id']}: {c['title']} -> {c['status']}" for c in blocking] + \
                                 [f"{c['id']}: {c['title']} -> {c['status']} ({c['detail'][:160]})" for c in open_defects]
        limits = [f"{c['id']}: {c['title']}" for c in c_by(lambda c: c["status"] == "UNRESOLVED")]
        if info["jsonschema_installed"] != info["jsonschema_in_uv_lock"]:
            limits.append("E1: validator version differs from uv.lock")
        n_sib = sum(self.val[(t, o)]["examples_with_ref_siblings"] for t in TASKS for o in ("new", "original"))
        if n_sib and info["jsonschema_installed"] != info["jsonschema_in_uv_lock"]:
            limits.append(f"{n_sib} examples have a schema with `$ref` siblings: their VALID results are also version dependent")
        if any(c["id"] == "A3" and c["status"] != "PASS" for c in self.checks):
            limits.append("A3: chunk chain not independently verified")
        if self.integrity_verdict.startswith(("NOT", "INCOMPLETE")):
            self.comparability = "NOT ESTABLISHED (the datasets being compared are not verified)"
        elif limits:
            self.comparability = "LIMITED"
        else:
            self.comparability = "ESTABLISHED (for the reported metrics)"
        self.comparability_reasons = limits
        self.verdict = self.readiness
        self.verdict_reasons = self.readiness_reasons
        self.summary = {
            "format": FORMAT, "verdict": self.readiness, "verdict_reasons": self.readiness_reasons,
            "data_integrity": {"verdict": self.integrity_verdict, "reasons": self.integrity_reasons},
            "training_readiness": {"verdict": self.readiness, "reasons": self.readiness_reasons},
            "scientific_comparability": {"verdict": self.comparability, "reasons": self.comparability_reasons,
                                         "always_true": ["new examples come from a different generator model and a controlled-reuse policy",
                                                         "differences are descriptive, not causal"]},
            "checks": self.checks, "counts": self.counts,
            "validation": {f"{t}/{o}": self.stable(t, c) for (t, o), c in self.val.items()},
            "validator": info, "external_references": self.external_refs,
            "reuse": {"cap": a.cap, "references_per_schema_id": dict(self.reuse["references_per_schema"]),
                      "examples_per_task_per_schema_id": dict(self.reuse["per_task_per_schema"])},
            "unknown_origin_files": unknown[:20],
            "metric_definitions": METRIC_DEFINITIONS}


def _compact_sorted(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _grounding(reference, description):
    from data_generation.benchmark_builder import _grounding_ratio
    return _grounding_ratio(reference, description)


# ------------------------------------------------------------------ writing

def _write_csv(path, rows, fields=None):
    fields = fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else ["empty"])
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _pct(n, d):
    return f"{100 * n / d:.2f}%" if d else "n/a"


def write_report(path, au):
    a, s = au.args, au.summary
    L = ["# Scale Only dataset quality audit", "",
         f"**Data integrity: {au.integrity_verdict}**", ""] + [f"- {r}" for r in au.integrity_reasons[:10]] + [""] + \
        [f"**Training readiness: {au.readiness}**", ""] + [f"- {r}" for r in au.readiness_reasons[:25]] + [""] + \
        [f"**Scientific comparability (original vs new): {au.comparability}**", ""] + [f"- {r}" for r in au.comparability_reasons[:10]] + \
        ["- Always: the new examples come from a different generator model (Granite) and a controlled-reuse policy; differences are descriptive, not causal.", ""]
    L += ["This audit is read-only: no example was changed, filtered or deleted, no model was called, nothing was uploaded.",
          "'independent' = verified by this audit from the files; 'inherited' = only compared with what the merge/chain manifests claim.", "",
          "## 1. Executive summary", ""]
    for t in TASKS:
        n, o = au.val[(t, "new")], au.val[(t, "original")]
        g = lambda c, p: sum(x for k, x in c.items() if k.startswith(p))
        L.append(f"- **{t}**: new {n['examples']} / original {o['examples']} train examples. "
                 f"New: genuine failures {g(n, 'genuine:')}, unconfirmed fields {g(n, 'unconfirmed:')}, examples with unresolved (technical) validation {n['examples_with_technical_validation']}. "
                 f"Original: genuine failures {g(o, 'genuine:')}, unconfirmed fields {g(o, 'unconfirmed:')}, examples with unresolved (technical) validation {o['examples_with_technical_validation']}.")
    L.append("")
    L.append("Counting units: *examples* are records; *fields* are validations of one field of one record (a Fix example has two validated fields, a Modify example two, a Create one); "
             "*genuine* = a resolved validation contradicts the task's requirement and does not depend on `$ref`-sibling semantics; *unconfirmed* = the result depends on the validator version; "
             "*unresolved (technical)* = a remote `$ref` / malformed schema / validator error prevented a verdict. Field-level technical events are in `summary.json`.")
    L += ["", "## 2. Dataset composition", "", "| Task | Original train | New train | Merged train | Test |", "|---|---:|---:|---:|---:|"]
    for t in TASKS:
        orig = sum(1 for o in au.origin[t].values() if o == "original")
        new = sum(1 for o in au.origin[t].values() if o == "new")
        L.append(f"| {t} | {orig} | {new} | {au.counts[t]['train']} | {au.counts[t]['test']} |")
    L += ["", "## 3. Integrity and provenance", "", "| Id | Check | Result | Basis |", "|---|---|---|---|"]
    for c in au.checks:
        if not c["id"].startswith("D-"):
            L.append(f"| {c['id']} | {c['title']} | **{c['status']}** | {c['basis']} |")
    L += [""] + [f"- {c['id']}: {c['detail']}" for c in au.checks if c["detail"] and not c["id"].startswith("D-")][:20]
    L += ["", "Provenance that needs the chunk directories (state files, seals, generation ledger) is verified only with `--chunks-root`; "
          "otherwise the chain is the manifest's claim and check A3 says NOT_VERIFIED.", "",
          "## 4. JSON and schema validity", "",
          "Three-way validation (`schema_validation.validate_instance`): a TECHNICAL result (unresolvable `$ref`, malformed schema, validator crash) is never counted as invalid and never as valid.", "",
          "| Task | Origin | Examples | Parseable | Missing fields | Primary valid | Primary INVALID | Primary TECHNICAL (fields) | Examples with technical validation | Examples whose schema has an external $ref | Malformed schema (fields) | Unresolvable $ref (fields) | Validator error (fields) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in TASKS:
        for org in ("original", "new"):
            c = au.val[(t, org)]
            p = PRIMARY[t]
            L.append(f"| {t} | {org} | {c['examples']} | {_pct(c['parseable'], c['examples'])} | {c['missing_fields']} | {c[p + ':valid']} | {c[p + ':invalid']} | "
                     f"{c[p + ':technical']} | {c['examples_with_technical_validation']} | {c['examples_with_external_refs']} | {c['technical:malformed_schema']} | "
                     f"{c['technical:unresolvable_external_ref']} | {c['technical:validator_error']} |")
    ext = s["external_references"]
    L += ["", f"External `$ref` documents involved: {ext['distinct_external_documents']}. SHA256 evidence of the documents fetched in this process (evidence only; it is not part of the dataset seals):", ""]
    for u, h in list(ext["documents_fetched_by_this_process"].items())[:40]:
        L.append(f"- `{u}` sha256 `{h}`")
    L += ["", "## 5. Task-specific correctness", ""]
    for t in TASKS:
        for org in ("original", "new"):
            c = au.val[(t, org)]
            if t == "create":
                L.append(f"- create/{org}: reference_json valid {c['reference_json:valid']}, INVALID {c['reference_json:invalid']}, technical {c['reference_json:technical']}; "
                         f"empty descriptions {c['empty_text']}; below grounding threshold {a.grounding_threshold}: {c['below_grounding_threshold']} ({_pct(c['below_grounding_threshold'], c['examples'])}).")
            elif t == "fix":
                L.append(f"- fix/{org}: valid_json valid {c['valid_json:valid']}/INVALID {c['valid_json:invalid']}/technical {c['valid_json:technical']}; "
                         f"erroneous_json INVALID (as required) {c['erroneous_json:invalid']}, actually valid {c['erroneous_json:valid']}, technical {c['erroneous_json:technical']}; "
                         f"identical pairs {c['identical_pair']}.")
            else:
                L.append(f"- modify/{org}: data valid {c['data:valid']}/INVALID {c['data:invalid']}/technical {c['data:technical']}; ground_truth valid {c['ground_truth:valid']}, "
                         f"schema-violating {c['ground_truth:invalid']} (allowed by the original Modify task), technical {c['ground_truth:technical']}; "
                         f"empty instructions {c['empty_text']}; ground_truth == data {c['identical_pair']}.")
    L += ["", "Grounding is a lexical metric (see definitions). Modify correctness (does the instruction describe the edit?) is **not automatically verified**; "
          "only the lexical `instruction_path_mentions` proxy is reported. Fix: the audit proves the pair differs and that erroneous_json violates the schema, not that the "
          "fix is the 'right' one beyond the structural coverage check.", "",
          "## 6. Complexity comparisons", "", "Descriptive statistics, original vs new (means; full N/median/SD/min/max/P25/P75/P90/P95 and a 95% schema-cluster bootstrap interval of the difference are in `metrics_by_task.csv`).", ""]
    for t in TASKS:
        L += [f"**{t}**", "", "| Metric | Original mean | New mean | Delta | Rel. delta | 95% CI of delta (schema clusters) |", "|---|---:|---:|---:|---:|---|"]
        for r in au.stat_rows:
            if r["task"] == t and r["metric"] in KEY_METRICS and "delta_mean" in r:
                L.append(f"| {r['metric']} | {r['original_mean']} | {r['new_mean']} | {r['delta_mean']} | {r['relative_delta_mean_pct']}% | {r['delta_mean_ci95_cluster_bootstrap']} |")
        L.append("")
    L += ["## 7. Diversity and duplication", ""]
    for sec in ("unique", "duplicates_within", "duplicates_between", "reuse", "cross_task"):
        rows = [r for r in au.rows["dups"] if r["section"] == sec]
        if rows:
            L += [f"**{sec}**", "", "| Task | Origin | Item | Value |", "|---|---|---|---:|"] + [f"| {r['task']} | {r['origin']} | {r['key']} | {r['value']} |" for r in rows] + [""]
    L += ["Duplicate definitions: *exact* = same canonical JSON of schema + task fields; *same reference on the same schema* = same reference object (exact); "
          "*same JSON under different schemas* = exact on the JSON only; lower-casing/whitespace folding = structural. No approximate/embedding similarity is used. "
          "Two examples are never called duplicates merely because they share a schema.", "",
          "## 8. Findings requiring investigation", ""]
    bad = [c for c in au.checks if c["status"] in ("FAIL", "UNRESOLVED", "WARN", "NOT_VERIFIED")]
    L += [f"- {c['id']} [{c['status']}]: {c['title']} - {c['detail'][:400]}" for c in bad] or ["- none"]
    L += [f"- {len(au.rows['failures'])} rows in `validation_failures.csv` (genuine, technical and warning rows; the file is the complete list).", "",
          "## 9. Scientific limitations", "",
          "- Schema compliance and structural comparisons do not establish semantic correctness of descriptions, instructions or fixes.",
          "- Complexity metrics are counts; no single metric represents task difficulty. Schema complexity (static, `$ref` not followed) is different from instance complexity.",
          "- The new examples come from a different generator model (Granite) and a controlled-reuse policy; differences from the baseline are not attributed to causes here.",
          "- Examples from the same schema are not independent; intervals resample whole schemas, and effective sample sizes are the schema counts in `metrics_by_task.csv`.",
          "- The audit cannot show that the original dataset's generation settings equal ours (not recoverable).",
          "- Remote `$ref` documents are not covered by any seal; their content at audit time is what was validated (SHA256 listed above).", "",
          "## 10. Recommendations before training", "",
          "- Do not train while the verdict is NOT READY or any UNRESOLVED validation remains; rerun with network access to the referenced schema hosts, or decide explicitly how to treat those examples (this audit never filters).",
          "- Review the rows of `validation_failures.csv` with status other than 'warning' and decide per case (original-dataset defects are not ours to alter silently).",
          "- Run with `--chunks-root` so the chain itself is verified, if not already done.", "",
          "## Metric definitions", ""] + [f"- `{k}`: {v}" for k, v in METRIC_DEFINITIONS.items()]
    with open(path, "x", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


KEY_METRICS = ("schema_bytes", "schema_properties", "schema_required", "schema_depth", "schema_constraints", "schema_enums", "schema_refs",
               "schema_external_refs", "json_leaves", "json_depth", "json_fields", "json_chars", "grounding", "description_words",
               "instruction_words", "changed_paths", "changed_max_depth", "unrelated_changes", "instruction_path_mentions")


def git_state():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "src", "tests"], capture_output=True, text=True, timeout=10).stdout.strip())
        return {"head": head or None, "uncommitted_changes_in_src_or_tests": dirty}
    except Exception:
        return {"head": None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--jsonl", help="the JSONL export of the merged dataset")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-revision", default=msd.PINNED_REVISION)
    ap.add_argument("--expected-new", type=msd._parse_counts, default=msd.EXPECTED_NEW, metavar="create=N,fix=N,modify=N")
    ap.add_argument("--expected-baseline-train", type=int, default=msd.EXPECTED_BASELINE["train"])
    ap.add_argument("--expected-baseline-test", type=int, default=msd.EXPECTED_BASELINE["test"])
    ap.add_argument("--expected-chunks", type=int, default=msd.EXPECTED_CHUNKS)
    ap.add_argument("--cap", type=int, default=3, help="max new references per schema id (the generation cap)")
    ap.add_argument("--grounding-threshold", type=float, default=0.9)
    ap.add_argument("--bootstrap", type=int, default=400, help="schema-cluster bootstrap replicates (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true", help="forbid network access: examples needing a remote $ref become TECHNICAL")
    ap.add_argument("--chunks-root"); ap.add_argument("--anchor-file"); ap.add_argument("--reserved-file")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out_dir)
    sources = [p for p in (args.baseline, args.new, args.merged, args.jsonl, args.chunks_root, args.anchor_file, args.reserved_file) if p]
    if os.path.lexists(out):
        sys.exit(f"{out} already exists. Refusing to overwrite or reuse it; choose a new --out-dir.")
    for p in sources:
        if os.path.exists(p) and msd._overlap(out, p):
            sys.exit("--out-dir must be outside every input directory (and must not contain one).")
    missing = [p for p in sources if not os.path.exists(p)]     # a missing input is a finding in the report, not a crash
    os.makedirs(os.path.dirname(out), exist_ok=True)
    created = {}
    staging = tempfile.mkdtemp(prefix=".merge_staging_", dir=os.path.dirname(out))
    created[staging] = os.path.realpath(staging)
    try:
        au = Audit(args)
        au.missing_inputs = missing
        au.run()
        write_report(os.path.join(staging, "quality_report.md"), au)
        with open(os.path.join(staging, "summary.json"), "x", encoding="utf-8") as f:
            json.dump(au.summary, f, indent=1, default=str)
        _write_csv(os.path.join(staging, "metrics_by_task.csv"), au.stat_rows)
        _write_csv(os.path.join(staging, "validation_failures.csv"), au.rows["failures"],
                   ["task", "origin", "name", "field", "status", "category", "reason", "remote_url"])
        _write_csv(os.path.join(staging, "complexity_metrics.csv"), au.rows["complexity"])
        _write_csv(os.path.join(staging, "duplicates_and_reuse.csv"), au.rows["dups"], ["section", "task", "origin", "key", "value"])
        prov = {"format": FORMAT, "verdict": au.verdict, "data_integrity": au.integrity_verdict, "training_readiness": au.readiness,
                "scientific_comparability": au.comparability, "provenance": au.prov, "checks": au.checks,
                "inputs": {"baseline": os.path.basename(os.path.abspath(args.baseline)), "new": os.path.basename(os.path.abspath(args.new)),
                           "merged": os.path.basename(os.path.abspath(args.merged))},
                "audit": {"script_sha256": sha256_file(os.path.abspath(__file__)), "git": git_state(),
                          "arguments": {k: v for k, v in vars(args).items() if k not in ("baseline", "new", "merged", "jsonl", "out_dir", "chunks_root", "anchor_file", "reserved_file")}},
                "remote_documents_sha256": au.external_refs["documents_fetched_by_this_process"]}
        with open(os.path.join(staging, "provenance_report.json"), "x", encoding="utf-8") as f:
            json.dump(prov, f, indent=1, default=str)
        msd._publish(staging, out)
    except BaseException:
        msd._remove_own_staging(staging, created, [os.path.abspath(p) for p in sources])
        raise
    print(f"Data integrity: {au.integrity_verdict}\nTraining readiness: {au.readiness}\nScientific comparability: {au.comparability}\n"
          f"report: {os.path.join(out, 'quality_report.md')}")
    return 2 if au.readiness.startswith("NOT") else 3 if au.readiness.startswith("REVIEW") else 0


if __name__ == "__main__":
    sys.exit(main())
