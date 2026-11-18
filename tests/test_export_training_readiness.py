"""Offline, deterministic readiness tests for the training exporter (`src/export_training.py`).

The exporter, the prompt builders and the metrics are NOT modified; these tests only pin their current behaviour so a
silent change (prompt text, target choice, ordering, skipping, serialization) is noticed before a training run.
No network, no model, no GPU. Synthetic records live in a temporary directory.

Tests marked "real data" run only when the untracked local release (release/scale-only-7k) exists; otherwise they are
SKIPPED (a skip is not a pass).
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "lm_eval_tasks"))

import export_training as ET                                     # noqa: E402
import utils as lm_utils                                         # noqa: E402  (lm_eval_tasks/utils.py)
from evaluations import metrics as repo_metrics                  # noqa: E402
from data_generation.published import decode_row, encode_record  # noqa: E402

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}, "r": {"$ref": "#/definitions/x"}},
          "definitions": {"x": {"type": "string"}}, "required": ["n"]}
SCHEMA_EXT = {"type": "object", "properties": {"e": {"$ref": "http://example.invalid/ext.json#/a"}}}
REL = os.path.join(ROOT, "release", "scale-only-7k")


def rec_create(i, schema=SCHEMA):
    return {"name": f"c{i}", "source": "schemas", "subset": None, "schema": schema, "description": f"desc {i} é",
            "reference_json": {"n": i, "s": "é "}}


def rec_fix(i):
    return {"name": f"f{i}", "source": "schemas", "subset": None, "schema": SCHEMA, "erroneous_json": {"n": "bad"},
            "valid_json": {"n": i}, "description": f"fix desc {i}"}


def rec_modify(i, truth=None):
    return {"name": f"m{i}", "source": "schemas", "subset": None, "schema": SCHEMA, "data": {"n": 1},
            "instructions": f"set n to {i}", "ground_truth": {"n": i} if truth is None else truth, "modification": {"op": "set", "path": ["n"]}}


class Workdir:
    """Builds data/<dataset>/{train,test}/*.json under a temp cwd and runs the real exporter there."""

    def __init__(self, test_case):
        self.tmp = tempfile.TemporaryDirectory()
        test_case.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def put(self, dataset, split, name, record):
        d = os.path.join(self.root, "data", dataset, split)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            if isinstance(record, str):
                f.write(record)
            else:
                json.dump(record, f)

    def export(self, out="out/train.jsonl", fmt="chat"):
        old = os.getcwd()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                ET.export({"output": {"path": os.path.join(self.root, out), "format": fmt}})
        finally:
            os.chdir(old)
        with open(os.path.join(self.root, out), "rb") as f:
            raw = f.read()
        return [json.loads(x) for x in raw.decode("utf-8").splitlines()], raw, buf.getvalue()


def populate(w, n=3):
    for i in range(n):
        w.put("benchmark_create", "train", f"{i:03d}.json", rec_create(i))
        w.put("benchmark_fix", "train", f"{i:03d}.json", rec_fix(i))
        w.put("benchmark_modify", "train", f"{i:03d}.json", rec_modify(i))


class ExportStructureAndTargets(unittest.TestCase):
    def setUp(self):
        self.w = Workdir(self)
        populate(self.w)
        self.rows, self.raw, self.log = self.w.export()

    def test_counts_task_order_and_single_combined_file(self):
        self.assertEqual([r["task_type"] for r in self.rows], ["create"] * 3 + ["fix"] * 3 + ["modify"] * 3)
        self.assertIn("Wrote 9 training examples", self.log)

    def test_chat_roles_keys_and_non_empty_content(self):
        for r in self.rows:
            self.assertEqual(set(r), {"messages", "task_type"})
            self.assertEqual([m["role"] for m in r["messages"]], ["system", "user", "assistant"])
            for m in r["messages"]:
                self.assertTrue(m["content"].strip())

    def test_target_field_per_task(self):
        for r in self.rows:
            i = int(json.loads(r["messages"][2]["content"]).get("n", -1))
            tgt = json.loads(r["messages"][2]["content"])
            if r["task_type"] == "create":
                self.assertEqual(tgt, {"n": i, "s": "é "})      # reference_json
            elif r["task_type"] == "fix":
                self.assertEqual(tgt, {"n": i})                       # valid_json, never erroneous_json
                self.assertNotIn('"bad"', r["messages"][2]["content"])
            else:
                self.assertEqual(tgt, {"n": i})                       # ground_truth

    def test_fix_prompt_contains_erroneous_json_not_the_answer(self):
        fix = [r for r in self.rows if r["task_type"] == "fix"]
        for r in fix:
            self.assertIn(json.dumps({"n": "bad"}, indent=2), r["messages"][1]["content"])

    def test_target_serialization_is_compact_default_separators_ascii_escaped(self):
        c = self.rows[0]["messages"][2]["content"]
        self.assertEqual(c, json.dumps({"n": 0, "s": "é "}))
        self.assertEqual(c, '{"n": 0, "s": "\\u00e9\\u2028"}')      # ", " and ": " separators, ensure_ascii True
        self.assertNotIn("\n", c)

    def test_prompt_schema_uses_indent2_and_leaves_refs_unresolved(self):
        u = self.rows[0]["messages"][1]["content"]
        self.assertIn(json.dumps(SCHEMA, indent=2), u)
        self.assertIn('"$ref": "#/definitions/x"', u)

    def test_external_ref_is_kept_verbatim_and_not_fetched(self):
        w = Workdir(self)
        w.put("benchmark_create", "train", "a.json", rec_create(1, SCHEMA_EXT))
        rows, _, _ = w.export()
        self.assertIn("http://example.invalid/ext.json#/a", rows[0]["messages"][1]["content"])

    def test_modify_target_may_violate_its_schema_and_is_exported_as_is(self):
        w = Workdir(self)
        w.put("benchmark_modify", "train", "a.json", rec_modify(1, truth={"n": "not an int"}))
        rows, _, _ = w.export()
        self.assertEqual(json.loads(rows[0]["messages"][2]["content"]), {"n": "not an int"})

    def test_null_gold_is_exported_as_the_string_null(self):
        w = Workdir(self)
        w.put("benchmark_modify", "train", "a.json", rec_modify(1, truth=None) | {"ground_truth": None})
        rows, _, _ = w.export()
        self.assertEqual(rows[0]["messages"][2]["content"], "null")


class ExportOrderingDeterminismAndSkipping(unittest.TestCase):
    def test_order_is_sorted_file_name_not_insertion_order_and_no_shuffle(self):
        w = Workdir(self)
        for name, i in (("b.json", 2), ("a.json", 1), ("c.json", 3)):
            w.put("benchmark_create", "train", name, rec_create(i))
        rows, _, _ = w.export()
        self.assertEqual([json.loads(r["messages"][2]["content"])["n"] for r in rows], [1, 2, 3])

    def test_two_runs_are_byte_identical(self):
        w = Workdir(self)
        populate(w, 5)
        _, a, _ = w.export("o1/t.jsonl")
        _, b, _ = w.export("o2/t.jsonl")
        self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())

    def test_known_silent_skips_are_pinned(self):
        """Documents CURRENT behaviour: the exporter drops records without the gold key, non-.json files and
        missing datasets WITHOUT an error. A caller must compare counts with the source (see next test)."""
        w = Workdir(self)
        w.put("benchmark_create", "train", "ok.json", rec_create(1))
        bad = rec_create(2)
        del bad["reference_json"]
        w.put("benchmark_create", "train", "nogold.json", bad)
        w.put("benchmark_create", "train", "note.txt", "ignored")
        rows, _, log = w.export()
        self.assertEqual(len(rows), 1)
        self.assertIn("[create] exported 1 examples", log)
        self.assertNotIn("benchmark_fix", log)            # absent dataset: no message at all

    def test_malformed_json_record_aborts_instead_of_being_skipped(self):
        w = Workdir(self)
        w.put("benchmark_create", "train", "broken.json", "{not json")
        with self.assertRaises(ValueError):
            w.export()

    def test_exporter_never_reads_the_test_split(self):
        w = Workdir(self)
        w.put("benchmark_create", "train", "t.json", rec_create(1))
        w.put("benchmark_create", "test", "x.json", rec_create(99))
        w.put("benchmark_create", "test", "broken.json", "{not json")      # would raise if it were read
        rows, raw, _ = w.export()
        self.assertEqual(len(rows), 1)
        self.assertNotIn(b"desc 99", raw)


class PromptEquivalenceWithLmEval(unittest.TestCase):
    """Training prompt == the prompt lm_eval_tasks builds for the same record (system + blank line + user)."""

    def _prompt(self, row):
        m = row["messages"]
        return f"{m[0]['content']}\n\n{m[1]['content']}"

    def test_create_fix_modify_prompts_match_lm_eval_text(self):
        w = Workdir(self)
        w.put("benchmark_create", "train", "a.json", rec_create(1))
        w.put("benchmark_fix", "train", "a.json", rec_fix(1))
        w.put("benchmark_modify", "train", "a.json", rec_modify(1))
        w.put("benchmark_create", "train", "b.json", rec_create(2) | {"description": None, "instructions": None})
        rows, _, _ = w.export()
        by = {}
        for r in rows:
            by.setdefault(r["task_type"], []).append(r)
        self.assertEqual(self._prompt(by["create"][0]), lm_utils.doc_to_text_create(rec_create(1)))
        self.assertEqual(self._prompt(by["fix"][0]), lm_utils.doc_to_text_fix(rec_fix(1)))
        self.assertEqual(self._prompt(by["modify"][0]), lm_utils.doc_to_text_modify(rec_modify(1)))
        self.assertEqual(self._prompt(by["create"][1]),
                         lm_utils.doc_to_text_create(rec_create(2) | {"description": None, "instructions": None}))

    def test_lm_eval_targets_match_exported_targets(self):
        w = Workdir(self)
        populate(w, 1)
        rows, _, _ = w.export()
        got = {r["task_type"]: r["messages"][2]["content"] for r in rows}
        self.assertEqual(got["create"], lm_utils.doc_to_target_create(rec_create(0)))
        self.assertEqual(got["fix"], lm_utils.doc_to_target_fix(rec_fix(0)))
        self.assertEqual(got["modify"], lm_utils.doc_to_target_modify(rec_modify(0)))


class PublishedFormatRoundTrip(unittest.TestCase):
    def test_encode_decode_then_export_equals_export_of_original_record(self):
        for task, rec in (("create", rec_create(1)), ("fix", rec_fix(1)), ("modify", rec_modify(1))):
            enc = encode_record(rec)
            self.assertIsInstance(enc["schema"], str)                                # JSON fields are stringified
            self.assertEqual(decode_row(task, enc), rec)                             # lossless for the exported fields
            a, b = Workdir(self), Workdir(self)
            a.put(f"benchmark_{task}", "train", "r.json", rec)
            b.put(f"benchmark_{task}", "train", "r.json", decode_row(task, enc))
            self.assertEqual(a.export()[1], b.export()[1])


class MetricConsistency(unittest.TestCase):
    """The lm-eval copy of the metrics must agree with src/evaluations/metrics.py (utils.py says it MIRRORS it)."""
    CASES = [({"a": 1, "b": [1, 2]}, {"a": 1.0, "b": [1, 2]}), ({"a": True}, {"a": 1}), ({"a": "10"}, {"a": 10}),
             ({"a": {"b": 1}}, {"a": {}}), ({}, {"x": 1}), ({"a": [1, 2, 3]}, {"a": [1, 2]})]

    def test_semantic_fidelity_and_changed_paths_agree(self):
        for ref, out in self.CASES:
            self.assertEqual(repo_metrics.semantic_fidelity(ref, out), lm_utils.semantic_fidelity(ref, out))
            self.assertEqual(repo_metrics.changed_paths(ref, out), lm_utils.changed_paths(ref, out))

    def test_schema_compliance_and_parse_agree(self):
        for obj in ({"n": 1}, {"n": "x"}, None):
            self.assertEqual(repo_metrics.schema_compliance(obj, SCHEMA)[0], lm_utils.schema_compliance(obj, SCHEMA))
        for txt in ('{"a": 1}', '```json\n{"a": 1}\n```', "not json", ""):
            self.assertEqual(repo_metrics.parse_json(txt)[0], lm_utils.parse_json(txt))

    def test_exact_match_requires_validity_compliance_and_full_fidelity(self):
        r = lm_utils.process_results_create(rec_create(1, {"type": "object"}), ['{"n": 1, "s": "é\\u2028"}'])
        self.assertEqual((r["json_validity"], r["schema_compliance"], r["exact_match"]), (1, 1, 1))
        r = lm_utils.process_results_create(rec_create(1, {"type": "object"}), ['{"n": 2, "s": "é\\u2028"}'])
        self.assertEqual((r["json_validity"], r["exact_match"]), (1, 0))
        self.assertLess(r["semantic_fidelity"], 1.0)

    def test_change_fidelity_exists_only_for_fix_and_modify(self):
        self.assertNotIn("change_fidelity", lm_utils.process_results_create(rec_create(1), ['{"n": 1}']))
        self.assertIn("change_fidelity", lm_utils.process_results_fix(rec_fix(1), ['{"n": 1}']))
        self.assertIn("change_fidelity", lm_utils.process_results_modify(rec_modify(1), ['{"n": 1}']))


class RealDataReadiness(unittest.TestCase):
    """real data: the untracked local copy of the published 7K (skipped when absent)."""

    @unittest.skipUnless(os.path.isfile(os.path.join(REL, "row_origin.tsv")), "local release/scale-only-7k not present")
    def test_7k_export_counts_and_original_rows_identical_to_3k_export(self):
        import csv
        pub = [d for d in os.listdir(os.path.join(ROOT, "data_published")) if os.path.isdir(os.path.join(ROOT, "data_published", d, "original_files"))]
        if not pub:
            self.skipTest("data_published baseline not present")
        base = os.path.join(ROOT, "data_published", pub[0], "original_files")

        def load(p):
            with open(p, encoding="utf-8") as f:
                return [json.loads(x) for x in f if x.strip()]

        def build(paths):
            w = Workdir(self)
            for task, p in paths.items():
                for i, r in enumerate(load(p), 1):
                    w.put(f"benchmark_{task}", "train", f"{i:05d}.json", decode_row(task, r))
            return w.export()[0]

        r3 = build({t: os.path.join(base, t, "train.jsonl") for t in ("create", "fix", "modify")})
        r7 = build({t: os.path.join(REL, t, "train.jsonl") for t in ("create", "fix", "modify")})
        self.assertEqual((len(r3), len(r7)), (3000, 7000))
        origin = {}
        with open(os.path.join(REL, "row_origin.tsv"), encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["split"] == "train":
                    origin[(r["task"], int(r["line"]))] = r["origin"]
        pos, orig7 = 0, []
        for t, n in (("create", 2334), ("fix", 2333), ("modify", 2333)):
            for i in range(1, n + 1):
                if origin[(t, i)] == "original":
                    orig7.append(json.dumps(r7[pos + i - 1], sort_keys=True))
            pos += n
        self.assertEqual(sorted(orig7), sorted(json.dumps(x, sort_keys=True) for x in r3))


if __name__ == "__main__":
    unittest.main()
