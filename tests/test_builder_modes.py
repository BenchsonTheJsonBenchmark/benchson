"""Integration checks: BenchmarkBuilder with the standard/complex mix, using the offline mock model.

No real model and no network are used. Plain unittest. From the repo root:
    python3 -m unittest discover -s tests -v
"""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from data_generation.benchmark_builder import BenchmarkBuilder
from llm.llm_provider import LLMProvider
from llm.mock.mock_provider import MockProvider
from llm.recording_provider import CallBudget, ProviderFailing, RecordingProvider

TASKS = ("create", "fix", "modify")


def rich_schema(i):
    return {
        "title": f"rich-{i}", "type": "object", "required": ["id", "count"],
        "properties": {
            "id": {"type": "string"}, "count": {"type": "integer", "minimum": 0, "maximum": 100},
            "enabled": {"type": "boolean"}, "mode": {"enum": ["fast", "slow"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "owner": {"type": "object", "properties": {
                "name": {"type": "string"}, "email": {"type": "string"},
                "address": {"type": "object", "required": ["city"],
                            "properties": {"city": {"type": "string"}}}}},
            "orders": {"type": "array", "items": {"type": "object", "required": ["sku", "qty"],
                       "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1},
                                      "note": {"type": "string"}}}},
        },
    }


FLAT_SCHEMA = {"title": "flat", "type": "object", "required": ["a"],
               "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}


class Env:
    """A temp folder with schema files and helpers to run the builder."""

    def __init__(self, test, schemas):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.files = []
        for i, schema in enumerate(schemas):
            path = os.path.join(self.tmp.name, f"schema_{i:02d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(schema, f)
            self.files.append(path)

    def config(self, counts, generation="default", out="out", **extra):
        cfg = {"sources": [{"name": "t", "files": self.files}], "max_retries": 2, "seed": 1,
               "concurrency": 1, "counts": counts, "validation": {"round_trip": False},
               "output_root": os.path.join(self.tmp.name, out)}
        if generation == "default":
            generation = {"standard_ratio": 0.8, "complex_ratio": 0.2,
                          "complex": {"create": {"min_leaves": 8, "min_depth": 3, "min_distinct_types": 3}}}
        if generation is not None:
            cfg["generation"] = generation
        cfg.update(extra)
        return cfg

    def run(self, cfg, llm=None):
        builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            builder.build(llm or MockProvider())
        return builder

    def records(self, cfg, task, split="test"):
        folder = os.path.join(cfg["output_root"], f"benchmark_{task}", split)
        out = {}
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.endswith(".json"):
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        out[name] = json.load(f)
        return out

    def sidecars(self, cfg, task, split="test"):
        folder = os.path.join(cfg["output_root"], f"benchmark_{task}", "_metadata", split)
        out = {}
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    out[name] = json.load(f)
        return out


def modes_of(env, cfg, task):
    counts = {"standard": 0, "complex": 0}
    for meta in env.sidecars(cfg, task).values():
        counts[meta["mode"]] += 1
    return counts


class MixedRunTests(unittest.TestCase):

    def setUp(self):
        self.env = Env(self, [rich_schema(i) for i in range(14)])
        self.cfg = self.env.config({"create": 5, "fix": 5, "modify": 5})
        self.builder = self.env.run(self.cfg)
        self.summary = self.builder.last_summary

    def test_eighty_twenty_quotas_are_met_and_reported(self):
        totals = self.summary["totals"]
        self.assertEqual((totals["requested_standard"], totals["requested_complex"]), (12, 3))
        self.assertEqual((totals["produced_standard"], totals["produced_complex"]), (12, 3))
        self.assertEqual(self.summary["shortfall"], [])
        for task in TASKS:
            self.assertEqual(modes_of(self.env, self.cfg, task), {"standard": 4, "complex": 1})
            self.assertEqual(len(self.env.records(self.cfg, task)), 5)

    def test_report_files_are_written_where_the_summary_says(self):
        outputs = self.summary["outputs"]
        self.assertTrue(os.path.isfile(os.path.join(outputs["report_dir"], "generation_summary.json")))
        self.assertTrue(os.path.isfile(outputs["failures_file"]))
        for key in ("requested", "produced_on_disk", "failed_attempts", "totals", "complexity",
                    "failure_reasons", "validation_check_failures", "builder_llm_calls"):
            self.assertIn(key, self.summary)
        self.assertTrue(self.summary["completed"])
        self.assertGreater(self.summary["builder_llm_calls"], 0)

    def test_complex_modify_has_several_verified_operations(self):
        rows = [(r, self.env.sidecars(self.cfg, "modify")[name]) for name, r in
                self.env.records(self.cfg, "modify").items()]
        complex_rows = [(r, m) for r, m in rows if m["mode"] == "complex"]
        self.assertEqual(len(complex_rows), 1)
        record, meta = complex_rows[0]
        self.assertGreaterEqual(record["modification"]["n_operations"], 2)
        self.assertEqual(len(record["modification"]["operations"]), record["modification"]["n_operations"])
        self.assertTrue(meta["validation"]["ok"])
        self.assertEqual(meta["complexity"]["operations"]["count"], record["modification"]["n_operations"])
        for op in record["modification"]["operations"]:
            self.assertIn(op["instruction"], record["modification"]["precise_instruction"])

    def test_standard_modify_stays_a_single_original_style_edit(self):
        for name, record in self.env.records(self.cfg, "modify").items():
            meta = self.env.sidecars(self.cfg, "modify")[name]
            if meta["mode"] == "standard":
                self.assertEqual(set(record["modification"]), {"op", "path", "precise_instruction"})
                self.assertLessEqual(len(record["modification"]["path"]), 2)
                self.assertEqual(meta["complexity"]["operations"]["count"], 1)

    def test_complex_fix_has_several_recorded_errors(self):
        for name, meta in self.env.sidecars(self.cfg, "fix").items():
            if meta["mode"] == "complex":
                errors = meta["complexity"]["errors"]
                self.assertGreaterEqual(errors["count"], 2)
                self.assertEqual(len(errors["errors"]), errors["count"])
                self.assertTrue(meta["validation"]["ok"])
                record = self.env.records(self.cfg, "fix")[name]
                self.assertNotEqual(record["erroneous_json"], record["valid_json"])

    def test_complex_create_reference_clears_the_complexity_thresholds(self):
        for name, meta in self.env.sidecars(self.cfg, "create").items():
            ref = meta["complexity"]["reference"]
            if meta["mode"] == "complex":
                self.assertGreaterEqual(ref["leaf_count"], 8)
                self.assertGreaterEqual(ref["depth"], 3)
                self.assertGreaterEqual(ref["distinct_leaf_types"], 3)
            self.assertTrue(meta["validation"]["grounding_passed"])
            self.assertEqual(meta["validation"]["grounding_threshold"], 0.9)   # unchanged gate

    def test_sidecars_carry_the_required_metadata(self):
        for task in TASKS:
            for name, meta in self.env.sidecars(self.cfg, task).items():
                for key in ("task", "mode", "schema_id", "seed", "model", "provider", "complexity",
                            "validation", "attempts", "failure_reason", "split",
                            "builder_llm_calls_for_schema", "requires_manual_review"):
                    self.assertIn(key, meta, (task, name, key))
                self.assertEqual(meta["task"], task)
                self.assertEqual(meta["model"], "mock")
                self.assertEqual(meta["schema_id"] + ".json", name)
                self.assertIn("depth", meta["complexity"]["reference"])
                self.assertIn("leaf_count", meta["complexity"]["reference"])

    def test_duplicate_content_is_detected_and_reported_not_removed(self):
        # the 14 fixture schemas are structurally identical, so the mock yields identical content
        count = self.summary["duplicates"]["count"]
        self.assertGreater(count, 0)
        self.assertLess(count, 15)          # the first of each content group is not a duplicate
        flagged = [m for task in TASKS for m in self.env.sidecars(self.cfg, task).values()
                   if m.get("duplicate_of")]
        self.assertTrue(flagged)
        self.assertEqual(sum(len(self.env.records(self.cfg, t)) for t in TASKS), 15)

    def test_training_record_formats_are_unchanged(self):
        base = {"name", "source", "subset", "schema"}
        expected = {"create": base | {"description", "reference_json"},
                    "fix": base | {"erroneous_json", "valid_json", "description"},
                    "modify": base | {"data", "instructions", "ground_truth", "modification"}}
        for task in TASKS:
            for record in self.env.records(self.cfg, task).values():
                self.assertEqual(set(record), expected[task], task)

    def test_sidecars_live_outside_the_split_folders(self):
        for task in TASKS:
            for split in ("test", "train"):
                folder = os.path.join(self.cfg["output_root"], f"benchmark_{task}", split)
                self.assertTrue(all(n.endswith(".json") and not n.startswith("_")
                                    for n in os.listdir(folder)))


class DuplicateDetectionTests(unittest.TestCase):

    def test_distinct_content_is_not_flagged(self):
        seen = {}
        a = BenchmarkBuilder._note_content(seen, "a", "create", {"reference_json": {"x": 1}, "description": "d1"})
        b = BenchmarkBuilder._note_content(seen, "b", "create", {"reference_json": {"x": 2}, "description": "d2"})
        c = BenchmarkBuilder._note_content(seen, "c", "create", {"reference_json": {"x": 1}, "description": "d1"})
        self.assertEqual((a, b, c), (None, None, "a"))


class LegacyTests(unittest.TestCase):

    def test_without_a_generation_block_nothing_new_is_written(self):
        env = Env(self, [rich_schema(i) for i in range(6)])
        cfg = env.config({"create": 3, "fix": 3, "modify": 3}, generation=None)
        builder = env.run(cfg)
        self.assertIsNone(builder.last_summary)
        for task in TASKS:
            self.assertEqual(len(env.records(cfg, task)), 3)
            self.assertFalse(os.path.exists(os.path.join(cfg["output_root"], f"benchmark_{task}", "_metadata")))
        self.assertFalse(os.path.exists(os.path.join(cfg["output_root"], "_generation_reports")))


class ReproducibilityTests(unittest.TestCase):

    def test_same_seed_gives_identical_data(self):
        env = Env(self, [rich_schema(i) for i in range(10)])
        a = env.config({"create": 4, "fix": 4, "modify": 4}, out="run_a")
        b = env.config({"create": 4, "fix": 4, "modify": 4}, out="run_b")
        env.run(a)
        env.run(b)
        for task in TASKS:
            self.assertEqual(env.records(a, task), env.records(b, task), task)
            self.assertEqual(env.sidecars(a, task), env.sidecars(b, task), task)

    def test_a_different_seed_changes_the_data(self):
        env = Env(self, [rich_schema(i) for i in range(10)])
        a = env.config({"modify": 4}, out="run_a")
        b = env.config({"modify": 4}, out="run_b", seed=2)
        env.run(a)
        env.run(b)
        self.assertNotEqual(env.records(a, "modify"), env.records(b, "modify"))


class SingleTaskTests(unittest.TestCase):

    def test_four_complex_examples_of_each_task_on_its_own(self):
        env = Env(self, [rich_schema(i) for i in range(12)])
        for task in TASKS:
            cfg = env.config({task: 4}, generation={"standard_ratio": 0.0, "complex_ratio": 1.0,
                             "complex": {"create": {"min_leaves": 8, "min_depth": 3, "min_distinct_types": 3}}},
                             out=f"only_{task}")
            builder = env.run(cfg)
            self.assertEqual(modes_of(env, cfg, task), {"standard": 0, "complex": 4}, task)
            for other in TASKS:
                if other != task:
                    self.assertEqual(env.records(cfg, other), {})
            self.assertEqual(builder.last_summary["totals"]["produced_standard"], 0)

    def test_four_standard_examples_use_no_complex_logic(self):
        env = Env(self, [rich_schema(i) for i in range(8)])
        cfg = env.config({"create": 4}, generation={"standard_ratio": 1.0, "complex_ratio": 0.0})
        env.run(cfg)
        self.assertEqual(modes_of(env, cfg, "create"), {"standard": 4, "complex": 0})


class FailureHandlingTests(unittest.TestCase):

    def test_failed_complex_examples_are_recorded_and_never_replaced_by_standard_ones(self):
        env = Env(self, [FLAT_SCHEMA] * 1 + [dict(FLAT_SCHEMA, title=f"flat{i}") for i in range(5)])
        cfg = env.config({"create": 2}, generation={"standard_ratio": 0.0, "complex_ratio": 1.0})
        builder = env.run(cfg)
        self.assertEqual(env.records(cfg, "create"), {})
        summary = builder.last_summary
        self.assertEqual(summary["totals"]["produced_complex"], 0)
        self.assertEqual(summary["totals"]["produced_standard"], 0)
        self.assertEqual(summary["shortfall"][0]["requested"], 2)
        self.assertIn("create/complex/schema_cannot_support_complexity", summary["failure_reasons"])
        with open(summary["outputs"]["failures_file"], encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertTrue(rows and all(r["mode"] == "complex" for r in rows))

    def test_a_complex_cell_stops_after_its_failure_budget(self):
        # one required string and nothing else: only one possible edit, so 2-3 operations never fit
        single = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}
        env = Env(self, [dict(single, title=f"single{i}") for i in range(30)])
        cfg = env.config({"modify": 1}, generation={"standard_ratio": 0.0, "complex_ratio": 1.0,
                         "complex": {"max_failures_per_example": 3}},
                         validation={"round_trip": False, "min_populated_fields": 1})
        builder = env.run(cfg)
        self.assertEqual(builder.last_summary["failed_attempts"]["test"]["modify"]["complex"], 3)
        self.assertEqual(builder.last_summary["shortfall"][0]["reason"], "failure budget exhausted")

    def test_a_model_that_raises_is_recorded_and_the_run_finishes(self):
        class Broken(MockProvider):
            def _generate(self, messages, parameters=None):
                raise RuntimeError("model unavailable")

        env = Env(self, [rich_schema(i) for i in range(3)])
        cfg = env.config({"create": 1, "fix": 1, "modify": 1})
        builder = env.run(cfg, llm=Broken())
        reasons = builder.last_summary["failure_reasons"]
        self.assertTrue(any(key.endswith("/worker_error") for key in reasons), reasons)
        self.assertEqual(builder.last_summary["totals"]["produced_on_disk"], 0)

    def test_provider_failing_stops_the_run_but_the_report_is_still_written(self):
        class Broken(MockProvider):
            def _generate(self, messages, parameters=None):
                raise RuntimeError("403 no_associated_service_instance")

        env = Env(self, [rich_schema(i) for i in range(5)])
        cfg = env.config({"create": 2})
        llm = RecordingProvider(Broken(), os.path.join(env.tmp.name, "calls.jsonl"), "builder",
                                CallBudget(None, max_consecutive_errors=2))
        builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ProviderFailing):
            builder.build(llm)
        self.assertIsNotNone(builder.last_summary)
        self.assertFalse(builder.last_summary["completed"])

    def test_call_budget_is_respected(self):
        env = Env(self, [rich_schema(i) for i in range(5)])
        cfg = env.config({"create": 3})
        llm = RecordingProvider(MockProvider(), os.path.join(env.tmp.name, "calls.jsonl"), "builder",
                                CallBudget(2))
        builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(BaseException):
            builder.build(llm)
        self.assertLessEqual(llm.budget.used, 2)


class ResumeAndSplitTests(unittest.TestCase):

    def test_rerunning_tops_up_without_duplicating_and_counts_modes_from_sidecars(self):
        env = Env(self, [rich_schema(i) for i in range(12)])
        cfg = env.config({"modify": 5})
        first = env.run(cfg)
        before = env.records(cfg, "modify")
        again = env.run(cfg)
        self.assertEqual(env.records(cfg, "modify"), before)
        self.assertEqual(sum(again.last_summary["produced_this_run"]["test"]["modify"].values()), 0)
        self.assertEqual(modes_of(env, cfg, "modify"), {"standard": 4, "complex": 1})
        self.assertEqual(first.last_summary["totals"]["produced_complex"], 1)

    def test_train_only_build_works_and_never_uses_test_schemas(self):
        env = Env(self, [rich_schema(i) for i in range(20)])
        full = env.config({"modify": 3}, out="both")
        full["counts"] = {"test": {"modify": 3}, "train": {"modify": 3}}
        builder_both = env.run(full)
        test_ids = set(env.records(full, "modify", "test"))
        train_ids = set(env.records(full, "modify", "train"))
        self.assertFalse(test_ids & train_ids)

        only_train = env.config({"modify": 3}, out="train_only")
        only_train["counts"] = {"train": {"modify": 3}}
        env.run(only_train)                                   # used to raise KeyError('test')
        produced = set(env.records(only_train, "modify", "train"))
        self.assertEqual(len(produced), 3)
        test_pool = {BenchmarkBuilder._safe_id(s, sub, o) + ".json"
                     for _, s, sub, o in BenchmarkBuilder(full)._partition(
                         BenchmarkBuilder(full)._load_schema_records())["test"]}
        self.assertFalse(produced & test_pool)
        self.assertEqual(builder_both._partition_sizes["test"] + builder_both._partition_sizes["train"], 20)


if __name__ == "__main__":
    unittest.main()
