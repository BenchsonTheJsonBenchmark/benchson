"""Checks for complexity metrics and for the guarantees that standard mode / quality gates are untouched.

Plain unittest. From the repo root:
    python3 -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from data_generation.benchmark_builder import BenchmarkBuilder, _grounding_ratio
from data_generation.complexity import (meets_create_thresholds, operations_metrics,
                                        reference_metrics, schema_can_support, single_fix_metrics)
from data_generation.tasks.base_task import BaseDataTask

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0},
    "role": {"enum": ["a", "b"]}, "plain": {"type": "string"},
    "orders": {"type": "array", "items": {"type": "object", "properties": {
        "sku": {"type": "string", "pattern": "^X"}, "qty": {"type": "integer"}}}}}}
REFERENCE = {"name": "Ada Lovelace", "age": 36, "role": "a", "plain": "ok",
             "orders": [{"sku": "X1", "qty": 2}, {"sku": "X2", "qty": 3}]}


class ReferenceMetricsTests(unittest.TestCase):

    def test_counts(self):
        m = reference_metrics(REFERENCE, SCHEMA)
        self.assertEqual(m["leaf_count"], 8)
        self.assertEqual(m["depth"], 3)                  # orders[0].sku
        self.assertEqual(m["distinct_leaf_types"], 2)    # string, integer
        self.assertEqual(m["object_arrays"], 1)
        self.assertEqual(m["object_array_items"], 2)
        self.assertEqual(m["checkable_strings"], 1)      # only "Ada Lovelace": "ok", "a", "X1", "X2" are < 4 chars
        self.assertGreaterEqual(m["constrained_leaves"], 4)

    def test_instance_size_and_task_complexity_are_separate_numbers(self):
        flat = {f"k{i}": f"value number {i}" for i in range(20)}
        self.assertEqual(reference_metrics(flat)["depth"], 1)
        self.assertEqual(reference_metrics(flat)["leaf_count"], 20)
        self.assertEqual(reference_metrics(flat)["object_arrays"], 0)

    def test_thresholds(self):
        ok, why = meets_create_thresholds(REFERENCE, SCHEMA, {"min_leaves": 5, "min_depth": 3})
        self.assertTrue(ok and not why)
        ok, why = meets_create_thresholds(REFERENCE, SCHEMA, {"min_leaves": 50, "min_depth": 5,
                                                              "min_distinct_types": 4,
                                                              "min_object_arrays": 2})
        self.assertFalse(ok)
        self.assertEqual(len(why), 4)

    def test_schema_capacity_precheck(self):
        thresholds = {"min_leaves": 5, "min_depth": 3}
        self.assertTrue(schema_can_support(SCHEMA, thresholds)[0])
        flat = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertFalse(schema_can_support(flat, thresholds)[0])
        open_ended = {"type": "object", "additionalProperties": {"type": "object"}}
        self.assertTrue(schema_can_support(open_ended, thresholds)[0])   # cannot rule it out
        recursive = {"type": "object", "properties": {"child": {"$ref": "#"}}}
        schema_can_support(recursive, thresholds)                        # terminates

    def test_operation_metrics(self):
        m = operations_metrics([["a", "b"], ["c"], ["items", 0, "x"]], ["set_string", "delete_field", "set_bool"])
        self.assertEqual((m["count"], m["max_depth"], m["distinct_branches"]), (3, 3, 3))
        self.assertTrue(m["different_branches"] and m["inside_array_items"])
        single = single_fix_metrics({"a": 1}, {"a": "x"}, {"kind": "wrong_type_number", "field": "a"})
        self.assertEqual((single["count"], single["types"]), (1, ["wrong_type_number"]))


class StandardModeIsUntouchedTests(unittest.TestCase):

    def test_default_quality_gates_are_unchanged(self):
        gates = BenchmarkBuilder._parse_gates({})
        self.assertEqual(gates["grounding_threshold"], 0.9)
        self.assertEqual(gates["min_populated_fields"], 2)
        self.assertEqual(gates["max_build_attempts"], 3)
        self.assertEqual(gates["round_trip_threshold"], 0.9)

    def test_standard_creation_prompt_has_no_complexity_hint(self):
        task = BaseDataTask()
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        plain = task._build_schema_creation_prompt(schema)
        self.assertEqual(plain[1]["content"],
                         "Generate a JSON object that conforms to the following schema: "
                         + json.dumps(schema, indent=2))
        hinted = task._build_schema_creation_prompt(schema, complexity_hint=True)
        self.assertTrue(hinted[1]["content"].startswith(plain[1]["content"]))
        self.assertGreater(len(hinted[1]["content"]), len(plain[1]["content"]))
        self.assertEqual(plain[0], hinted[0])

    def test_standard_naturalize_prompt_is_word_for_word_the_original(self):
        sent = []

        class Spy:
            def generate(self, messages, parameters=None):
                sent.append(messages)
                return "ok", None

        BenchmarkBuilder({"sources": []})._naturalize("Remove the \"x\" field.", Spy())
        self.assertEqual(sent[0][0]["content"],
                         "Rewrite the following data-edit instruction as a natural, conversational "
                         "request a user might type. Keep it unambiguous and preserve the exact field "
                         "names, values, and numbers. Output only the rewritten instruction.\n\n"
                         "Instruction: Remove the \"x\" field.")

    def test_grounding_metric_is_the_unchanged_literal_match(self):
        ref = {"name": "Ada Lovelace", "city": "Paris"}
        self.assertEqual(_grounding_ratio(ref, "Ada Lovelace lives in Paris."), 1.0)
        self.assertEqual(_grounding_ratio(ref, "Ada Lovelace lives somewhere."), 0.5)

    def test_create_pilot_files_are_not_part_of_this_change(self):
        # The pilot reuses _mint_reference/_make_description positionally; the new arguments are
        # keyword-only additions with defaults, so its call sites keep working unchanged.
        import inspect
        mint = inspect.signature(BenchmarkBuilder._mint_reference).parameters
        describe = inspect.signature(BenchmarkBuilder._make_description).parameters
        self.assertEqual(list(mint)[:4], ["self", "schema", "llm_provider", "gates"])
        self.assertEqual(list(describe)[:5], ["self", "schema", "reference_json", "llm_provider", "gates"])
        self.assertIs(mint["complexity_hint"].default, False)
        self.assertIsNone(mint["min_complexity"].default)
        self.assertIsNone(mint["diag"].default)
        self.assertIsNone(describe["diag"].default)


if __name__ == "__main__":
    unittest.main()
