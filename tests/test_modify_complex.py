"""Checks for the modify task's standard (original) and complex (multi-edit) modes.

Plain unittest. From the repo root:
    python3 -m unittest discover -s tests -v
"""

import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data_generation.modifications import (_candidates, apply_change, apply_changes,
                                           instruction_covers_operations, paths_conflict,
                                           replay_operation, verify_modification)
from evaluations import metrics

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "active": {"type": "boolean"},
        "role": {"enum": ["admin", "user"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {"type": "object", "properties": {
            "city": {"type": "string"}, "zip": {"type": "string"},
            "geo": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}}}},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}}, "required": ["sku"]}},
    },
    "required": ["name"],
}
OBJ = {"name": "Ada", "age": 36, "active": True, "role": "user", "tags": ["a", "b"],
       "address": {"city": "Paris", "zip": "75001", "geo": {"lat": 48.8, "lon": 2.3}},
       "items": [{"sku": "X1", "qty": 2}, {"sku": "X2", "qty": 5}]}

# (op, path) picked by the ORIGINAL pre-complexity modify code for seeds 0..7 on SCHEMA/OBJ,
# recorded from commit 2a3ae68. "standard" mode must keep reproducing them.
ORIGINAL_GOLDEN = [
    ("delete_field", ["age"]), ("delete_field", ["address", "city"]), ("delete_field", ["age"]),
    ("delete_field", ["address", "city"]), ("array_remove_first", ["items"]), ("set_bool", ["active"]),
    ("delete_field", ["address"]), ("array_append_copy", ["items"]),
]


class StandardModeTests(unittest.TestCase):

    def test_original_strategy_reproduces_the_pre_complexity_choices(self):
        picks = []
        for seed in range(8):
            _, spec = apply_change(OBJ, SCHEMA, random.Random(seed), strategy="original")
            picks.append((spec["op"], spec["path"]))
        self.assertEqual(picks, ORIGINAL_GOLDEN)

    def test_original_strategy_never_goes_deeper_than_one_nested_level(self):
        for seed in range(150):
            _, spec = apply_change(OBJ, SCHEMA, random.Random(seed), strategy="original")
            self.assertLessEqual(len(spec["path"]), 2, spec)

    def test_original_strategy_ignores_objects_inside_arrays(self):
        paths = [c["path"] for c in _candidates(OBJ, SCHEMA, max_depth=1)]
        self.assertFalse(any(isinstance(step, int) for p in paths for step in p))

    def test_standard_spec_keeps_the_single_operation_shape(self):
        _, spec = apply_change(OBJ, SCHEMA, random.Random(1), strategy="original")
        self.assertEqual(set(spec), {"op", "path", "precise_instruction"})

    def test_default_strategy_is_still_the_recursive_weighted_one(self):
        deep = {t for seed in range(200)
                for t in [len(apply_change(OBJ, SCHEMA, random.Random(seed))[1]["path"])]}
        self.assertIn(3, deep)   # address.geo.lat is only reachable by the recursive strategy

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_change(OBJ, SCHEMA, random.Random(0), strategy="nope")


class ArrayTraversalTests(unittest.TestCase):

    def test_objects_inside_arrays_are_discovered_with_the_list_index(self):
        paths = [c["path"] for c in _candidates(OBJ, SCHEMA, arrays=True, root=SCHEMA)]
        self.assertIn(["items", 0, "qty"], paths)
        self.assertIn(["items", 1, "sku"], paths)

    def test_default_discovery_does_not_enter_arrays(self):
        paths = [c["path"] for c in _candidates(OBJ, SCHEMA)]
        self.assertFalse(any(isinstance(step, int) for p in paths for step in p))

    def test_only_the_first_few_elements_are_traversed(self):
        obj = {"name": "n", "items": [{"sku": str(i)} for i in range(10)]}
        indexes = {c["path"][1] for c in _candidates(obj, SCHEMA, arrays=True) if len(c["path"]) > 2}
        self.assertEqual(indexes, {0, 1, 2})

    def test_local_refs_are_followed_when_a_root_is_given(self):
        schema = {"type": "object", "properties": {"box": {"$ref": "#/$defs/box"}},
                  "$defs": {"box": {"type": "object", "properties": {"w": {"type": "integer", "minimum": 1}},
                                    "required": ["w"]}}}
        obj = {"box": {"w": 3}}
        with_root = [(c["op"], c["path"]) for c in _candidates(obj, schema, root=schema)]
        self.assertIn(("increment_number", ["box", "w"]), with_root)
        self.assertNotIn(("delete_field", ["box", "w"]), with_root)   # required: must not be removed


class ComplexModeTests(unittest.TestCase):

    def _results(self, seeds=range(60)):
        for seed in seeds:
            result = apply_changes(OBJ, SCHEMA, random.Random(seed))
            self.assertIsNotNone(result, seed)
            yield seed, result

    def test_two_or_three_operations_each_time(self):
        counts = {spec["n_operations"] for _, (_, spec) in self._results()}
        self.assertEqual(counts, {2, 3})

    def test_every_result_is_valid_changed_and_verified(self):
        for seed, (modified, spec) in self._results():
            self.assertEqual(metrics.schema_compliance(modified, SCHEMA)[0], 1, seed)
            self.assertNotEqual(modified, OBJ)
            check = verify_modification(OBJ, modified, spec, SCHEMA)
            self.assertTrue(check["ok"], (seed, check))
            self.assertTrue(all(v is True for v in check["checks"].values()), check)

    def test_operations_never_conflict(self):
        for seed, (_, spec) in self._results():
            ops = spec["operations"]
            for i, a in enumerate(ops):
                for b in ops[i + 1:]:
                    self.assertFalse(paths_conflict(a["path"], b["path"]), (seed, a, b))

    def test_operations_vary_in_kind_depth_and_branch(self):
        kinds, depths, branched, in_arrays = set(), set(), 0, 0
        for _, (_, spec) in self._results(range(100)):
            ops = spec["operations"]
            kinds |= {op["op"] for op in ops}
            depths |= {len(op["path"]) for op in ops}
            branched += len({op["path"][0] for op in ops}) > 1
            in_arrays += any(isinstance(s, int) for op in ops for s in op["path"])
        self.assertGreaterEqual(len(kinds), 6)
        self.assertTrue({1, 2}.issubset(depths))
        self.assertGreater(branched, 80)       # nearly always more than one branch
        self.assertGreater(in_arrays, 0)       # objects inside lists do get edited

    def test_instruction_lists_every_operation(self):
        for _, (_, spec) in self._results():
            text = spec["precise_instruction"]
            for i, op in enumerate(spec["operations"], start=1):
                self.assertIn(f"({i}) {op['instruction']}", text)
            ok, missing = instruction_covers_operations(text, spec["operations"])
            self.assertTrue(ok, missing)

    def test_deterministic_for_a_seed_and_global_random_untouched(self):
        random.seed(5)
        before = random.random()
        random.seed(5)
        first = apply_changes(OBJ, SCHEMA, random.Random(11))
        second = apply_changes(OBJ, SCHEMA, random.Random(11))
        self.assertEqual(first, second)
        self.assertEqual(random.random(), before)

    def test_n_ops_is_configurable(self):
        for seed in range(30):
            _, spec = apply_changes(OBJ, SCHEMA, random.Random(seed), n_ops=(4, 4))
            self.assertEqual(spec["n_operations"], 4)

    def test_returns_none_when_not_enough_compatible_edits_exist(self):
        schema = {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]}
        self.assertIsNone(apply_changes({"a": True}, schema, random.Random(0), n_ops=(2, 3)))

    def test_input_is_not_mutated(self):
        snapshot = copy.deepcopy(OBJ)
        apply_changes(OBJ, SCHEMA, random.Random(3))
        self.assertEqual(OBJ, snapshot)

    def test_schema_constraints_hold_after_array_clear_and_removal(self):
        schema = {"type": "object", "properties": {
            "a": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "b": {"type": "string"}, "c": {"type": "integer"}}, "required": ["a"]}
        for seed in range(80):
            result = apply_changes({"a": ["x", "y"], "b": "q", "c": 1}, schema, random.Random(seed))
            if result:
                self.assertEqual(metrics.schema_compliance(result[0], schema)[0], 1)


class VerificationTests(unittest.TestCase):

    def setUp(self):
        self.modified, self.spec = apply_changes(OBJ, SCHEMA, random.Random(4))

    def test_replaying_the_spec_reproduces_the_ground_truth(self):
        replayed = copy.deepcopy(OBJ)
        for op in self.spec["operations"]:
            replay_operation(replayed, op)
        self.assertEqual(replayed, self.modified)

    def test_an_unintended_extra_change_is_detected(self):
        tampered = copy.deepcopy(self.modified)
        tampered["name"] = "Someone Else" if tampered.get("name") != "Someone Else" else "X"
        if tampered == self.modified:
            self.skipTest("seed happened to touch name")
        check = verify_modification(OBJ, tampered, self.spec, SCHEMA)
        self.assertFalse(check["ok"])
        self.assertIn("replay_matches", check["problems"])

    def test_a_missing_change_is_detected(self):
        check = verify_modification(OBJ, copy.deepcopy(OBJ), self.spec, SCHEMA)
        self.assertFalse(check["ok"])
        self.assertIn("modified_differs", check["problems"])

    def test_a_wrong_operation_count_is_detected(self):
        spec = {**self.spec, "n_operations": self.spec["n_operations"] + 1}
        self.assertIn("operation_count", verify_modification(OBJ, self.modified, spec, SCHEMA)["problems"])

    def test_a_schema_violating_ground_truth_is_detected(self):
        bad = copy.deepcopy(self.modified)
        bad["age"] = -5
        self.assertIn("schema_valid", verify_modification(OBJ, bad, self.spec, SCHEMA)["problems"])

    def test_legacy_single_spec_is_checked_where_possible_and_flags_manual_review(self):
        modified, spec = apply_change(OBJ, SCHEMA, random.Random(2), strategy="original")
        check = verify_modification(OBJ, modified, spec, SCHEMA)
        self.assertTrue(check["ok"])
        self.assertIsNone(check["checks"]["replay_matches"])
        self.assertTrue(check["manual_review"])

    def test_a_rewrite_that_drops_a_change_is_caught(self):
        ops = self.spec["operations"]
        ok, missing = instruction_covers_operations("Please tidy the record.", ops)
        self.assertFalse(ok)
        self.assertTrue(missing)


if __name__ == "__main__":
    unittest.main()
