"""Checks for the fix task's standard (single error) and complex (multi-error) modes.

Plain unittest. From the repo root:
    python3 -m unittest discover -s tests -v
"""

import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data_generation.fix_errors import (apply_error_record, inject_errors, restore_error_record,
                                        verify_fix)
from data_generation.modifications import paths_conflict
from data_generation.tasks.error_correction import _inject_error, _inject_error_info
from evaluations import metrics

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 20},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"},
        "role": {"enum": ["admin", "user"]},
        "active": {"type": "boolean"},
        "tags": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
        "address": {
            "type": "object", "additionalProperties": False, "required": ["city"],
            "properties": {"city": {"type": "string"}, "zip": {"type": "string"},
                           "geo": {"type": "object", "properties": {"lat": {"type": "number", "minimum": -90, "maximum": 90}}}},
        },
        "items": {"type": "array", "items": {"type": "object", "required": ["sku"],
                  "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}}}},
    },
    "required": ["name", "age"],
}
VALID = {"name": "Ada", "age": 36, "email": "ada@example.com", "role": "user", "active": True,
         "tags": ["a", "b"], "address": {"city": "Paris", "zip": "75001", "geo": {"lat": 48.8}},
         "items": [{"sku": "X1", "qty": 2}, {"sku": "X2", "qty": 5}]}


class StandardFixTests(unittest.TestCase):

    def test_the_fixture_is_valid(self):
        self.assertEqual(metrics.schema_compliance(VALID, SCHEMA)[0], 1)

    def test_single_error_behaviour_is_unchanged(self):
        # The original injector can pick a mutation that does not actually break the schema
        # (the builder discards those); this mode keeps that behaviour, so only the rate is checked.
        broken = 0
        for seed in range(60):
            erroneous = _inject_error(VALID, SCHEMA, random.Random(seed))
            self.assertIsNotNone(erroneous)
            broken += metrics.schema_compliance(erroneous, SCHEMA)[0] == 0
            # one mutation = one top-level field touched
            self.assertEqual(len({path[0] for path in metrics.changed_paths(VALID, erroneous)}), 1)
        self.assertGreater(broken, 45)

    def test_a_seeded_rng_makes_the_choice_reproducible(self):
        a = _inject_error(VALID, SCHEMA, random.Random(9))
        b = _inject_error(VALID, SCHEMA, random.Random(9))
        self.assertEqual(a, b)

    def test_without_an_rng_the_global_random_module_is_still_used(self):
        random.seed(3)
        first = _inject_error(VALID, SCHEMA)
        random.seed(3)
        self.assertEqual(first, _inject_error(VALID, SCHEMA))

    def test_info_names_the_mutation(self):
        erroneous, info = _inject_error_info(VALID, SCHEMA, random.Random(1))
        self.assertIsNotNone(erroneous)
        self.assertIn("kind", info)

    def test_nothing_to_mutate_returns_none(self):
        self.assertEqual(_inject_error_info({"a": 1}, {"type": "object"}), (None, None))
        self.assertIsNone(_inject_error([], SCHEMA))


class ComplexFixTests(unittest.TestCase):

    def _results(self, seeds=range(60)):
        for seed in seeds:
            result = inject_errors(VALID, SCHEMA, random.Random(seed))
            self.assertIsNotNone(result, seed)
            yield seed, result

    def test_two_or_three_errors(self):
        self.assertEqual({len(errors) for _, (_, errors) in self._results()}, {2, 3})

    def test_every_result_passes_all_programmatic_checks(self):
        for seed, (erroneous, errors) in self._results():
            check = verify_fix(VALID, erroneous, errors, SCHEMA)
            self.assertTrue(check["ok"], (seed, check))
            self.assertTrue(all(v is True for v in check["checks"].values()), (seed, check))

    def test_original_is_valid_and_corrupted_is_invalid(self):
        for _, (erroneous, _) in self._results():
            self.assertEqual(metrics.schema_compliance(VALID, SCHEMA)[0], 1)
            self.assertEqual(metrics.schema_compliance(erroneous, SCHEMA)[0], 0)

    def test_putting_the_originals_back_recovers_the_valid_object(self):
        for _, (erroneous, errors) in self._results():
            restored = copy.deepcopy(erroneous)
            for record in errors:
                restore_error_record(restored, record)
            self.assertEqual(restored, VALID)

    def test_each_error_breaks_the_schema_on_its_own(self):
        for _, (_, errors) in self._results():
            for record in errors:
                single = copy.deepcopy(VALID)
                apply_error_record(single, record)
                self.assertEqual(metrics.schema_compliance(single, SCHEMA)[0], 0, record)

    def test_errors_never_overlap(self):
        for _, (_, errors) in self._results():
            for i, a in enumerate(errors):
                for b in errors[i + 1:]:
                    self.assertFalse(paths_conflict(a["path"], b["path"]), (a, b))

    def test_errors_vary_in_kind_depth_and_branch(self):
        kinds, depths, branched, in_arrays = set(), set(), 0, 0
        for _, (_, errors) in self._results(range(120)):
            kinds |= {e["kind"] for e in errors}
            depths |= {len(e["path"]) for e in errors}
            branched += len({e["path"][0] for e in errors}) > 1
            in_arrays += any(isinstance(s, int) for e in errors for s in e["path"])
        self.assertGreaterEqual(len(kinds), 8)
        self.assertTrue({1, 2}.issubset(depths))
        self.assertGreater(branched, 100)
        self.assertGreater(in_arrays, 0)

    def test_distinct_kinds_within_an_example_whenever_possible(self):
        for _, (_, errors) in self._results():
            self.assertEqual(len({e["kind"] for e in errors}), len(errors))

    def test_deterministic_and_leaves_global_random_alone(self):
        random.seed(8)
        before = random.random()
        random.seed(8)
        self.assertEqual(inject_errors(VALID, SCHEMA, random.Random(21)),
                         inject_errors(VALID, SCHEMA, random.Random(21)))
        self.assertEqual(random.random(), before)

    def test_input_is_not_mutated(self):
        snapshot = copy.deepcopy(VALID)
        inject_errors(VALID, SCHEMA, random.Random(2))
        self.assertEqual(VALID, snapshot)

    def test_count_is_configurable_and_too_few_candidates_give_none(self):
        _, errors = inject_errors(VALID, SCHEMA, random.Random(0), n_errors=(4, 4))
        self.assertEqual(len(errors), 4)
        tiny = {"type": "object", "properties": {"a": {"type": "boolean"}}}
        self.assertIsNone(inject_errors({"a": True}, tiny, random.Random(0), n_errors=(5, 5)))

    def test_non_object_input_gives_none(self):
        self.assertIsNone(inject_errors([1, 2], SCHEMA, random.Random(0)))

    def test_local_refs_are_followed(self):
        schema = {"type": "object", "required": ["box"], "properties": {"box": {"$ref": "#/$defs/box"}},
                  "$defs": {"box": {"type": "object", "required": ["w"], "additionalProperties": False,
                                    "properties": {"w": {"type": "integer", "minimum": 1},
                                                   "h": {"type": "integer", "minimum": 1}}}}}
        valid = {"box": {"w": 3, "h": 4}}
        _, errors = inject_errors(valid, schema, random.Random(0), n_errors=(2, 2))
        self.assertTrue(all(e["path"][0] == "box" and len(e["path"]) >= 1 for e in errors))
        self.assertTrue(any(len(e["path"]) == 2 for e in errors))


class VerificationTests(unittest.TestCase):

    def setUp(self):
        self.erroneous, self.errors = inject_errors(VALID, SCHEMA, random.Random(5))

    def test_a_corrupted_object_that_is_actually_valid_is_detected(self):
        check = verify_fix(VALID, copy.deepcopy(VALID), self.errors, SCHEMA)
        self.assertFalse(check["ok"])
        self.assertIn("corrupted_invalid", check["problems"])

    def test_an_unrecorded_extra_change_is_detected(self):
        tampered = copy.deepcopy(self.erroneous)
        tampered["role"] = "admin" if tampered.get("role") != "admin" else "user"
        if tampered == self.erroneous:
            self.skipTest("seed touched role")
        check = verify_fix(VALID, tampered, self.errors, SCHEMA)
        self.assertFalse(check["ok"])

    def test_an_invalid_target_is_detected(self):
        bad_valid = copy.deepcopy(VALID)
        bad_valid["age"] = -1
        self.assertIn("original_valid", verify_fix(bad_valid, self.erroneous, self.errors, SCHEMA)["problems"])

    def test_overlapping_errors_are_detected(self):
        errors = [dict(self.errors[0]), dict(self.errors[0])]
        self.assertIn("paths_disjoint", verify_fix(VALID, self.erroneous, errors, SCHEMA)["problems"])

    def test_manual_review_is_flagged(self):
        self.assertTrue(verify_fix(VALID, self.erroneous, self.errors, SCHEMA)["manual_review"])


if __name__ == "__main__":
    unittest.main()
