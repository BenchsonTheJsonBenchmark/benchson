"""Checks for the modify-task candidate discovery (src/data_generation/modifications.py).

Plain unittest, so there is no extra dependency. From the repo root:
    python -m unittest discover -s tests -v
"""

import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import jsonschema

from data_generation.modifications import _candidate_complexity, _candidates, apply_change

DEEP_PATH = ["settings", "server", "network", "port"]

DEEP_SCHEMA = {
    "type": "object",
    "properties": {
        "settings": {
            "type": "object",
            "properties": {
                "server": {
                    "type": "object",
                    "properties": {
                        "network": {
                            "type": "object",
                            "properties": {
                                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                                "secure": {"type": "boolean"},
                            },
                            "required": ["port"],
                        }
                    },
                }
            },
        }
    },
}

DEEP_OBJECT = {"settings": {"server": {"network": {"port": 3000, "secure": False}}}}


def _find(candidates, path, op=None):
    return next((c for c in candidates if c["path"] == path and (op is None or c["op"] == op)), None)


class RecursiveCandidateTests(unittest.TestCase):

    def test_field_at_depth_four_is_discovered(self):
        candidates = list(_candidates(DEEP_OBJECT, DEEP_SCHEMA))
        self.assertIsNotNone(_find(candidates, DEEP_PATH), "deep path was not discovered")

    def test_deep_edit_applies_and_stays_schema_valid(self):
        original = copy.deepcopy(DEEP_OBJECT)
        candidate = _find(list(_candidates(DEEP_OBJECT, DEEP_SCHEMA)), DEEP_PATH, "increment_number")
        self.assertIsNotNone(candidate)

        modified = copy.deepcopy(DEEP_OBJECT)
        candidate["apply"](modified)

        self.assertEqual(modified["settings"]["server"]["network"]["port"], 3001)
        jsonschema.validate(instance=modified, schema=DEEP_SCHEMA)
        self.assertEqual(DEEP_OBJECT, original, "discovery/apply must not touch the input object")

    def test_instruction_names_the_full_parent_path(self):
        candidate = _find(list(_candidates(DEEP_OBJECT, DEEP_SCHEMA)), DEEP_PATH, "increment_number")
        self.assertEqual(candidate["instruction"],
                         'Increase "port" inside "settings.server.network" by 1.')

    def test_required_field_is_not_offered_for_deletion_at_depth(self):
        # Proves the matching sub-schema (with its `required` list) is passed down.
        candidates = list(_candidates(DEEP_OBJECT, DEEP_SCHEMA))
        self.assertIsNone(_find(candidates, DEEP_PATH, "delete_field"))
        self.assertIsNotNone(_find(candidates, DEEP_PATH[:-1] + ["secure"], "delete_field"))

    def test_apply_change_reaches_deep_fields_and_every_result_is_valid(self):
        deep_paths = 0
        for seed in range(40):
            result = apply_change(DEEP_OBJECT, DEEP_SCHEMA, random.Random(seed))
            self.assertIsNotNone(result, f"seed {seed} found no valid edit")
            modified, spec = result
            jsonschema.validate(instance=modified, schema=DEEP_SCHEMA)
            self.assertNotEqual(modified, DEEP_OBJECT)
            if len(spec["path"]) >= 4:
                deep_paths += 1
        self.assertGreater(deep_paths, 0, "no edit ever landed at depth >= 4")

    def test_same_seed_gives_same_edit(self):
        first = apply_change(DEEP_OBJECT, DEEP_SCHEMA, random.Random(7))
        second = apply_change(DEEP_OBJECT, DEEP_SCHEMA, random.Random(7))
        self.assertEqual(first, second)

    def test_one_level_nesting_still_works_with_old_wording(self):
        schema = {"type": "object", "properties": {
            "name": {"type": "string"},
            "meta": {"type": "object", "properties": {"flag": {"type": "boolean"}}}}}
        obj = {"name": "x", "meta": {"flag": True}}
        candidates = list(_candidates(obj, schema))
        self.assertIsNotNone(_find(candidates, ["name"]))
        flag = _find(candidates, ["meta", "flag"], "set_bool")
        self.assertEqual(flag["instruction"], 'Set "flag" inside "meta" to false.')


# ---------------------------------------------------------------- complexity-aware selection

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "port": {"type": "integer"},
        "settings": DEEP_SCHEMA["properties"]["settings"],
    },
}
SCORE_OBJECT = {"port": 80, "settings": {"server": {"network": {"port": 3000}}}}

ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}
ARRAY_OBJECT = {"count": 1, "name": "x", "tags": ["a", "b"]}

# Candidate scores for MIX_OBJECT (operation base + depth - 1):
#   count: increment 1, decrement 1, delete 2     meta: delete 2
#   meta.inner: delete 3      meta.inner.deep: delete 4
#   meta.inner.deep.level: increment 4, decrement 4, delete 5     (total weight 26)
MIX_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer"},
        "meta": {"type": "object", "properties": {
            "inner": {"type": "object", "properties": {
                "deep": {"type": "object", "properties": {
                    "level": {"type": "integer"}}}}}}},
    },
}
MIX_OBJECT = {"count": 5, "meta": {"inner": {"deep": {"level": 5}}}}

RICH_SCHEMA = {
    "type": "object",
    "required": ["name", "tags"],
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "tags": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
        "mode": {"enum": ["a", "b", "c"]},
        "limits": {"type": "object", "required": ["max"], "properties": {
            "max": {"type": "integer", "minimum": 1, "maximum": 9},
            "strict": {"type": "boolean"}}},
    },
}
RICH_OBJECT = {"name": "demo", "tags": ["x", "y"], "count": 3, "mode": "a",
               "limits": {"max": 5, "strict": True}}


def _selected_edits(obj, schema, seeds):
    """The (op, path) that apply_change picks for each seed."""
    edits = []
    for seed in seeds:
        result = apply_change(obj, schema, random.Random(seed))
        assert result is not None, f"seed {seed} found no valid edit"
        edits.append((result[1]["op"], result[1]["path"]))
    return edits


class ComplexityScoreTests(unittest.TestCase):

    def test_deeper_candidate_scores_higher_than_top_level_equivalent(self):
        candidates = list(_candidates(SCORE_OBJECT, SCORE_SCHEMA))
        top = _candidate_complexity(_find(candidates, ["port"], "increment_number"))
        deep = _candidate_complexity(_find(candidates, DEEP_PATH, "increment_number"))
        self.assertGreater(deep, top)

    def test_score_rises_with_every_extra_level(self):
        scores = [_candidate_complexity({"op": "set_bool", "path": ["k"] * depth})
                  for depth in range(1, 6)]
        self.assertEqual(scores, sorted(set(scores)), f"not strictly increasing: {scores}")

    def test_array_and_field_edits_score_above_simple_top_level_scalar_edits(self):
        candidates = list(_candidates(ARRAY_OBJECT, ARRAY_SCHEMA))
        scalar = _candidate_complexity(_find(candidates, ["count"], "increment_number"))
        string = _candidate_complexity(_find(candidates, ["name"], "set_string"))
        field = _candidate_complexity(_find(candidates, ["name"], "delete_field"))
        array = _candidate_complexity(_find(candidates, ["tags"], "array_remove_first"))
        self.assertGreater(array, scalar)
        self.assertGreater(array, string)
        self.assertGreater(field, scalar)       # add/delete field: slightly higher
        self.assertGreater(array, field)        # array edits: higher still


class WeightedSelectionTests(unittest.TestCase):

    def test_same_seed_selects_the_same_edit(self):
        for seed in range(25):
            first = apply_change(MIX_OBJECT, MIX_SCHEMA, random.Random(seed))
            second = apply_change(MIX_OBJECT, MIX_SCHEMA, random.Random(seed))
            self.assertEqual(first, second)

    def test_global_random_state_is_never_touched(self):
        random.seed(123)
        state = random.getstate()
        apply_change(MIX_OBJECT, MIX_SCHEMA, random.Random(1))
        self.assertEqual(random.getstate(), state)

    def test_complex_edits_are_favoured_but_simple_ones_still_happen(self):
        edits = _selected_edits(MIX_OBJECT, MIX_SCHEMA, range(1000))
        n = len(edits)
        deep_share = sum(1 for _, p in edits if len(p) >= 4) / n
        # the simple top-level scalar edits: increment / decrement of "count"
        simple_share = sum(1 for op, p in edits if p == ["count"] and op != "delete_field") / n
        # the single highest-scoring edit: deleting the deepest field
        top_score_share = sum(1 for op, p in edits if op == "delete_field" and len(p) == 4) / n

        # Weights are the scores (total 26). Deep edits (path length 4) are 3 of the 9
        # candidates (33% if uniform) but carry 13/26 = 50% of the weight; the two simple
        # scalar edits on "count" carry 2/26 = 7.7% (22% if uniform).
        self.assertGreater(deep_share, 0.42)       # clearly above the uniform 33%
        self.assertLess(deep_share, 0.60)          # but nowhere near "always deep"
        self.assertGreater(simple_share, 0.02)     # simple edits are still chosen
        self.assertLess(simple_share, 0.15)        # ... far less often than uniform (22%)
        self.assertGreater(len({(op, tuple(p)) for op, p in edits}), 5)   # real variety
        self.assertLess(top_score_share, 0.35)     # the top-scoring edit is not always picked

    def test_every_selected_edit_passes_schema_validation(self):
        for seed in range(300):
            result = apply_change(RICH_OBJECT, RICH_SCHEMA, random.Random(seed))
            self.assertIsNotNone(result, f"seed {seed} found no valid edit")
            modified, _ = result
            jsonschema.validate(instance=modified, schema=RICH_SCHEMA)
            self.assertNotEqual(modified, RICH_OBJECT)

    def test_an_invalid_pick_does_not_block_the_remaining_candidates(self):
        # tags has minItems 1 and one item: clear / remove-first / remove-last / remove-at
        # all break the schema, and only appending a copy is valid.
        schema = {"type": "object", "required": ["tags"],
                  "properties": {"tags": {"type": "array", "minItems": 1,
                                          "items": {"type": "string"}}}}
        for seed in range(50):
            modified, spec = apply_change({"tags": ["a"]}, schema, random.Random(seed))
            self.assertEqual(spec["op"], "array_append_copy")
            self.assertEqual(modified, {"tags": ["a", "a"]})


if __name__ == "__main__":
    unittest.main()
