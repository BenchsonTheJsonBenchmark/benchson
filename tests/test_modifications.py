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

from data_generation.modifications import _candidates, apply_change

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


if __name__ == "__main__":
    unittest.main()
