"""Checks for the Create pilot runner (src/preview_create_pilot.py) and the builder's `files` source.

Plain unittest. From the repo root:
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data_generation.benchmark_builder import BenchmarkBuilder, _grounding_ratio
from llm.llm_provider import LLMProvider
from llm.recording_provider import CallBudget, RecordingProvider
from preview_create_pilot import PhaseProbe, checkable_strings, run_one, ungrounded_strings

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "city": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "city", "count"],
}
REFERENCE = {"name": "Alice Cohen", "city": "Tel Aviv", "count": 3}


class ScriptedProvider(LLMProvider):
    """Returns the next scripted reply on every call."""

    def __init__(self, replies):
        super().__init__()
        self.model = "scripted"
        self.replies = list(replies)

    def _generate(self, messages, parameters=None):
        return self.replies.pop(0)


def make_builder(files=None, **extra):
    cfg = {"sources": [{"name": "schemas", **({"files": files} if files else {})}],
           "validation": {"round_trip": False}, "max_retries": 2, **extra}
    return BenchmarkBuilder(cfg)


class FilesSourceTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.old_cwd)
        folder = os.path.join(self.tmp.name, "data", "schemas", "test")
        os.makedirs(folder)
        for name in ("a", "b", "c"):
            with open(os.path.join(folder, f"{name}.json"), "w") as f:
                json.dump({**SCHEMA, "title": name}, f)
        os.chdir(self.tmp.name)

    def origins(self, builder):
        return sorted(rec[3] for rec in builder._load_schema_records())

    def test_files_selects_exactly_those_schemas(self):
        builder = make_builder(files=["data/schemas/test/b.json"])
        self.assertEqual(self.origins(builder), ["b"])

    def test_without_files_the_whole_folder_is_loaded_as_before(self):
        self.assertEqual(self.origins(make_builder()), ["a", "b", "c"])


class GroundingMirrorTest(unittest.TestCase):

    def test_checkable_strings_match_the_builders_grounding_ratio(self):
        description = "Record for Alice Cohen, 3 items."
        checked = checkable_strings(REFERENCE)
        missing = ungrounded_strings(REFERENCE, description)
        self.assertEqual(checked, ["Alice Cohen", "Tel Aviv"])
        self.assertEqual(missing, ["Tel Aviv"])
        self.assertAlmostEqual(_grounding_ratio(REFERENCE, description), 1 - len(missing) / len(checked))

    def test_nothing_checkable_is_ratio_one(self):
        self.assertEqual(checkable_strings({"n": 1, "ok": True}), [])
        self.assertEqual(_grounding_ratio({"n": 1, "ok": True}, "x"), 1.0)


class RunOneTest(unittest.TestCase):

    def run_schema(self, replies, max_calls=None):
        builder = make_builder()
        budget = CallBudget(max_calls)
        path = os.path.join(tempfile.gettempdir(), "pilot_test_calls.jsonl")
        if os.path.exists(path):
            os.remove(path)
        probe = PhaseProbe(RecordingProvider(ScriptedProvider(replies), path, "builder", budget), budget)
        gates = builder.gates_by_split["test"]
        item = {"schema": SCHEMA, "name": "s", "tier": "LOW"}
        return run_one(builder, probe, gates, item, 1)

    def test_accepted_on_first_description(self):
        row, _ = self.run_schema([json.dumps(REFERENCE), "Alice Cohen lives in Tel Aviv and has 3."])
        self.assertTrue(row["create_accepted"])
        self.assertEqual((row["ref_calls"], row["desc_attempts"], row["llm_calls"]), (1, 1, 2))
        self.assertEqual(row["ref_leaves"], 3)
        self.assertEqual(row["ref_depth"], 1)
        self.assertEqual(row["desc_groundings"], [1.0])
        self.assertIsNone(row["failure_stage"])

    def test_grounding_failure_is_recorded_with_every_attempt(self):
        replies = [json.dumps(REFERENCE)] + ["Someone lives somewhere."] * 3
        row, details = self.run_schema(replies)
        self.assertFalse(row["create_accepted"])
        self.assertEqual(row["failure_stage"], "grounding_failed")
        self.assertEqual(row["desc_attempts"], 3)
        self.assertEqual(row["desc_groundings"], [0.0, 0.0, 0.0])
        self.assertEqual(details["ungrounded_strings"][0], ["Alice Cohen", "Tel Aviv"])

    def test_empty_descriptions_are_their_own_stage(self):
        row, _ = self.run_schema([json.dumps(REFERENCE)] + [""] * 3)
        self.assertEqual(row["failure_stage"], "description_empty")

    def test_reference_failure_is_classified(self):
        row, details = self.run_schema(['{"name": "Al'] * 9)  # cut off, never valid
        self.assertFalse(row["ref_success"])
        self.assertEqual(row["failure_stage"], "reference_truncated")
        self.assertEqual((row["ref_build_attempts"], row["ref_calls"]), (3, 9))
        self.assertEqual(row["desc_attempts"], 0)
        self.assertEqual(details["reference_failure_counts"]["truncated"], 9)

    def test_trivial_reference_is_the_non_triviality_gate(self):
        row, _ = self.run_schema([json.dumps({"name": "", "city": "", "count": 0})] * 3)
        self.assertEqual(row["failure_stage"], "reference_non_trivial")

    def test_budget_stop_is_interrupted_not_a_genuine_failure(self):
        row, details = self.run_schema([json.dumps(REFERENCE), "Alice Cohen Tel Aviv"], max_calls=1)
        self.assertTrue(details["interrupted"])
        self.assertEqual(row["failure_stage"], "call_budget_exhausted")
        self.assertEqual(row["llm_calls"], 1)


if __name__ == "__main__":
    unittest.main()
