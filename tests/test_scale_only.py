import json, os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

from build_benchmark import main as build_main, validate_scale_only
from data_generation.benchmark_builder import BenchmarkBuilder, _looks_truncated, _CountingLLM
from data_generation.published import schema_hash, safe_id


def _config(tmp, reserved_file, **over):
    cfg = {"scale_only": True, "llm_provider": {"module": "src.llm.mock.mock_provider", "class": "MockProvider", "params": {}},
           "sources": [{"name": "schemas", "max_schema_kb": 10}], "exclusions": {"file": reserved_file},
           "output_root": os.path.join(tmp, "out"), "seed": 0, "split": {"test_ratio": 0.15}, "reuse": {"max_new_references_per_schema": 2},
           "counts": {"train": {"create": 3, "fix": 3, "modify": 3}}, "validation": {"train": {"round_trip": False}},
           "generation": {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"}}
    cfg.update(over)
    return cfg


class ScaleOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.reserved = os.path.join(self.tmp.name, "base", "reserved_schemas.json")
        os.makedirs(os.path.dirname(self.reserved))

    def _write_reserved(self, ids=(), hashes=()):
        json.dump({"dataset": "x", "revision": "a" * 40, "ids": {"train": list(ids), "test": []},
                   "schema_hashes": {"train": list(hashes), "test": []}, "reference_hashes_by_schema": {},
                   "task_keys": {t: {"exact": [], "content": []} for t in ("create", "fix", "modify")}},
                  open(self.reserved, "w"))

    def test_safety_validation(self):
        self._write_reserved()
        self.assertEqual(validate_scale_only(_config(self.tmp.name, self.reserved)), [])
        bad = _config(self.tmp.name, self.reserved, generation={"standard_ratio": 0.8, "complex_ratio": 0.2})
        self.assertTrue(any("complex_ratio" in p for p in validate_scale_only(bad)))
        no_gen = _config(self.tmp.name, self.reserved); del no_gen["generation"]
        self.assertTrue(any("generation" in p for p in validate_scale_only(no_gen)))
        inside = _config(self.tmp.name, self.reserved, output_root=os.path.join(os.path.dirname(self.reserved), "new"))
        self.assertTrue(any("baseline" in p for p in validate_scale_only(inside)))
        self.assertTrue(any("test_ratio" in p for p in validate_scale_only(
            _config(self.tmp.name, self.reserved, split={"test_ratio": 0}))))
        no_reuse = _config(self.tmp.name, self.reserved); del no_reuse["reuse"]
        self.assertTrue(any("reuse" in p for p in validate_scale_only(no_reuse)))

    def test_exclusions_remove_by_id_and_hash(self):
        self._write_reserved()
        full = BenchmarkBuilder(_config(self.tmp.name, self.reserved))._load_schema_records()
        by_id, by_hash = full[0], full[1]
        self._write_reserved(ids=[safe_id(by_id[1], by_id[2], by_id[3])], hashes=[schema_hash(by_hash[0])])
        b = BenchmarkBuilder(_config(self.tmp.name, self.reserved))
        kept = b._load_schema_records()
        self.assertEqual(len(kept), len(full) - 2)
        self.assertEqual(b._pool_stats["excluded_already_published_by_id"], 1)
        self.assertEqual(b._pool_stats["excluded_already_published_by_schema_hash"], 1)

    def test_smoke_run_and_manifest(self):
        self._write_reserved()
        cfg_path = os.path.join(self.tmp.name, "cfg.json")
        json.dump(_config(self.tmp.name, self.reserved), open(cfg_path, "w"))
        build_main(["--config", cfg_path, "--smoke-total", "4"])
        out = os.path.join(self.tmp.name, "out_smoke")
        m = json.load(open(os.path.join(out, "generation_manifest.json")))
        self.assertEqual(m["status"], "completed")
        self.assertEqual(sum(m["counts"]["train"].values()), 4)
        self.assertEqual(m["generation_mode"]["standard_modify_strategy"], "original")
        self.assertFalse(os.listdir(os.path.join(out, "benchmark_create", "test")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "out")))  # full-run dir untouched

    def test_event_logging(self):
        events = []
        class Bad:
            def generate(self, *a, **k): raise RuntimeError("boom")
        class Cut:
            def generate(self, *a, **k): return ('{"a": [1, 2', None)
        for inner in (Bad(), Cut()):
            llm = _CountingLLM(inner, sink=lambda k, d: events.append(k))
            try: llm.generate([])
            except RuntimeError: pass
        self.assertEqual(events, ["provider_error", "suspected_truncation"])
        self.assertTrue(_looks_truncated("<think>hmm"))
        self.assertFalse(_looks_truncated('{"a": 1}'))
        self.assertFalse(_looks_truncated("A plain sentence."))


if __name__ == "__main__":
    unittest.main()
