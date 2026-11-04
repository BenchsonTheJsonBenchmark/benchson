import hashlib, json, os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.published import canon_hash, reference_hash, schema_hash, task_keys
from data_generation.reuse import ReuseSettings, SchemaLedger, DuplicateIndex
from llm.mock.mock_provider import MockProvider, make_instance


def schema_for(i):
    return {"type": "object", "title": f"s{i}",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}, "c": {"type": "integer"}},
            "required": ["a", "b", "c"]}


class VaryingMock(MockProvider):
    """Mock whose reference objects differ on every call (integers are bumped), except for schemas
    whose title is in `stuck`, which always return the same object."""
    stuck = ()

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n = 0

    def _generate(self, messages, parameters=None):
        text = super()._generate(messages, parameters)
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(obj, dict) and set(obj) == {"a", "b", "c"}:
            title = next((m["content"] for m in messages if '"title": "s' in m["content"]), "")
            if not any(f'"title": "{t}"' in title for t in self.stuck):
                self.n += 1
                obj["a"] += self.n
                obj["c"] += 2 * self.n
            return json.dumps(obj, indent=2)
        return text


def make_config(tmp, reserved, counts, **reuse):
    return {"llm_provider": {}, "sources": [{"name": "mini", "max_schema_kb": 10}], "seed": 0,
            "exclusions": {"file": reserved}, "output_root": os.path.join(tmp, "out"),
            "split": {"test_ratio": 0.15}, "counts": {"train": counts},
            "validation": {"train": {"round_trip": False}},
            "generation": {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"},
            "reuse": {"max_new_references_per_schema": 2, **reuse}}


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.old)
        d = os.path.join(self.tmp.name, "data", "mini", "test")
        os.makedirs(d)
        for i in range(20):
            json.dump(schema_for(i), open(os.path.join(d, f"s{i:02d}.json"), "w"))
        # s19 is an exact content copy of s18 (pool content duplicate)
        json.dump(schema_for(18), open(os.path.join(d, "s19.json"), "w"))
        self.reserved = os.path.join(self.tmp.name, "reserved.json")

    def write_reserved(self, test_ids=(), test_hashes=(), train_ids=(), refs=None):
        json.dump({"dataset": "x", "revision": "a" * 40, "ids": {"train": list(train_ids), "test": list(test_ids)},
                   "schema_hashes": {"train": [], "test": list(test_hashes)},
                   "reference_hashes_by_schema": {k: sorted(v) for k, v in (refs or {}).items()},
                   "task_keys": {t: {"exact": [], "content": []} for t in ("create", "fix", "modify")}},
                  open(self.reserved, "w"))

    def build(self, counts, provider=None, **reuse):
        b = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, counts, **reuse))
        b.build(provider or VaryingMock())
        return b

    def test_pool_excludes_test_and_dedupes(self):
        self.write_reserved()
        b = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        pool = b._reuse_pool()
        ids = {b._safe_id(r[1], r[2], r[3]) for r in pool}
        part_test = {b._safe_id(r[1], r[2], r[3]) for r in b._partition(b._load_schema_records(False))["test"]}
        self.assertFalse(ids & part_test)                       # only the original training pool
        banned = sorted(ids)[0]
        rec = next(r for r in pool if b._safe_id(r[1], r[2], r[3]) == banned)
        self.write_reserved(test_ids=[banned])
        b2 = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        self.assertNotIn(banned, {b2._safe_id(r[1], r[2], r[3]) for r in b2._reuse_pool()})
        self.assertEqual(b2._pool_stats["removed_published_test_by_id"], 1)
        self.write_reserved(test_hashes=[schema_hash(rec[0])])  # same content under another id is excluded too
        b3 = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        pool3 = b3._reuse_pool()
        self.assertTrue(all(schema_hash(r[0]) != schema_hash(rec[0]) for r in pool3))

    def test_multiple_distinct_examples_per_schema_with_cap(self):
        self.write_reserved()
        b = self.build({"create": 12, "fix": 12, "modify": 12})
        out = os.path.join(self.tmp.name, "out")
        files = sorted(os.listdir(os.path.join(out, "benchmark_fix", "train")))
        self.assertEqual(len(files), 12)
        schemas = {}
        for f in files:
            rec = json.load(open(os.path.join(out, "benchmark_fix", "train", f)))
            schemas.setdefault(f.rpartition("__a")[0], set()).add(reference_hash("fix", rec))
        self.assertTrue(any(len(v) > 1 for v in schemas.values()) or len(schemas) == 12)
        # every file of one schema holds a different reference; cap respected
        for sid, refs in schemas.items():
            self.assertLessEqual(len(refs), 2)
        summary = b._reuse_summary()
        self.assertLessEqual(max(summary["new_references_per_schema_histogram"]), 2)
        # all three tasks share the 1/3 split
        counts = {t: len(os.listdir(os.path.join(out, f"benchmark_{t}", "train"))) for t in ("create", "fix", "modify")}
        self.assertEqual(counts, {"create": 12, "fix": 12, "modify": 12})

    def test_reuse_reaches_target_larger_than_pool(self):
        self.write_reserved()
        b = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        pool_size = len(b._reuse_pool())
        target = pool_size * 2   # impossible with one example per schema
        b = self.build({"create": target, "fix": target, "modify": target})
        out = os.path.join(self.tmp.name, "out")
        self.assertEqual(len(os.listdir(os.path.join(out, "benchmark_modify", "train"))), target)
        refs = set()
        for t in ("create", "fix", "modify"):
            for f in os.listdir(os.path.join(out, f"benchmark_{t}", "train")):
                rec = json.load(open(os.path.join(out, f"benchmark_{t}", "train", f)))
                refs.add((f.rpartition("__a")[0], reference_hash(t, rec)))
        self.assertEqual(len(refs), pool_size * 2)       # genuinely different references, 2 per schema
        report = b.last_summary["schema_reuse"]
        self.assertEqual(report["new_references_per_schema_histogram"], {2: pool_size})

    def test_duplicate_of_published_reference_is_rejected_and_schema_suspended(self):
        self.write_reserved()
        probe = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        pool = probe._reuse_pool()
        stuck = [r for r in pool][:2]
        titles = [r[0]["title"] for r in stuck]
        refs = {}
        for r in stuck:   # published references equal to what the stuck mock always returns
            inst = make_instance(r[0])
            refs[schema_hash(r[0])] = {canon_hash(inst)}
        self.write_reserved(refs=refs)
        VaryingMock.stuck = tuple(titles)
        self.addCleanup(setattr, VaryingMock, "stuck", ())
        n = 2 * (len(pool) - 2)   # needs both passes over the healthy schemas
        b = self.build({"create": n, "fix": n, "modify": n}, max_failures_per_schema=1, suspend_passes=0,
                       max_suspensions=1)
        events = b._reuse_state["ledger"].events
        stuck_ids = {probe._safe_id(r[1], r[2], r[3]) for r in stuck}
        kinds = {(e["schema_id"], e["event"]) for e in events}
        for sid in stuck_ids:
            self.assertIn((sid, "suspended"), kinds)
            self.assertIn((sid, "retired"), kinds)
        self.assertGreater(b._reuse_state["dup"].stats["rejected_reference_same_as_published"], 0)
        out = os.path.join(self.tmp.name, "out")
        self.assertEqual(len(os.listdir(os.path.join(out, "benchmark_fix", "train"))), n)   # target still met
        written = {f.rpartition("__a")[0] for f in os.listdir(os.path.join(out, "benchmark_fix", "train"))}
        self.assertFalse(written & stuck_ids)
        skipped = os.listdir(os.path.join(out, "_generation_reports"))[0]
        self.assertTrue(os.path.exists(os.path.join(out, "_generation_reports", skipped, "skipped_schemas.jsonl")))

    def test_bounded_when_everything_fails(self):
        self.write_reserved()
        VaryingMock.stuck = tuple(f"s{i}" for i in range(20))
        self.addCleanup(setattr, VaryingMock, "stuck", ())
        probe = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        refs = {schema_hash(r[0]): {canon_hash(make_instance(r[0]))} for r in probe._reuse_pool()}
        self.write_reserved(refs=refs)
        b = self.build({"create": 5, "fix": 5, "modify": 5}, max_failures_per_schema=1, max_suspensions=0)
        self.assertEqual(len(os.listdir(os.path.join(self.tmp.name, "out", "benchmark_fix", "train"))), 0)
        self.assertTrue(b.last_summary["shortfall"])
        self.assertEqual(b._reuse_summary()["schema_status"], {"retired": len(probe._reuse_pool())})

    def test_resume_keeps_counts_and_does_not_repeat_references(self):
        self.write_reserved()
        self.build({"create": 6, "fix": 6, "modify": 6})
        before = sorted(os.listdir(os.path.join(self.tmp.name, "out", "benchmark_fix", "train")))
        self.build({"create": 6, "fix": 6, "modify": 6})
        self.assertEqual(before, sorted(os.listdir(os.path.join(self.tmp.name, "out", "benchmark_fix", "train"))))
        self.build({"create": 10, "fix": 10, "modify": 10})
        out = os.path.join(self.tmp.name, "out")
        seen = []
        for f in os.listdir(os.path.join(out, "benchmark_fix", "train")):
            rec = json.load(open(os.path.join(out, "benchmark_fix", "train", f)))
            seen.append((f.rpartition("__a")[0], reference_hash("fix", rec)))
        self.assertEqual(len(seen), 10)
        self.assertEqual(len(set(seen)), 10)

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            ReuseSettings(max_new_references_per_schema=0)
        with self.assertRaises(ValueError):
            ReuseSettings(nonsense=1)
        self.assertIsNone(ReuseSettings.from_config({}))
        self.write_reserved()
        cfg = make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1})
        cfg["generation"] = {"standard_ratio": 0.8, "complex_ratio": 0.2, "standard_modify_strategy": "original"}
        with self.assertRaises(ValueError):
            BenchmarkBuilder(cfg)

    def test_pool_sanity_aborts_on_mismatched_corpus(self):
        self.write_reserved(train_ids=["not_in_pool_1", "not_in_pool_2"])
        b = BenchmarkBuilder(make_config(self.tmp.name, self.reserved, {"create": 1, "fix": 1, "modify": 1}))
        with self.assertRaises(ValueError):
            b._reuse_pool()


class LedgerTest(unittest.TestCase):
    def test_suspension_then_retirement(self):
        s = ReuseSettings(max_failures_per_schema=2, suspend_passes=1, max_suspensions=1, max_new_references_per_schema=3)
        led = SchemaLedger(s)
        led.entry("a", "h")
        for _ in range(2):
            led.begin_attempt("a"); ev = led.finish_attempt("a", [], 1, "x")
        self.assertEqual(ev["event"], "suspended")
        self.assertFalse(led.eligible("a", 2)); self.assertTrue(led.eligible("a", 3))
        for _ in range(2):
            led.begin_attempt("a"); ev = led.finish_attempt("a", [], 3, "x")
        self.assertEqual(ev["event"], "retired"); self.assertFalse(led.eligible("a", 99))


if __name__ == "__main__":
    unittest.main()
