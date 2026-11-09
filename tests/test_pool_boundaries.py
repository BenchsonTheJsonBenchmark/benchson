"""Original train/test boundaries and the pool-mismatch diagnosis (offline, read-only)."""
import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

import diagnose_pool_mismatch as dpm
from data_generation.benchmark_builder import BenchmarkBuilder

SOURCES = [{"name": "schemas", "max_schema_kb": 10}, {"name": "jsb_schemas", "max_schema_kb": 10}]


def _make_root(tmp, big=False, broken=False):
    os.makedirs(os.path.join(tmp, "data", "schemas", "test"))
    os.makedirs(os.path.join(tmp, "data", "jsb_schemas", "test"))
    for i in range(24):
        with open(os.path.join(tmp, "data", "schemas", "test", f"plain{i}.json"), "w") as f:
            json.dump({"type": "object", "title": f"plain{i}", "properties": {f"p{i}": {"type": "string"}}}, f)
    for subset in ("A", "B"):
        for i in range(20):
            with open(os.path.join(tmp, "data", "jsb_schemas", "test", f"{subset}{i}.json"), "w") as f:
                json.dump({"name": f"{subset}{i}", "source": "jsonschemabench", "subset": subset,
                           "schema": {"type": "object", "properties": {f"{subset}{i}": {"type": "integer"}}}}, f)
    if big:
        with open(os.path.join(tmp, "data", "schemas", "test", "huge.json"), "w") as f:
            json.dump({"type": "object", "description": "x" * 12000}, f)
    if broken:
        with open(os.path.join(tmp, "data", "schemas", "test", "broken.json"), "w") as f:
            f.write("{not json")


def _pools(tmp):
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        b = BenchmarkBuilder({"sources": SOURCES, "seed": 0, "split": {"test_ratio": 0.15},
                              "counts": {"train": {"create": 1, "fix": 1, "modify": 1}}})
        part = b._partition(b._load_schema_records(apply_exclusions=False))
        sid = lambda r: b._safe_id(r[1], r[2], r[3])
        return {sid(r) for r in part["train"]}, {sid(r) for r in part["test"]}
    finally:
        os.chdir(cwd)


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def _reserved(self, train, test=()):
        p = os.path.join(self.root, "reserved.json")
        with open(p, "w") as f:
            json.dump({"revision": "r", "ids": {"train": sorted(train), "test": sorted(test)},
                       "schema_hashes": {"train": [], "test": []}}, f)
        return p

    def _diag(self, reserved):
        return dpm.diagnose(reserved, SOURCES, root=self.root)

    def test_consistent_published_ids_all_found(self):
        _make_root(self.root)
        train, test = _pools(self.root)
        r = self._diag(self._reserved(train, test))
        self.assertEqual(r["train_id_reasons"], {"in_train_pool": len(train)})
        self.assertEqual(r["published_test_ids_in_test_pool"], len(test))
        self.assertEqual(r["published_test_ids_in_train_pool"], 0)

    def test_every_reason_is_distinguished(self):
        _make_root(self.root, big=True, broken=True)
        train, test = _pools(self.root)
        in_test = sorted(test)[0]
        r = self._diag(self._reserved(list(train)[:5] + [in_test, "schemas__huge", "schemas__broken", "schemas__gone"]))
        reasons = r["train_id_reasons"]
        self.assertEqual(reasons["in_train_pool"], 5)
        self.assertEqual(reasons["in_test_partition"], 1)
        self.assertEqual(reasons["over_size_limit"], 1)
        self.assertEqual(reasons["unreadable_file"], 1)
        self.assertEqual(reasons["not_found"], 1)
        self.assertTrue(any("partitioning" in v for v in r["verdict"]))
        self.assertTrue(any("filtering" in v for v in r["verdict"]))

    def test_groups_are_reported(self):
        _make_root(self.root)
        train, _ = _pools(self.root)
        gone = [f"jsonschemabench__A__missing{i}" for i in range(3)] + ["schemas__missing"]
        r = self._diag(self._reserved(list(train) + gone))
        self.assertEqual(r["train_id_reasons_by_group"]["jsonschemabench/A"]["not_found"], 3)
        self.assertEqual(r["train_id_reasons_by_group"]["schemas"]["not_found"], 1)

    def test_corpus_that_differs_from_original_is_flagged(self):
        _make_root(self.root)
        rep = dpm.corpus_report(self.root)
        self.assertFalse(rep["schemas"]["matches_original_build"])
        self.assertFalse(rep["jsb_schemas"]["matches_original_build"])
        self.assertEqual(rep["schemas"]["files"], 24)

    def test_missing_source_folder_is_reported(self):
        os.makedirs(os.path.join(self.root, "data"))
        self.assertFalse(dpm.corpus_report(self.root)["schemas"]["exists"])

    def test_cli_exit_codes(self):
        _make_root(self.root)
        train, _ = _pools(self.root)
        cfg = os.path.join(self.root, "cfg.json")
        json.dump({"seed": 0, "split": {"test_ratio": 0.15}, "sources": SOURCES}, open(cfg, "w"))
        ok = self._reserved(train)
        old = os.getcwd()
        try:
            os.chdir(self.root)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dpm.main(["--reserved", ok, "--config", cfg, "--root", self.root]), 0)
                self.assertEqual(dpm.main(["--reserved", self._reserved(list(train) + ["schemas__gone"]), "--config", cfg, "--root", self.root]), 1)
                self.assertEqual(dpm.main(["--reserved", os.path.join(self.root, "nope.json"), "--config", cfg, "--root", self.root]), 2)
        finally:
            os.chdir(old)


class SafetyCheckStillStrictTest(unittest.TestCase):
    def test_reuse_pool_still_refuses_a_mismatching_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_root(tmp)
            train, _ = _pools(tmp)
            res = os.path.join(tmp, "base", "reserved_schemas.json"); os.makedirs(os.path.dirname(res))
            ids = sorted(train)[:10] + [f"schemas__gone{i}" for i in range(30)]
            json.dump({"revision": "a" * 40, "ids": {"train": ids, "test": []}, "schema_hashes": {"train": [], "test": []},
                       "reference_hashes_by_schema": {}, "task_keys": {t: {"exact": [], "content": []} for t in ("create", "fix", "modify")}},
                      open(res, "w"))
            cfg = {"scale_only": True, "sources": SOURCES, "exclusions": {"file": res}, "output_root": os.path.join(tmp, "out"),
                   "seed": 0, "split": {"test_ratio": 0.15}, "reuse": {"max_new_references_per_schema": 2},
                   "counts": {"train": {"create": 1, "fix": 1, "modify": 1}},
                   "generation": {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"}}
            old = os.getcwd(); os.chdir(tmp)
            try:
                with self.assertRaisesRegex(ValueError, "published train schemas are in the reconstructed training pool"):
                    BenchmarkBuilder(cfg)._reuse_pool()
            finally:
                os.chdir(old)


def _git_has(commit):
    return subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT, capture_output=True).returncode == 0


class OriginalBoundariesTest(unittest.TestCase):
    """The original v2 build (commit e7f8b22) lists its train/test examples in git. The rebuilt partition must put
    every one of those train ids in the training pool and every test id in the test pool."""

    def setUp(self):
        if not _git_has(dpm.ORIGINAL_COMMIT):
            self.skipTest(f"commit {dpm.ORIGINAL_COMMIT} not in this clone")
        rep = dpm.corpus_report(ROOT)
        if not all(e.get("matches_original_build") for e in rep.values()):
            self.skipTest("data/schemas or data/jsb_schemas is not the original corpus here: "
                          + json.dumps({k: v.get("matches_original_build") for k, v in rep.items()}))

    def test_partition_sizes_and_boundaries(self):
        base = dpm.git_baseline_ids(dpm.ORIGINAL_COMMIT)
        self.assertEqual((len(base["train"]), len(base["test"])), (1267, 189))
        old = os.getcwd(); os.chdir(ROOT)
        try:
            b = BenchmarkBuilder({"sources": SOURCES, "seed": 0, "split": {"test_ratio": 0.15},
                                  "counts": {"train": {"create": 1, "fix": 1, "modify": 1}}})
            records = b._load_schema_records(apply_exclusions=False)
            part = b._partition(records)
        finally:
            os.chdir(old)
        sid = lambda r: b._safe_id(r[1], r[2], r[3])
        train, test = {sid(r) for r in part["train"]}, {sid(r) for r in part["test"]}
        self.assertEqual((len(records), len(test), len(train)), (1669, 250, 1419))
        self.assertEqual(base["train"] - train, set())
        self.assertEqual(base["test"] - test, set())
        self.assertEqual(base["train"] & test, set())
        self.assertEqual(base["test"] & train, set())


if __name__ == "__main__":
    unittest.main()
