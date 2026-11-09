"""Published-baseline-anchored pool (pool_source "published_train"): identity, integrity, exclusions, duplicates,
capacity, reproducibility. Offline; the baseline is a synthetic stand-in written with the importer's layout."""
import copy, hashlib, io, json, os, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation import anchor as A
from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.published import TASKS, record_id, reference_hash, schema_hash, task_keys
from data_generation.reuse import ReuseSettings
from build_benchmark import main as build_main, validate_scale_only
from test_import_published import _snapshot, SHA


def _schema(tag, extra=None, pad=0, desc="a field"):
    s = {"type": "object", "title": f"T{tag}", "properties": {f"f_{tag}": {"type": "string", "description": desc},
                                                              "n": {"type": "integer"}}, "required": [f"f_{tag}"]}
    if pad:
        s["description"] = "x" * pad
    if extra:
        s.update(extra)
    return s


def _record(task, name, schema, subset="x"):
    base = {"name": name, "source": "jsb", "subset": subset, "schema": schema}
    if task == "create":
        base.update(description="d", reference_json={"n": 1})
    elif task == "fix":
        base.update(erroneous_json={"n": "1"}, valid_json={"n": 1})
    else:
        base.update(data={"n": 1}, instructions="i", ground_truth={"n": 2}, modification={"op": "x"})
    return base


def write_baseline(root, train, test, revision=SHA, tasks_per_id=None):
    """train/test: {name: schema}. Writes benchmark_<task>/{train,test}/<id>.json + reserved_schemas.json."""
    reserved = {"dataset": "x/y", "revision": revision, "ids": {"train": [], "test": []},
                "schema_hashes": {"train": [], "test": []}, "ids_by_task": {t: {"train": [], "test": []} for t in TASKS}}
    refs, keys = {}, {t: {"exact": set(), "content": set()} for t in TASKS}
    for split, items in (("train", train), ("test", test)):
        for k, (name, schema) in enumerate(items.items()):
            for task in (tasks_per_id or {}).get(name, TASKS if k % 2 == 0 else TASKS[:1]):
                rec = _record(task, name, schema)
                d = os.path.join(root, f"benchmark_{task}", split)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, record_id(rec) + ".json"), "w") as f:
                    json.dump(rec, f)
                reserved["ids_by_task"][task][split].append(record_id(rec))
                refs.setdefault(schema_hash(schema), set()).add(reference_hash(task, rec))
                ex, co = task_keys(task, rec); keys[task]["exact"].add(ex); keys[task]["content"].add(co)
            reserved["ids"][split].append(record_id(_record("create", name, schema)))
            reserved["schema_hashes"][split].append(schema_hash(schema))
    for split in ("train", "test"):
        reserved["ids"][split] = sorted(set(reserved["ids"][split]))
        reserved["schema_hashes"][split] = sorted(set(reserved["schema_hashes"][split]))
    reserved["reference_hashes_by_schema"] = {k: sorted(v) for k, v in refs.items()}
    reserved["task_keys"] = {t: {k: sorted(v) for k, v in d.items()} for t, d in keys.items()}
    with open(os.path.join(root, "reserved_schemas.json"), "w") as f:
        json.dump(reserved, f)
    return reserved


def scenario():
    """A baseline with every special case the pool must handle."""
    train = {f"t{i}.json": _schema(f"t{i}") for i in range(20)}
    test = {f"e{i}.json": _schema(f"e{i}") for i in range(5)}
    train["same_id_as_test.json"] = _schema("shared_id")                 # id also in test (different content)
    test["same_id_as_test.json"] = _schema("shared_id_test")
    train["copy_of_test_content.json"] = copy.deepcopy(test["e0.json"])  # other id, identical content
    train["big.json"] = _schema("big", pad=12000)                        # > 10 KB
    train["dup_a.json"] = _schema("dup")                                 # two ids, same content
    train["dup_b.json"] = _schema("dup")
    train["struct_twin.json"] = _schema("e1", desc="different words")    # same structure as test e1, other text
    return train, test


class AnchorBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, "pub")
        os.makedirs(self.base)
        self.train, self.test = scenario()
        self.reserved = write_baseline(self.base, self.train, self.test)
        self.path = os.path.join(self.base, A.ANCHOR_FILENAME)
        self.anchor = A.write_anchor(self.base)

    def tamper(self, fn):
        data = json.load(open(self.path)); fn(data)
        json.dump(data, open(self.path, "w"))


class IdentityAndIntegrityTest(AnchorBase):
    def test_ids_content_provenance_and_hashes_are_preserved(self):
        by_id = {e["id"]: e for e in self.anchor["train"]}
        self.assertEqual(set(by_id), set(self.reserved["ids"]["train"]))
        for name, schema in self.train.items():
            e = by_id[record_id(_record("create", name, schema))]
            self.assertEqual(e["schema"], schema)
            self.assertEqual((e["name"], e["source"], e["subset"]), (name, "jsb", "x"))
            self.assertEqual(e["schema_hash"], schema_hash(schema))
        self.assertEqual({e["schema_hash"] for e in self.anchor["train"]}, set(self.reserved["schema_hashes"]["train"]))
        self.assertEqual(self.anchor["revision"], SHA)
        self.assertEqual(self.anchor["counts"], {"train_schemas": len(self.train), "test_schemas": len(self.test)})

    def test_tasks_per_schema_are_recorded(self):
        tasks = {e["name"]: e["tasks"] for e in self.anchor["train"]}
        self.assertEqual(tasks["t0.json"], sorted(TASKS)); self.assertEqual(tasks["t1.json"], ["create"])

    def test_loads_and_agrees_with_reserved(self):
        A.load_anchor(self.path, os.path.join(self.base, "reserved_schemas.json"))

    def test_modified_schema_is_rejected(self):
        self.tamper(lambda d: d["train"][0]["schema"].update(extra_key=1))
        with self.assertRaises(A.AnchorError): A.load_anchor(self.path)

    def test_modified_hash_or_count_is_rejected(self):
        self.tamper(lambda d: d["train"][0].update(schema_hash="0" * 64))
        with self.assertRaises(A.AnchorError): A.load_anchor(self.path)

    def test_removed_entry_is_rejected(self):
        self.tamper(lambda d: d["train"].pop())
        with self.assertRaises(A.AnchorError): A.load_anchor(self.path)

    def test_disagreement_with_reserved_is_rejected(self):
        rp = os.path.join(self.base, "reserved_schemas.json")
        res = json.load(open(rp)); res["ids"]["train"].append("jsb__x__ghost"); json.dump(res, open(rp, "w"))
        with self.assertRaisesRegex(A.AnchorError, "disagree"): A.load_anchor(self.path, rp)
        res["ids"]["train"].pop(); res["revision"] = "b" * 40; json.dump(res, open(rp, "w"))
        with self.assertRaisesRegex(A.AnchorError, "revision"): A.load_anchor(self.path, rp)

    def test_missing_or_garbage_file(self):
        with self.assertRaises(A.AnchorError): A.load_anchor(os.path.join(self.base, "nope.json"))
        open(self.path, "w").write("{")
        with self.assertRaises(A.AnchorError): A.load_anchor(self.path)

    def test_same_id_with_two_schemas_across_tasks_is_refused(self):
        d = os.path.join(self.base, "benchmark_fix", "train")
        rid = record_id(_record("create", "t1.json", {}))
        rec = _record("fix", "t1.json", _schema("other"))
        json.dump(rec, open(os.path.join(d, rid + ".json"), "w"))
        with self.assertRaisesRegex(A.AnchorError, "two different schemas"): A.build_anchor(self.base)

    def test_never_overwrites(self):
        with self.assertRaises(A.AnchorError): A.write_anchor(self.base)

    def test_baseline_files_untouched(self):
        before = {}
        for dp, _, fs in os.walk(self.base):
            for f in fs:
                if f != A.ANCHOR_FILENAME:
                    before[os.path.join(dp, f)] = hashlib.sha256(open(os.path.join(dp, f), "rb").read()).hexdigest()
        os.remove(self.path); A.write_anchor(self.base)
        for p, h in before.items():
            self.assertEqual(hashlib.sha256(open(p, "rb").read()).hexdigest(), h)


class ExclusionsAndStatisticsTest(AnchorBase):
    def test_exact_exclusion_counts_and_kept_ids(self):
        kept, st = A.select_pool(self.anchor, 10)
        self.assertEqual(st["published_train_schemas"], 26)
        self.assertEqual(st["excluded_published_test_by_id"], 1)
        self.assertEqual(st["excluded_published_test_by_content_hash"], 1)
        self.assertEqual(st["excluded_over_size_limit"], 1)
        self.assertEqual(st["removed_content_duplicates_in_pool"], 1)
        self.assertEqual(st["usable_schemas"], len(kept))
        self.assertEqual(len(kept), 26 - 4)
        names = {e["name"] for e in kept}
        for gone in ("same_id_as_test.json", "copy_of_test_content.json", "big.json"):
            self.assertNotIn(gone, names)
        self.assertEqual(sum(n in names for n in ("dup_a.json", "dup_b.json")), 1)
        self.assertEqual(sum(st["usable_by_source"].values()), len(kept))

    def test_no_kept_schema_overlaps_test_by_id_or_hash(self):
        kept, _ = A.select_pool(self.anchor, 10)
        self.assertFalse({e["id"] for e in kept} & {e["id"] for e in self.anchor["test"]})
        self.assertFalse({e["schema_hash"] for e in kept} & {e["schema_hash"] for e in self.anchor["test"]})
        self.assertEqual(len({e["schema_hash"] for e in kept}), len(kept))

    def test_structural_overlap_is_reported_not_filtered(self):
        kept, st = A.select_pool(self.anchor, 10)
        self.assertIn("struct_twin.json", {e["name"] for e in kept})
        so = st["structural_overlap_with_published_test"]
        self.assertGreaterEqual(so["pool_schemas"], 1); self.assertFalse(so["filtered"])
        self.assertTrue(any("struct_twin" in i for i in so["ids_first_20"]))
        self.assertFalse(st["structural_duplicates_inside_pool"]["filtered"])

    def test_structure_hash_ignores_text_but_not_property_names(self):
        a = _schema("p", desc="one"); b = _schema("p", desc="two")
        self.assertEqual(A.structure_hash(a), A.structure_hash(b))
        self.assertNotEqual(A.structure_hash(a), A.structure_hash(_schema("q")))
        named_title = {"type": "object", "properties": {"title": {"type": "string"}}}
        self.assertNotEqual(A.structure_hash(named_title), A.structure_hash({"type": "object", "properties": {}}))

    def test_size_restriction_is_explicit_and_documented(self):
        _, st10 = A.select_pool(self.anchor, 10); kept50, st50 = A.select_pool(self.anchor, 50)
        self.assertEqual(st50["excluded_over_size_limit"], 0)
        self.assertEqual(len(kept50), st10["usable_schemas"] + 1)
        self.assertTrue(st10["size_restriction"]["experimental_restriction_not_a_claim_about_the_hf_build"])
        big = next(e for e in self.anchor["train"] if e["name"] == "big.json")
        self.assertEqual(big["size_bytes"], A.schema_size_bytes(big["schema"]))
        self.assertGreater(big["size_bytes"], 10 * 1024)

    def test_dedupe_can_be_switched_off_like_the_reconstructed_mode(self):
        kept, st = A.select_pool(self.anchor, 10, dedupe_by_content=False)
        self.assertEqual(st["removed_content_duplicates_in_pool"], 0); self.assertEqual(len(kept), 23)


class ReproducibilityTest(unittest.TestCase):
    def test_identical_across_builds_and_file_creation_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, test = scenario()
            shas = []
            for k, order in enumerate((list(train), list(reversed(list(train))))):
                root = os.path.join(tmp, f"b{k}"); os.makedirs(root)
                write_baseline(root, {n: train[n] for n in order}, test, tasks_per_id={n: TASKS for n in train})
                shas.append((A.build_anchor(root)["anchor_sha256"], A.build_anchor(root)["anchor_sha256"]))
            self.assertEqual(shas[0][0], shas[0][1]); self.assertEqual(shas[0][0], shas[1][0])


def _builder_config(tmp, base, **reuse):
    return {"scale_only": True, "llm_provider": {"module": "src.llm.mock.mock_provider", "class": "MockProvider", "params": {}},
            "sources": [{"name": "schemas", "max_schema_kb": 10}, {"name": "jsb_schemas", "max_schema_kb": 10}],
            "exclusions": {"file": os.path.join(base, "reserved_schemas.json")},
            "output_root": os.path.join(tmp, "out"), "seed": 0, "split": {"test_ratio": 0.15},
            "reuse": {"pool_source": "published_train", "anchor_max_schema_kb": 10, "max_new_references_per_schema": 3, **reuse},
            "counts": {"train": {"create": 1334, "fix": 1333, "modify": 1333}},
            "validation": {"train": {"round_trip": False}},
            "generation": {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"}}


class BuilderIntegrationTest(AnchorBase):
    def _builder(self, **reuse):
        return BenchmarkBuilder(_builder_config(self.tmp.name, self.base, **reuse))

    def test_pool_needs_no_data_folder_and_is_deterministic(self):
        old = os.getcwd(); os.chdir(self.tmp.name)       # no data/ here: the pool must not touch the corpus
        try:
            a, b = self._builder()._reuse_pool(), self._builder()._reuse_pool()
        finally:
            os.chdir(old)
        self.assertEqual([r[3] for r in a], [r[3] for r in b])
        self.assertEqual(len(a), 22)

    def test_pool_tuples_carry_published_identity(self):
        pool = self._builder()._reuse_pool()
        by_name = {r[3]: r for r in pool}
        self.assertEqual(by_name["t3.json"][0], self.train["t3.json"])
        self.assertEqual((by_name["t3.json"][1], by_name["t3.json"][2]), ("jsb", "x"))

    def test_capacity_report_counts_pool_and_targets(self):
        b = self._builder(); rep = b.reuse_capacity_report(0.7)
        self.assertEqual(rep["pool"]["usable_schemas"], 22); self.assertEqual(rep["pool"]["pool_source"], "published_train")
        self.assertEqual(rep["max_new_references"], 22 * 3)
        self.assertEqual({t: v["target"] for t, v in rep["per_task"].items()}, {"create": 1334, "fix": 1333, "modify": 1333})
        self.assertFalse(any(v["feasible_at_assumed_acceptance"] for v in rep["per_task"].values()))   # 63 refs << 1333

    def test_corrupt_anchor_stops_the_builder(self):
        self.tamper(lambda d: d["train"][0]["schema"].update(x=1))
        with self.assertRaisesRegex(ValueError, "published_train pool refused"): self._builder()._reuse_pool()

    def test_missing_anchor_stops_the_builder(self):
        os.remove(self.path)
        with self.assertRaisesRegex(ValueError, "export_anchor_pool"): self._builder()._reuse_pool()

    def test_settings_validation(self):
        with self.assertRaises(ValueError): ReuseSettings(pool_source="published_train")                      # kb required
        with self.assertRaises(ValueError): ReuseSettings(pool_source="published_train", anchor_max_schema_kb=0)
        with self.assertRaises(ValueError): ReuseSettings(pool_source="nonsense")
        with self.assertRaises(ValueError): ReuseSettings(anchor_max_schema_kb=10)                            # reconstructed mode
        self.assertEqual(ReuseSettings().pool_source, "reconstructed")

    def test_scale_only_validation(self):
        cfg = _builder_config(self.tmp.name, self.base)
        self.assertEqual(validate_scale_only(cfg), [])
        bad = copy.deepcopy(cfg); bad["reuse"]["anchor_max_schema_kb"] = 20
        self.assertTrue(any("anchor_max_schema_kb" in p for p in validate_scale_only(bad)))
        os.remove(self.path)
        self.assertTrue(any("anchor file not found" in p for p in validate_scale_only(cfg)))
        for key, val in (("seed", 1), ("split", {"test_ratio": 0.2})):                  # scientific settings stay guarded
            worse = copy.deepcopy(cfg); worse[key] = val
            self.assertTrue(validate_scale_only(worse))

    def test_repo_configs_use_the_anchored_pool_with_the_original_targets(self):
        for name in ("scale_only_vllm_granite.json", "scale_only_mock.json"):
            cfg = json.load(open(os.path.join(ROOT, "configs", name)))
            self.assertEqual(cfg["reuse"]["pool_source"], "published_train"); self.assertEqual(cfg["reuse"]["anchor_max_schema_kb"], 10)
            self.assertEqual(cfg["counts"]["train"], {"create": 1334, "fix": 1333, "modify": 1333})
            self.assertEqual((cfg["seed"], cfg["split"]["test_ratio"]), (0, 0.15))
            self.assertEqual([s["name"] for s in cfg["sources"]], ["schemas", "jsb_schemas"])
            self.assertEqual(cfg["generation"]["standard_modify_strategy"], "original")


class EndToEndMockTest(unittest.TestCase):
    """A real mock-model build on anchored data: every new example's schema comes from the published train set."""

    def setUp(self):
        src = os.path.join(ROOT, "data", "schemas", "test")
        if not os.path.isdir(src):
            self.skipTest("data/schemas not present")
        files = [f for f in sorted(os.listdir(src)) if f.endswith(".json") and os.path.getsize(os.path.join(src, f)) < 4000][:40]
        if len(files) < 30:
            self.skipTest("not enough small schemas")
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        schemas = {f: json.load(open(os.path.join(src, f))) for f in files}
        self.train = dict(list(schemas.items())[:30]); self.test = dict(list(schemas.items())[30:])
        self.base = os.path.join(self.tmp.name, "pub"); os.makedirs(self.base)
        write_baseline(self.base, self.train, self.test, tasks_per_id={n: TASKS for n in schemas})
        A.write_anchor(self.base)

    def test_smoke_build_uses_only_published_train_schemas(self):
        cfg = _builder_config(self.tmp.name, self.base, max_new_references_per_schema=2)
        cfg_path = os.path.join(self.tmp.name, "cfg.json"); json.dump(cfg, open(cfg_path, "w"))
        out = os.path.join(self.tmp.name, "smoke")
        with redirect_stdout(io.StringIO()):
            build_main(["--config", cfg_path, "--smoke-total", "6", "--out-dir", out])
        anchor = A.load_anchor(os.path.join(self.base, A.ANCHOR_FILENAME), os.path.join(self.base, "reserved_schemas.json"))
        train_hashes = {e["schema_hash"] for e in anchor["train"]}; test_hashes = {e["schema_hash"] for e in anchor["test"]}
        n = 0
        for task in TASKS:
            d = os.path.join(out, f"benchmark_{task}", "train")
            for fn in os.listdir(d):
                rec = json.load(open(os.path.join(d, fn))); n += 1
                self.assertIn(schema_hash(rec["schema"]), train_hashes)
                self.assertNotIn(schema_hash(rec["schema"]), test_hashes)
        self.assertEqual(n, 6)
        self.assertEqual(os.listdir(os.path.join(out, "benchmark_create", "test")), [])
        man = json.load(open(os.path.join(out, "generation_manifest.json")))
        self.assertEqual(man["schema_reuse"]["pool_source"], "published_train")

    def test_check_pool_stdout_is_pure_json_with_each_task_checked_separately(self):
        """The Slurm preflight pipes stdout of --check-pool into a JSON parser: nothing else may be printed there."""
        cfg = _builder_config(self.tmp.name, self.base, max_new_references_per_schema=2)
        cfg_path = os.path.join(self.tmp.name, "cfg.json"); json.dump(cfg, open(cfg_path, "w"))
        r = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "src", "build_benchmark.py"), "--config", cfg_path,
                            "--smoke-total", "6", "--out-dir", os.path.join(self.tmp.name, "chk"), "--check-pool"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["pool"]["pool_source"], "published_train")
        self.assertEqual(report["pool"]["anchor_sha256"], A.load_anchor(os.path.join(self.base, A.ANCHOR_FILENAME))["anchor_sha256"])
        self.assertEqual({t: row["target"] for t, row in report["per_task"].items()}, {"create": 2, "fix": 2, "modify": 2})
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "chk")))   # a pool check writes no output


class ImporterWritesAnchorTest(unittest.TestCase):
    def test_import_creates_a_verified_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
            _snapshot(src)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "import_published_benchmark.py"), "--source-dir", src,
                                "--out-dir", out, "--revision", SHA, "--allow-count-mismatch"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            anchor = A.load_anchor(os.path.join(out, A.ANCHOR_FILENAME), os.path.join(out, "reserved_schemas.json"))
            self.assertEqual(anchor["counts"], {"train_schemas": 3, "test_schemas": 2})
            man = json.load(open(os.path.join(out, "published_manifest.json")))
            self.assertEqual(man["anchor_pool"]["anchor_sha256"], anchor["anchor_sha256"])

    def test_export_cli_writes_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, test = scenario(); write_baseline(tmp, train, test)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "export_anchor_pool.py"), "--baseline-dir", tmp],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('"usable_schemas": 22', r.stdout)
            r2 = subprocess.run([sys.executable, os.path.join(ROOT, "src", "export_anchor_pool.py"), "--baseline-dir", tmp],
                                capture_output=True, text=True)
            self.assertEqual(r2.returncode, 1); self.assertIn("refusing to overwrite", r2.stdout)


if __name__ == "__main__":
    unittest.main()
