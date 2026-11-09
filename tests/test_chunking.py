"""CPU-only tests for chunked generation: exact totals, global per-schema cap, checkpoints, interruption and
recovery, duplicate control, leakage, safe paths and non-overwriting behaviour (mock model, synthetic anchor)."""
import contextlib, hashlib, io, json, os, shutil, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import chunked_run as cr
from build_benchmark import main as build_main
from data_generation import anchor as A
from data_generation import chunking
from data_generation.published import TASKS, schema_hash
from test_anchor_pool import _builder_config, _schema, write_baseline


def tree_hash(path):
    h = hashlib.sha256()
    for base, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(base, f)
            h.update(os.path.relpath(p, path).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


class ChunkBase(unittest.TestCase):
    N_TRAIN, TOTAL, CHUNKS, CAP = 12, 12, 3, 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        train = {f"t{i}.json": _schema(f"t{i}") for i in range(self.N_TRAIN)}
        test = {f"e{i}.json": _schema(f"e{i}") for i in range(3)}
        self.base = os.path.join(self.tmp.name, "pub"); os.makedirs(self.base)
        write_baseline(self.base, train, test); A.write_anchor(self.base)
        self.anchor_file = os.path.join(self.base, A.ANCHOR_FILENAME)
        self.reserved = os.path.join(self.base, "reserved_schemas.json")
        cfg = _builder_config(self.tmp.name, self.base, max_new_references_per_schema=self.CAP)
        cfg["llm_provider"]["module"] = "src.llm.mock.varying_mock_provider"
        cfg["llm_provider"]["class"] = "VaryingMockProvider"
        cfg["counts"] = {"train": {t: self.TOTAL for t in TASKS}}
        self.cfg_path = os.path.join(self.tmp.name, "cfg.json"); json.dump(cfg, open(self.cfg_path, "w"))
        self.root = os.path.join(self.tmp.name, "chunks"); os.makedirs(self.root)

    def run_chunk(self, index, name, parent=None, extra=(), seal=True):
        """Builds a chunk; like the Slurm job it then runs the final validation and, only if that passes, seals it."""
        out = os.path.join(self.root, name)
        self.calls_log = os.path.join(self.tmp.name, "logs", name + ".jsonl"); os.makedirs(os.path.dirname(self.calls_log), exist_ok=True)
        argv = ["--config", self.cfg_path, "--chunks", str(self.CHUNKS), "--chunk-index", str(index), "--out-dir", out,
                "--record-calls", self.calls_log, *extra]
        if parent:
            argv += ["--parent-chunk", os.path.join(self.root, parent)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            build_main(argv)
        if seal:
            res = cr.seal_chunk(out, self.anchor_file, self.reserved, self.calls_log, 0)
            self.assertEqual(res["code"], 0, res)
        return out

    def chain(self, names=None):
        names = names or [f"c{i:02d}" for i in range(1, self.CHUNKS + 1)]
        for i, n in enumerate(names, 1):
            self.run_chunk(i, n, names[i - 2] if i > 1 else None)
        return names


class ExactTotalsAndGlobalCapTest(ChunkBase):
    def test_increments_sum_to_the_totals_exactly(self):
        for totals in ({"create": 1334, "fix": 1333, "modify": 1333}, {"create": 9, "fix": 7, "modify": 1}):
            for n in (1, 3, 8):
                rows = [chunking.increments(totals, n, k) for k in range(1, n + 1)]
                self.assertEqual({t: sum(r[t] for r in rows) for t in totals}, totals)
        self.assertEqual(chunking.increments({"create": 1334, "fix": 1333, "modify": 1333}, 8, 1), {"create": 166, "fix": 166, "modify": 166})
        with self.assertRaises(chunking.ChunkError):
            chunking.increments({"create": 1}, 8, 9)

    def test_chain_reaches_exact_per_task_totals_and_merges(self):
        names = self.chain()
        res = cr.validate_chunk(os.path.join(self.root, names[-1]), self.anchor_file)
        self.assertEqual((res["verdict"], res["cumulative"]), ("SUCCESS", {t: self.TOTAL for t in TASKS}), res)
        before = {n: tree_hash(os.path.join(self.root, n)) for n in names}
        out = os.path.join(self.tmp.name, "merged")
        manifest = cr.merge_chain(self.root, names, self.anchor_file, out)
        self.assertEqual({t: len(v) for t, v in manifest["files"].items()}, {t: self.TOTAL for t in TASKS})
        self.assertEqual(before, {n: tree_hash(os.path.join(self.root, n)) for n in names})   # sources only read
        with self.assertRaises(SystemExit):                                                    # never overwrites
            cr.merge_chain(self.root, names, self.anchor_file, out)
        for t in TASKS:
            self.assertEqual(os.listdir(os.path.join(out, f"benchmark_{t}", "test")), [])
            self.assertEqual(len(os.listdir(os.path.join(out, f"benchmark_{t}", "train"))), self.TOTAL)

    def test_merge_refuses_incomplete_or_wrong_chains(self):
        names = self.chain()
        for bad in (names[:2], [names[0], names[2]], names[::-1]):
            with self.assertRaises(SystemExit):
                cr.merge_chain(self.root, bad, self.anchor_file, os.path.join(self.tmp.name, "m" + str(len(bad)) + bad[0]))

    def test_no_example_leaks_the_test_set_or_leaves_the_train_anchor(self):
        names = self.chain()
        anchor = A.load_anchor(self.anchor_file, self.reserved)
        train, test = {e["schema_hash"] for e in anchor["train"]}, {e["schema_hash"] for e in anchor["test"]}
        for n in names:
            for t in TASKS:
                d = os.path.join(self.root, n, f"benchmark_{t}", "train")
                for f in os.listdir(d):
                    h = schema_hash(json.load(open(os.path.join(d, f)))["schema"])
                    self.assertIn(h, train); self.assertNotIn(h, test)
            self.assertEqual(os.listdir(os.path.join(self.root, n, "benchmark_create", "test")), [])


class TightCapTest(ChunkBase):
    """3 schemas x cap 3 = exactly 9 references: the cap must bind globally, not per chunk."""
    N_TRAIN, TOTAL, CHUNKS, CAP = 3, 9, 3, 3

    def test_each_schema_ends_with_exactly_the_global_cap(self):
        names = self.chain()
        last = chunking.load_chain(os.path.join(self.root, names[-1]))[-1][1]
        self.assertEqual(sorted(r["accepted_references"] for r in last["ledger"].values()), [3, 3, 3])
        res = cr.validate_chunk(os.path.join(self.root, names[-1]), self.anchor_file)
        self.assertEqual(res["verdict"], "SUCCESS", res)

    def test_ledger_and_attempt_numbers_are_carried_not_reset(self):
        self.run_chunk(1, "c01"); self.run_chunk(2, "c02", "c01")
        s1, s2 = (chunking.load_state(os.path.join(self.root, n)) for n in ("c01", "c02"))
        for sid in s1["ledger"]:
            self.assertGreaterEqual(s2["ledger"][sid]["attempts"], s1["ledger"][sid]["attempts"])
            self.assertGreaterEqual(s2["ledger"][sid]["accepted_references"], s1["ledger"][sid]["accepted_references"])
        self.assertTrue(set(s1["files"]["create"]).isdisjoint(s2["files"]["create"]))      # distinct attempt numbers
        self.assertEqual(s2["parent"], {"dir": "c01", "state_sha256": s1["self_sha256"]})


class OverCapacityTest(ChunkBase):
    N_TRAIN, TOTAL, CHUNKS, CAP = 3, 12, 4, 3      # capacity 9 < 12: the last chunk cannot reach its quota

    def test_chunk_that_cannot_meet_its_quota_writes_no_checkpoint(self):
        for i, n in enumerate(("c01", "c02", "c03"), 1):
            self.run_chunk(i, n, f"c0{i - 1}" if i > 1 else None)
        with self.assertRaises(SystemExit) as ctx:
            self.run_chunk(4, "c04", "c03")
        self.assertEqual(ctx.exception.code, 2)
        self.assertFalse(os.path.exists(os.path.join(self.root, "c04", chunking.STATE_FILE)))
        # the global cap held even though the run wanted more
        refs = {}
        for n in ("c01", "c02", "c03", "c04"):
            for f in os.listdir(os.path.join(self.root, n, "benchmark_create", "train")):
                sid, _, att = f[:-5].rpartition("__a"); refs.setdefault(sid, set()).add(att)
        self.assertTrue(all(len(v) <= 3 for v in refs.values()), refs)
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c04"), self.anchor_file)["code"], 3)   # unusable as a parent


class InterruptionAndRecoveryTest(ChunkBase):
    def test_interrupted_chunk_is_kept_unusable_and_recovery_is_explicit(self):
        self.run_chunk(1, "c01")
        with self.assertRaises(BaseException):                      # call budget exhausted mid-chunk = a hard stop
            self.run_chunk(2, "c02_try0", "c01", extra=("--max-llm-calls", "6"))
        partial = os.path.join(self.root, "c02_try0")
        self.assertTrue(os.path.isdir(partial)); self.assertFalse(os.path.exists(os.path.join(partial, chunking.STATE_FILE)))
        frozen = tree_hash(partial)
        rows = {r["dir"]: r for r in cr.status(self.root)}
        self.assertEqual(rows["c01"]["state"], "sealed"); self.assertIn("incomplete", rows["c02_try0"]["state"])
        with self.assertRaises(chunking.ChunkError):                # a partial chunk can never be a parent
            chunking.load_chain(partial)
        with self.assertRaises(SystemExit):                         # ... and the builder refuses it
            self.run_chunk(3, "c03_bad", "c02_try0")
        self.assertFalse(os.path.exists(os.path.join(self.root, "c03_bad")))   # before creating anything
        with self.assertRaises(SystemExit):                         # a new attempt never reuses a directory
            self.run_chunk(2, "c02_try0", "c01")
        self.run_chunk(2, "c02_try1", "c01"); self.run_chunk(3, "c03", "c02_try1")
        names = ["c01", "c02_try1", "c03"]
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c03"), self.anchor_file)["verdict"], "SUCCESS")
        merged = os.path.join(self.tmp.name, "merged")
        cr.merge_chain(self.root, names, self.anchor_file, merged)
        self.assertEqual(frozen, tree_hash(partial))                # the partial output was preserved, never merged
        self.assertFalse(any("try0" in json.dumps(v) for v in json.load(open(os.path.join(merged, "merged_chunks_manifest.json")))["chain"]))

    def test_plan_names_the_next_chunk_only_for_an_explicit_validated_chain(self):
        self.run_chunk(1, "c01")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cr.main(["plan", "--root", self.root, "--chunks", "3", "--totals", "create=12,fix=12,modify=12",
                            "--chain", "c01", "--anchor-file", self.anchor_file])
        plan = json.loads(out.getvalue())
        self.assertEqual(code, 0); self.assertEqual(plan["next"]["CHUNK_INDEX"], 2); self.assertEqual(plan["next"]["PARENT_CHUNK"], "c01")
        self.assertEqual(plan["next"]["quota"], {"create": 4, "fix": 4, "modify": 4})

    def test_cli_requires_explicit_parent_and_fresh_directories(self):
        for argv in (["--chunks", "3", "--chunk-index", "2", "--out-dir", os.path.join(self.root, "x")],
                     ["--chunks", "3", "--chunk-index", "1", "--out-dir", os.path.join(self.root, "y"), "--parent-chunk", self.root],
                     ["--chunks", "3", "--out-dir", os.path.join(self.root, "z")],
                     ["--chunk-index", "1"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                build_main(["--config", self.cfg_path, *argv])
        self.run_chunk(1, "c01")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_chunk(1, "c01")                                # existing output directory


class TamperAndPolicyTest(ChunkBase):
    def test_modified_added_or_removed_files_break_the_chain(self):
        self.chain()
        victim = os.path.join(self.root, "c01", "benchmark_fix", "train")
        name = sorted(os.listdir(victim))[0]
        path = os.path.join(victim, name); original = open(path).read()
        open(path, "w").write(original.replace("{", "{ ", 1))
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c03"), self.anchor_file)["code"], 3)
        open(path, "w").write(original)
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c03"), self.anchor_file)["code"], 0)
        shutil.copy(path, os.path.join(victim, "extra__a9.json"))
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c03"), self.anchor_file)["code"], 3)
        os.remove(os.path.join(victim, "extra__a9.json")); os.remove(path)
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, "c03"), self.anchor_file)["code"], 3)

    def test_state_file_cannot_be_edited_or_overwritten(self):
        self.run_chunk(1, "c01")
        sp = os.path.join(self.root, "c01", chunking.STATE_FILE)
        state = json.load(open(sp)); state["ledger"][next(iter(state["ledger"]))]["accepted_references"] = 0
        json.dump(state, open(sp, "w"))
        with self.assertRaises(chunking.ChunkError):
            chunking.load_state(os.path.join(self.root, "c01"))
        shutil.rmtree(os.path.join(self.root, "c01"))
        self.run_chunk(1, "c01")
        with self.assertRaises(FileExistsError):
            chunking.write_state(os.path.join(self.root, "c01"), {"x": 1})

    def test_a_chain_with_a_different_pool_config_or_plan_is_refused(self):
        self.run_chunk(1, "c01")
        cfg = json.load(open(self.cfg_path)); cfg["reuse"]["max_failures_per_schema"] = 2     # different config => different hash
        other = os.path.join(self.tmp.name, "other.json"); json.dump(cfg, open(other, "w"))
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            build_main(["--config", other, "--chunks", "3", "--chunk-index", "2", "--out-dir", os.path.join(self.root, "c02x"),
                        "--parent-chunk", os.path.join(self.root, "c01")])
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):       # other plan (n_chunks)
            build_main(["--config", self.cfg_path, "--chunks", "4", "--chunk-index", "2", "--out-dir", os.path.join(self.root, "c02y"),
                        "--parent-chunk", os.path.join(self.root, "c01")])
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):       # wrong index for this parent
            build_main(["--config", self.cfg_path, "--chunks", "3", "--chunk-index", "3", "--out-dir", os.path.join(self.root, "c02z"),
                        "--parent-chunk", os.path.join(self.root, "c01")])

    def test_published_baseline_is_never_written(self):
        before = tree_hash(self.base)
        self.chain()
        self.assertEqual(before, tree_hash(self.base))

    def test_validator_catches_exceeded_global_cap_and_duplicates_even_with_consistent_hashes(self):
        """Someone who recomputes every hash still cannot get past the independent recount from the files."""
        names = self.chain()
        d3 = os.path.join(self.root, names[-1])
        state = chunking.load_state(d3)
        victim = next(iter(state["files"]["create"]))
        folder = os.path.join(d3, "benchmark_create", "train")
        dup = victim.replace("__a", "__a1000") if "__a" in victim else victim
        sid = victim[:-5].rpartition("__a")[0]
        for i in range(10):                                      # forge >cap references for one schema, then re-seal the state
            rec = json.load(open(os.path.join(folder, victim))); rec["reference_json"] = {"n": 5000 + i, f"f_{sid}": "x"}
            fake = f"{sid}__a{2000 + i}.json"; json.dump(rec, open(os.path.join(folder, fake), "w"))
            state["files"]["create"][fake] = chunking.file_sha256(os.path.join(folder, fake))
        state["counts"]["create"] = len(state["files"]["create"]); state.pop("self_sha256")
        os.remove(os.path.join(d3, chunking.STATE_FILE)); chunking.write_state(d3, state)
        res = cr.validate_chunk(d3, self.anchor_file, seal="none")   # a re-sealed STATE alone would also fail the marker
        self.assertEqual(res["code"], 4, res)
        self.assertTrue(any("global cap" in i for i in res["invalid"]), res["invalid"][:5])


if __name__ == "__main__":
    unittest.main()
