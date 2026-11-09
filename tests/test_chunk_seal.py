"""CPU-only tests for the chunk success marker (chunk_success.json): a valid state file alone is never enough."""
import contextlib, io, json, os, shutil, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import chunked_run as cr
from build_benchmark import main as build_main
from data_generation import chunking
from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.published import TASKS
from test_chunking import ChunkBase, tree_hash


class SealBase(ChunkBase):
    def d(self, name):
        return os.path.join(self.root, name)

    def seal_path(self, name):
        return os.path.join(self.d(name), chunking.SEAL_FILE)

    def seal_again(self, name, calls=None, rc=0):
        return cr.seal_chunk(self.d(name), self.anchor_file, self.reserved, calls or self.calls_log, rc)


class SuccessfulChunkTest(SealBase):
    def test_marker_binds_identity_state_counts_checker_and_calls_log(self):
        self.run_chunk(1, "c01")
        seal = json.load(open(self.seal_path("c01"))); state = chunking.load_state(self.d("c01"))
        self.assertEqual((seal["run_id"], seal["chunk_index"], seal["n_chunks"]), ("c01", 1, self.CHUNKS))
        self.assertEqual(seal["state_sha256"], state["self_sha256"]); self.assertEqual(seal["counts"], state["counts"])
        self.assertEqual((seal["checker"]["code"], seal["checker"]["verdict"]), (0, "SUCCESS"))
        self.assertEqual(seal["calls_log"]["n_failed"], 0); self.assertGreaterEqual(seal["calls_log"]["n_calls"], 1)
        self.assertEqual(seal["calls_log"]["sha256"], chunking.file_sha256(self.calls_log))
        self.assertEqual(seal["build_exit_code"], 0)
        chunking.load_seal(self.d("c01"), state)
        self.assertEqual({r["dir"]: r["state"] for r in cr.status(self.root)}, {"c01": "sealed"})

    def test_sealed_chain_continues_and_merges(self):
        names = self.chain()
        self.assertEqual(cr.validate_chunk(self.d(names[-1]), self.anchor_file)["code"], 0)
        out = os.path.join(self.tmp.name, "merged")
        manifest = cr.merge_chain(self.root, names, self.anchor_file, out)
        self.assertTrue(all(len(c["success_marker_sha256"]) == 64 for c in manifest["chain"]))

    def test_marker_is_never_overwritten(self):
        self.run_chunk(1, "c01"); before = open(self.seal_path("c01")).read()
        res = self.seal_again("c01")
        self.assertEqual(res["code"], 3); self.assertIn("already exists", " ".join(res["technical"]))
        self.assertEqual(before, open(self.seal_path("c01")).read())
        with self.assertRaises(FileExistsError):
            chunking.write_seal(self.d("c01"), chunking.load_state(self.d("c01")), {}, {}, 0)


class TechnicalFailureAfterStateTest(SealBase):
    def test_failed_calls_in_the_log_mean_no_marker_and_the_chunk_is_unusable(self):
        out = self.run_chunk(1, "c01", seal=False)
        self.assertTrue(os.path.exists(os.path.join(out, chunking.STATE_FILE)))            # the state exists ...
        bad = os.path.join(self.tmp.name, "bad_calls.jsonl")
        rows = [json.loads(l) for l in open(self.calls_log) if l.strip()]
        rows[0]["error"] = "Maximum context length is 8192 tokens"
        open(bad, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        res = self.seal_again("c01", calls=bad)
        self.assertEqual(res["code"], 3); self.assertTrue(any("model calls failed" in t for t in res["technical"]))
        self.assertFalse(os.path.exists(self.seal_path("c01")))                            # ... but no marker
        self.assertEqual(cr.validate_chunk(out, self.anchor_file)["code"], 3)
        with self.assertRaises(chunking.ChunkError):
            chunking.load_chain(out)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):     # the builder refuses it as a parent
            self.run_chunk(2, "c02", "c01", seal=False)
        self.assertFalse(os.path.exists(self.d("c02")))
        self.assertTrue(os.path.isdir(out))                                                # preserved for debugging

    def test_fail_closed_without_calls_log_nonzero_build_or_failed_validation(self):
        self.run_chunk(1, "c01", seal=False)
        before = tree_hash(self.d("c01"))
        for kwargs in (dict(calls=os.path.join(self.tmp.name, "missing.jsonl")), dict(rc=1), dict(rc=3), dict(rc=None)):
            res = self.seal_again("c01", **kwargs)
            self.assertNotEqual(res["code"], 0, kwargs)
        self.assertEqual(self.seal_again("c01", rc=2)["code"], 2)                          # build exit 2 stays a shortfall
        empty = os.path.join(self.tmp.name, "empty.jsonl"); open(empty, "w").close()
        self.assertEqual(self.seal_again("c01", calls=empty)["code"], 3)                   # an empty log means no model call
        self.assertEqual(before, tree_hash(self.d("c01")))                                 # nothing written, nothing changed
        # a data problem found at seal time (modified example) also means no marker
        victim = os.path.join(self.d("c01"), "benchmark_fix", "train"); f = sorted(os.listdir(victim))[0]
        p = os.path.join(victim, f); original = open(p).read(); open(p, "w").write(original.replace("{", "{ ", 1))
        self.assertEqual(self.seal_again("c01")["code"], 3); self.assertFalse(os.path.exists(self.seal_path("c01")))
        open(p, "w").write(original)
        self.assertEqual(self.seal_again("c01")["code"], 0); self.assertTrue(os.path.exists(self.seal_path("c01")))


class MarkerIntegrityTest(SealBase):
    def test_missing_malformed_or_inconsistent_marker_fails_closed(self):
        self.run_chunk(1, "c01"); self.run_chunk(2, "c02", "c01")
        good = open(self.seal_path("c01")).read(); seal = json.loads(good)

        def expect_refused(why):
            self.assertEqual(cr.validate_chunk(self.d("c02"), self.anchor_file)["code"], 3, why)   # ancestor marker is checked
            self.assertEqual(cr.validate_chunk(self.d("c01"), self.anchor_file)["code"], 3, why)
            with self.assertRaises(chunking.ChunkError):
                chunking.load_chain(self.d("c01"))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.run_chunk(2, "c02_x" + str(abs(hash(why)) % 10000), "c01", seal=False)

        os.remove(self.seal_path("c01")); expect_refused("absent")
        open(self.seal_path("c01"), "w").write("{not json"); expect_refused("malformed")
        open(self.seal_path("c01"), "w").write("[]"); expect_refused("not an object")
        mutate = {"verdict": lambda s: s["checker"].update(verdict="TECHNICAL_FAILURE", code=3),
                  "counts": lambda s: s["counts"].update(create=s["counts"]["create"] + 1),
                  "calls": lambda s: s["calls_log"].update(n_failed=1),
                  "run_id": lambda s: s.update(run_id="other"),
                  "state_hash": lambda s: s.update(state_sha256="0" * 64),
                  "build_rc": lambda s: s.update(build_exit_code=3)}
        for name, fn in mutate.items():
            s = json.loads(good); fn(s); s.pop("self_sha256")
            s["self_sha256"] = chunking.canonical_sha256(s)                                  # even with a consistent self-hash
            open(self.seal_path("c01"), "w").write(json.dumps(s)); expect_refused(name)
            s["self_sha256"] = "bad"; open(self.seal_path("c01"), "w").write(json.dumps(s)); expect_refused(name + " (self hash)")
        open(self.seal_path("c01"), "w").write(good)
        self.assertEqual(cr.validate_chunk(self.d("c02"), self.anchor_file)["code"], 0)       # restored: valid again
        self.assertEqual(seal["chunk_index"], 1)

    def test_marker_copied_from_another_chunk_is_refused(self):
        self.run_chunk(1, "c01"); self.run_chunk(2, "c02", "c01")
        shutil.copy(self.seal_path("c01"), os.path.join(self.tmp.name, "seal1.json"))
        os.remove(self.seal_path("c02")); shutil.copy(os.path.join(self.tmp.name, "seal1.json"), self.seal_path("c02"))
        self.assertEqual(cr.validate_chunk(self.d("c02"), self.anchor_file)["code"], 3)


class ModifiedAfterApprovalTest(SealBase):
    def test_changed_added_or_removed_data_or_state_invalidates_the_approval(self):
        names = self.chain()
        victim = os.path.join(self.d("c02"), "benchmark_modify", "train"); f = sorted(os.listdir(victim))[0]
        p = os.path.join(victim, f); original = open(p).read()
        for edit in (lambda: open(p, "w").write(original.replace("{", "{ ", 1)),
                     lambda: shutil.copy(p, os.path.join(victim, "extra__a9.json")),
                     lambda: os.remove(p)):
            edit()
            self.assertEqual(cr.validate_chunk(self.d(names[-1]), self.anchor_file)["code"], 3)
            with self.assertRaises(SystemExit):
                cr.merge_chain(self.root, names, self.anchor_file, os.path.join(self.tmp.name, "mm"))
            for extra in os.listdir(victim):
                if extra.startswith("extra"):
                    os.remove(os.path.join(victim, extra))
            open(p, "w").write(original)
        self.assertEqual(cr.validate_chunk(self.d(names[-1]), self.anchor_file)["code"], 0)
        # a re-written (re-hashed) state does not match the marker either
        state = chunking.load_state(self.d("c03")); state.pop("self_sha256"); state["seed"] = 123
        os.remove(os.path.join(self.d("c03"), chunking.STATE_FILE)); chunking.write_state(self.d("c03"), state)
        with self.assertRaises(chunking.ChunkError):
            chunking.load_chain(self.d("c03"))


class InvalidParentAndMergeTest(SealBase):
    def test_unsealed_parent_is_rejected_by_builder_and_script_validation(self):
        self.run_chunk(1, "c01", seal=False)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_chunk(2, "c02", "c01", seal=False)
        self.assertFalse(os.path.exists(self.d("c02")))
        res = cr.validate_chunk(self.d("c01"), self.anchor_file)                           # what the script's PYPARENT runs
        self.assertEqual(res["code"], 3); self.assertTrue(any("success marker" in t or "chunk chain" in t for t in res["technical"]))
        self.assertEqual(cr.validate_chunk(self.d("c01"), self.anchor_file, seal="ancestors")["code"], 0)   # only the sealing job may skip it
        # the builder itself also refuses a parent without marker when called directly (defence in depth)
        class Fake: pass
        fake = Fake(); fake.chunk = {"index": 2, "n_chunks": 3, "parent_dir": self.d("c01"), "totals": {}, "config_sha256": "x"}
        with self.assertRaises(chunking.ChunkError):
            BenchmarkBuilder._chunk_restore(fake, None, None, {})

    def test_merge_rejects_a_failed_chunk_and_a_chain_with_an_unsealed_link(self):
        names = self.chain()
        os.remove(self.seal_path("c02"))
        with self.assertRaises(SystemExit):
            cr.merge_chain(self.root, names, self.anchor_file, os.path.join(self.tmp.name, "m1"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "m1")))
        # a re-run of chunk 2 (new directory) that never got sealed cannot replace it in the chain
        self.run_chunk(2, "c02_retry", "c01", seal=False)
        with self.assertRaises(SystemExit):
            cr.merge_chain(self.root, ["c01", "c02_retry", "c03"], self.anchor_file, os.path.join(self.tmp.name, "m2"))
        self.assertTrue(os.path.isdir(self.d("c02_retry")))                                # preserved

    def test_unreachable_return_in_chunk_restore_now_refuses_a_missing_parent(self):
        class Fake: pass
        fake = Fake(); fake.chunk = {"index": 2, "n_chunks": 3, "parent_dir": None, "totals": {}, "config_sha256": "x"}
        with self.assertRaises(ValueError) as ctx:
            BenchmarkBuilder._chunk_restore(fake, None, None, {})
        self.assertIn("only chunk 1", str(ctx.exception))
        fake.chunk["index"] = 1
        self.assertIsNone(BenchmarkBuilder._chunk_restore(fake, None, None, {}))


if __name__ == "__main__":
    unittest.main()
