"""One-time, authorized, fail-closed recovery of chunk c01_24044333-r1 with its three historical token-count timeouts."""
import contextlib, io, json, os, shutil, sys, unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
import chunked_run as cr
from data_generation import chunking
from data_generation.published import TASKS
from test_chunk_chain import read
from test_chunk_recovery import JOB, NAME, RecoveryBase
from test_chunking import tree_hash

IDS = list(cr.HISTORICAL_POLICY["ids"])
POLICY = cr.HISTORICAL_POLICY["policy"]


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


class HistBase(RecoveryBase):
    """The unsealed chunk, plus what job 24044333 additionally left behind: 3 provider_error events, 9 worker_error failures,
    ledger rows that count the 3 failed attempts, and a guard whose counters match the calls log."""

    REAL_IDS = True      # the hard-coded ids of the real chunk; a subclass uses ids of schemas that exist in the test pool

    def setUp(self):
        super().setUp()
        ledger = chunking.load_state(self.chunk)["ledger"]
        self.ids = list(IDS) if self.REAL_IDS else [f"{sid}__a9" for sid in sorted(ledger)[:3]]
        if not self.REAL_IDS:
            patcher = mock.patch.dict(cr.HISTORICAL_POLICY, {"ids": tuple(self.ids)}); patcher.start(); self.addCleanup(patcher.stop)
        self.report = csr.latest_report_dir(self.chunk)
        self.events_file = os.path.join(self.report, "generation_events.jsonl")
        self.failures_file = os.path.join(self.report, "generation_failures.jsonl")
        self.n_calls = len(read(self.calls).splitlines())
        self.set_manifest(lambda m: m.update(context_guard={"enabled": True, "max_model_len": 8192, "checked": self.n_calls,
                                                           "overflow": 0, "largest_prompt_tokens": 3900}))
        self.append(self.events_file, [{"split": "train", "schema_id": i, "kind": "provider_error", "detail": "TimeoutError: timed out"}
                                       for i in self.ids])
        self.append(self.failures_file, [{"task": t, "mode": "standard", "schema_id": i, "stage": "worker_error", "detail": "timed out",
                                          "split": "train", "schema": i.rpartition("__a")[0], "attempt": 1} for i in self.ids for t in TASKS])

        def ledger_rows(state):
            for i in self.ids:
                sid = i.rpartition("__a")[0]
                row = state["ledger"].get(sid) or {
                    "accepted_references": 0, "attempts": 0, "consecutive_failures": 0, "examples": {t: 0 for t in TASKS},
                    "failed_attempts": 0, "published_references": 1, "reason": None, "schema_hash": "0" * 64, "schema_id": sid,
                    "status": "active", "suspended_until_pass": 0, "suspensions": 0}
                row.update(attempts=row["attempts"] + 1, failed_attempts=row["failed_attempts"] + 1, reason="worker_error")
                state["ledger"][sid] = row
        self.set_state(ledger_rows)

    def append(self, path, rows):
        with open(path, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))

    def set_manifest(self, fn):
        path = os.path.join(self.chunk, "generation_manifest.json"); m = json.load(open(path)); fn(m); json.dump(m, open(path, "w"))

    def set_state(self, fn):
        state = chunking.load_state(self.chunk); os.remove(os.path.join(self.chunk, chunking.STATE_FILE))
        fn(state); state.pop("self_sha256"); chunking.write_state(self.chunk, state)

    def recover(self, policy=POLICY, ids=None, **over):
        args = dict(chunk_dir=self.chunk, anchor_file=self.anchor_file, reserved_file=self.reserved, calls_log=self.calls,
                    slurm_log=self.slurm_log, config_file=self.cfg_path, require_context_guard=8192, confirm=NAME)
        args.update(over)
        return cr.recover_seal(args["chunk_dir"], args["anchor_file"], args["reserved_file"], args["calls_log"], args["slurm_log"],
                               args["config_file"], args["require_context_guard"], args["confirm"],
                               historical_policy=policy, historical_ids=self.ids if ids is None else ids)

    def strict(self):
        return RecoveryBase.recover(self)

    def seal(self):
        return json.load(open(os.path.join(self.chunk, chunking.SEAL_FILE)))


class StrictDefaultTest(HistBase):
    def test_without_the_explicit_authorization_the_incidents_block_recovery_and_validation(self):
        before = tree_hash(self.chunk)
        res = self.strict()
        self.assertEqual(res["code"], 3); self.assertEqual(before, tree_hash(self.chunk))
        self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE))); self.assertTrue(any("provider_error" in t or "worker error" in t for t in res["technical"]))
        self.assertEqual(cr.validate_chunk(self.chunk, self.anchor_file, self.reserved, self.calls, 0, 8192, seal="ancestors")["code"], 3)
        self.assertEqual(csr.check(self.chunk, per_task=chunking.load_state(self.chunk)["quota"])["code"], 3)

    def test_authorization_needs_both_options_and_the_exact_values(self):
        for kw in (dict(policy=None), dict(ids=[]), dict(policy="historical-token-count-timeouts/2"), dict(policy="any-timeout")):
            res = self.recover(**kw)
            self.assertNotEqual(res["code"], 0, kw); self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)), kw)

    def test_wrong_additional_or_missing_ids_are_refused(self):
        other = "schemas__other__a1"
        for ids in (self.ids[:2], self.ids + [other], self.ids[:2] + [other], [self.ids[0], self.ids[0], self.ids[1]], [i.replace("__a1", "__a2") for i in self.ids],
                    [i.replace("schemas__", "x__") for i in self.ids]):
            before = tree_hash(self.chunk)
            res = self.recover(ids=ids)
            self.assertNotEqual(res["code"], 0, ids); self.assertEqual(before, tree_hash(self.chunk))
            self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)))

    def test_the_policy_applies_only_to_its_own_chunk(self):
        bad = cr.historical_authorization_problems("c02_24044333-r1", POLICY, self.ids)
        self.assertTrue(any("applies only to chunk" in b for b in bad))
        self.assertEqual(cr.historical_authorization_problems(NAME, POLICY, self.ids), [])
        res = cr.validate_chunk(self.chunk, self.anchor_file, self.reserved, self.calls, 0, 8192, seal="ancestors",
                                historical={**cr.historical_spec(), "ids": self.ids[:1]})   # a caller cannot widen or narrow it
        self.assertEqual(res["code"], 3); self.assertTrue(any("not the hard-coded policy" in t for t in res["technical"]))


class AuthorizedRecoveryTest(HistBase):
    def test_recovers_records_the_policy_and_the_evidence_and_touches_nothing_else(self):
        before = {n: read(os.path.join(self.chunk, n)) for n in (chunking.STATE_FILE, "generation_manifest.json")}
        tree = tree_hash(self.chunk); calls, slurm = read(self.calls), read(self.slurm_log)
        events, failures = read(self.events_file), read(self.failures_file)
        res = self.recover()
        self.assertEqual(res["code"], 0, res); self.assertTrue(res["sealed"] and res["recovered"])
        self.assertEqual(res["tolerated"]["historical"]["events"], 3); self.assertEqual(res["tolerated"]["historical"]["failures"], 9)
        for n, text in before.items():
            self.assertEqual(read(os.path.join(self.chunk, n)), text)
        self.assertEqual((read(self.calls), read(self.slurm_log), read(self.events_file), read(self.failures_file)),
                         (calls, slurm, events, failures))
        files_now = tree_hash(self.chunk)
        self.assertNotEqual(tree, files_now)                                                   # only the marker was added ...
        os.rename(os.path.join(self.chunk, chunking.SEAL_FILE), os.path.join(self.tmp.name, "marker")); self.assertEqual(tree, tree_hash(self.chunk))
        os.rename(os.path.join(self.tmp.name, "marker"), os.path.join(self.chunk, chunking.SEAL_FILE))
        seal = self.seal(); tol = seal["recovery"]["historical_tolerance"]
        self.assertEqual((tol["policy"], tol["chunk"], tol["ids"]), (POLICY, NAME, sorted(self.ids)))
        self.assertEqual((tol["events"], tol["failures"]), (3, 9)); self.assertIn("inferred_not_proven", tol["uncertainty"])
        h = tol["evidence_sha256"]
        self.assertEqual(h["calls_log"], chunking.file_sha256(self.calls)); self.assertEqual(h["slurm_log"], chunking.file_sha256(self.slurm_log))
        self.assertEqual(h["generation_events.jsonl"], chunking.file_sha256(self.events_file))
        self.assertEqual(h["generation_failures.jsonl"], chunking.file_sha256(self.failures_file))
        self.assertEqual(h["manifest"], chunking.file_sha256(os.path.join(self.chunk, "generation_manifest.json")))
        self.assertEqual(h["state"], chunking.load_state(self.chunk)["self_sha256"])
        self.assertTrue(tol["no_example_or_sidecar_for_failed_attempts"] and tol["failed_attempts_remain_in_ledger"])
        self.assertEqual(seal["checker"]["tolerated"]["historical"]["ids"], sorted(self.ids))
        chunking.load_seal(self.chunk, chunking.load_state(self.chunk))


class ChainContinuationTest(HistBase):
    REAL_IDS = False

    def test_parent_chain_validation_resolve_and_the_next_chunk_work_on_the_recovered_chunk(self):
        self.assertEqual(self.recover()["code"], 0)
        res = cr.validate_chunk(self.chunk, self.anchor_file, self.reserved)                 # strict API, tolerance from the marker
        self.assertEqual(res["code"], 0, res); self.assertEqual(res["tolerated"]["historical"]["failures"], 9)
        code, name, _, _ = cr.resolve_chunk(self.root, 1, self.anchor_file, self.reserved, self.CHUNKS, {t: 12 for t in TASKS})
        self.assertEqual((code, name), (0, NAME))
        c2 = self.sealed(2, 222, 0, NAME)
        res2 = cr.validate_chunk(os.path.join(self.root, c2), self.anchor_file, self.reserved)
        self.assertEqual(res2["code"], 0, res2); self.assertEqual(res2["chain"], [NAME, c2])
        self.assertEqual(res2["tolerated"], {})                                               # the next chunk is judged strictly

    def test_the_historical_attempts_stay_counted_as_failures_in_the_saved_ledger(self):
        self.assertEqual(self.recover()["code"], 0)
        ledger = chunking.load_state(self.chunk)["ledger"]
        for i in self.ids:
            row = ledger[i.rpartition("__a")[0]]
            self.assertGreaterEqual(row["failed_attempts"], 1); self.assertGreaterEqual(row["attempts"], row["failed_attempts"])

    def test_cli(self):
        argv = ["recover-seal", self.chunk, "--anchor-file", self.anchor_file, "--reserved-file", self.reserved, "--calls-log", self.calls,
                "--slurm-log", self.slurm_log, "--config-file", self.cfg_path, "--require-context-guard", "8192", "--confirm-recovery", NAME]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cr.main(argv), 3)                                                # no authorization: strict
            self.assertEqual(cr.main(argv + ["--authorize-historical-timeouts", POLICY]), 3)  # ids are mandatory too
            self.assertEqual(cr.main(argv + ["--authorize-historical-timeouts", POLICY, "--historical-timeout-ids", *self.ids[:2]]), 3)
            self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)))
            self.assertEqual(cr.main(argv + ["--authorize-historical-timeouts", POLICY, "--historical-timeout-ids", *self.ids]), 0)
        self.assertIn("[tolerated]", out.getvalue())


class FailsClosedTest(HistBase):
    def test_an_additional_or_different_incident_blocks_recovery(self):
        extra_event = {"split": "train", "schema_id": "schemas__other__a1", "kind": "provider_error", "detail": "TimeoutError: timed out"}
        extra_fail = {"task": "create", "mode": "standard", "schema_id": "schemas__other__a1", "stage": "worker_error", "detail": "timed out"}
        cases = {"fourth event": [(self.events_file, [extra_event])],
                 "fourth failure": [(self.failures_file, [extra_fail])],
                 "second event for a listed id": [(self.events_file, [{**extra_event, "schema_id": self.ids[0]}])],
                 "other model error on a listed id": [(self.events_file, [{**extra_event, "schema_id": self.ids[1], "detail": "APIConnectionError: x"}])],
                 "token_count_error event": [(self.events_file, [{**extra_event, "kind": "token_count_error"}])],
                 "another technical failure": [(self.failures_file, [{**extra_fail, "schema_id": self.ids[0], "detail": "KeyError: 'x'"}])]}
        for why, edits in cases.items():
            saved = {p: read(p) for p, _ in edits}
            for path, rows in edits:
                self.append(path, rows)
            self.assert_refused(why)
            for p, text in saved.items():
                open(p, "w").write(text)
        self.assertEqual(self.recover()["code"], 0)

    def test_a_missing_incident_blocks_recovery(self):
        rows = jsonl(self.events_file); keep = [r for r in rows if r["schema_id"] != self.ids[2]]
        original = read(self.events_file)
        open(self.events_file, "w").write("".join(json.dumps(r) + "\n" for r in keep)); self.assert_refused("only two events")
        open(self.events_file, "w").write(original)
        original = read(self.failures_file); keep = [r for r in jsonl(self.failures_file) if not (r["schema_id"] == self.ids[0] and r["task"] == "fix")]
        open(self.failures_file, "w").write("".join(json.dumps(r) + "\n" for r in keep)); self.assert_refused("eight failures")
        open(self.failures_file, "w").write(original); self.assertEqual(self.recover()["code"], 0)

    def test_altered_or_missing_original_evidence_blocks_recovery(self):
        original = read(self.calls); lines = original.splitlines()
        open(self.calls, "w").write("\n".join(lines[:-1]) + "\n"); self.assert_refused("calls log shortened")
        rows = [json.loads(l) for l in lines]; rows[0]["error"] = "TimeoutError: timed out"
        open(self.calls, "w").write("\n".join(json.dumps(r) for r in rows) + "\n"); self.assert_refused("a failed model call")
        open(self.calls, "w").write(original)
        self.assert_refused("no slurm log", slurm_log=os.path.join(self.tmp.name, "gone.out"))
        self.write_slurm_log(("build_benchmark.py exit code: 1",)); self.assert_refused("exit code 1"); self.write_slurm_log()
        self.set_manifest(lambda m: m["context_guard"].update(checked=self.n_calls + 1))
        res = self.assert_refused("guard evidence inconsistent"); self.assertTrue(any("context-guard evidence" in t for t in res["technical"]))
        self.set_manifest(lambda m: m["context_guard"].update(checked=self.n_calls))
        self.set_manifest(lambda m: m.update(status="running")); self.assert_refused("manifest not completed")
        self.set_manifest(lambda m: m.update(status="completed"))
        self.assertEqual(self.recover()["code"], 0)

    def test_a_failed_attempt_with_an_example_or_sidecar_blocks_recovery(self):
        sidecar = os.path.join(self.chunk, "benchmark_create", "_metadata", "train", self.ids[1] + ".json")
        os.makedirs(os.path.dirname(sidecar), exist_ok=True); open(sidecar, "w").write("{}")
        res = self.assert_refused("sidecar for a failed attempt"); self.assertTrue(any("although attempt" in t for t in res["technical"]))
        os.remove(sidecar)
        example = os.path.join(self.chunk, "benchmark_fix", "train", self.ids[0] + ".json"); open(example, "w").write("{}")
        self.assert_refused("example for a failed attempt"); os.remove(example)
        self.assertEqual(self.recover()["code"], 0)

    def test_the_ledger_must_still_count_the_failed_attempts(self):
        sid = self.ids[0].rpartition("__a")[0]
        self.set_state(lambda s: s["ledger"][sid].update(failed_attempts=0)); self.assert_refused("failed attempt not counted")
        self.set_state(lambda s: s["ledger"][sid].update(failed_attempts=1, attempts=0)); self.assert_refused("attempt not counted")
        self.set_state(lambda s: s["ledger"].pop(sid)); self.assert_refused("no ledger row")
        self.set_state(lambda s: s["ledger"].setdefault(sid, {}).update(attempts=1, failed_attempts=1, accepted_references=0))
        self.assertEqual(self.recover()["code"], 0)

    def test_a_tampered_marker_or_a_late_extra_incident_invalidates_the_chunk(self):
        self.assertEqual(self.recover()["code"], 0)
        marker = os.path.join(self.chunk, chunking.SEAL_FILE); good = read(marker)
        for edit in (lambda s: s["recovery"]["historical_tolerance"].update(ids=self.ids[:2]),
                     lambda s: s["recovery"]["historical_tolerance"].update(policy="any-timeout")):
            seal = json.loads(good); edit(seal); seal.pop("self_sha256"); seal["self_sha256"] = chunking.canonical_sha256(seal)
            open(marker, "w").write(json.dumps(seal))                                         # even a correctly re-hashed forgery
            res = cr.validate_chunk(self.chunk, self.anchor_file, self.reserved)
            self.assertEqual(res["code"], 3, res)
        seal = json.loads(good); seal["recovery"]["historical_tolerance"]["ids"] = self.ids[:2]; open(marker, "w").write(json.dumps(seal))
        self.assertEqual(cr.validate_chunk(self.chunk, self.anchor_file, self.reserved)["code"], 3)   # hash mismatch
        open(marker, "w").write(good); self.assertEqual(cr.validate_chunk(self.chunk, self.anchor_file, self.reserved)["code"], 0)
        self.append(self.events_file, [{"split": "train", "schema_id": "schemas__other__a1", "kind": "provider_error",
                                        "detail": "TimeoutError: timed out"}])
        self.assertEqual(cr.validate_chunk(self.chunk, self.anchor_file, self.reserved)["code"], 3)   # tolerance never grows

    def test_a_marker_naming_the_policy_in_another_chunk_grants_nothing(self):
        other = self.run_chunk(1, "c01_99999-r1", seal=False)
        for p in (os.path.join(self.chunk, "generation_manifest.json"),):
            pass
        state = chunking.load_state(other)
        chunking.write_seal(other, state, {"code": 0, "verdict": "SUCCESS", "counts": state["counts"], "warnings": 0},
                            {"checked": True, "file": "x", "sha256": "0" * 64, "n_calls": 1, "n_failed": 0}, 0, 8192,
                            recovery={"historical_tolerance": {"policy": POLICY, "ids": sorted(self.ids)}})
        res = cr.validate_chunk(other, self.anchor_file, self.reserved)
        self.assertTrue(any("inapplicable" in t for t in res["technical"]), res)


if __name__ == "__main__":
    unittest.main()
