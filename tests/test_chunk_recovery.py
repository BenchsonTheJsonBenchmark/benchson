"""Regression tests for the two integration failures of the first full chunk (job 24044333) and for the one-time, fail-closed
recovery of a chunk that finished generating but could not be sealed."""
import contextlib, io, json, os, re, shutil, subprocess, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
import chunked_run as cr
import pilot_report
from data_generation import chunking
from data_generation.published import TASKS
from test_chunk_chain import ChainBase, read
from test_chunking import tree_hash

JOB = "24044333"
NAME = f"c01_{JOB}-r1"


class Regression(ChainBase):
    def test_checker_handles_per_task_quota_dicts_and_keeps_the_int_behaviour(self):
        out = self.run_chunk(1, "c01", seal=False)
        quota = chunking.load_state(out)["quota"]
        self.assertEqual(csr.check(out, per_task=quota)["code"], 0)                       # the failing call of job 24044333
        self.assertEqual(csr.check(out, per_task=quota["create"])["code"], 0)              # Smoke/Pilot style: one int for every task
        res = csr.check(out, per_task={**quota, "fix": quota["fix"] - 1})
        self.assertEqual(res["code"], 4); self.assertTrue(any("more than the expected" in i for i in res["invalid"]))
        res = csr.check(out, per_task={**quota, "modify": quota["modify"] + 1})
        self.assertEqual(res["code"], 2); self.assertTrue(any("modify" in s for s in res["shortfall"]))
        self.assertEqual(csr.check(out, per_task=dict(quota))["counts"], {t: quota[t] for t in TASKS})

    def test_seal_path_with_quota_dict_no_longer_crashes(self):
        out = self.run_chunk(1, "c01", seal=False)
        res = cr.seal_chunk(out, self.anchor_file, self.reserved, self.calls_log, 0)
        self.assertEqual(res["code"], 0, res); self.assertTrue(res["sealed"])

    def test_every_option_the_scripts_pass_to_the_tools_is_accepted_by_the_tools(self):
        def help_text(*argv):
            return subprocess.run([sys.executable, "-W", "ignore", *argv, "--help"], capture_output=True, text=True, cwd=ROOT).stdout
        report_help = help_text("src/pilot_report.py")
        sb = read(os.path.join(ROOT, "slurm", "full_chunk_granite_v2.sbatch"))
        call = re.search(r'src/pilot_report\.py(.*?)\|\| echo', sb, re.S).group(1)
        flags = set(re.findall(r"(--[a-z-]+)", call)); self.assertIn("--targets", flags)
        for flag in flags:
            self.assertIn(flag, report_help, f"pilot_report.py does not accept {flag} (the second failure of job 24044333)")
        for script in ("full_chunk_granite_v2.sbatch", "full_chunk_chain_job.sbatch", "submit_chunk_chain.sh"):
            text = read(os.path.join(ROOT, "slurm", script))
            for sub in set(re.findall(r'src/chunked_run\.py\s+(\w[\w-]*)', text)):
                self.assertIn(sub, help_text("src/chunked_run.py"), f"{script} uses unknown chunked_run subcommand {sub}")
        for sub, flags in (("seal", ("--anchor-file", "--reserved-file", "--calls-log", "--build-exit-code", "--require-context-guard")),
                           ("resolve", ("--root", "--index", "--anchor-file", "--reserved-file", "--chunks", "--totals", "--pin"))):
            for flag in flags:
                self.assertIn(flag, help_text("src/chunked_run.py", sub))

    def test_selfcheck_passes_here_and_catches_stale_tools(self):
        self.assertEqual(cr.selfcheck(), [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cr.main(["selfcheck"]), 0)
        original = csr.check
        def stale(*a, **k):
            raise TypeError("'>' not supported between instances of 'int' and 'dict'")
        csr.check = stale
        try:
            problems = cr.selfcheck()
            self.assertTrue(problems and "per-task quotas" in problems[0])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cr.main(["selfcheck"]), 3)
        finally:
            csr.check = original
        original_main = pilot_report.main
        def stale_report(argv=None):
            raise SystemExit(2)                       # argparse: unrecognized arguments: --targets
        pilot_report.main = stale_report
        try:
            self.assertTrue(any("pilot_report.py" in p for p in cr.selfcheck()))
        finally:
            pilot_report.main = original_main

    def test_chunk_script_runs_the_selfcheck_before_any_gpu_work_and_logs_the_fingerprint(self):
        sb = read(os.path.join(ROOT, "slurm", "full_chunk_granite_v2.sbatch"))
        self.assertLess(sb.index("chunked_run.py selfcheck"), sb.index("nvidia-smi"))
        self.assertLess(sb.index("chunked_run.py selfcheck"), sb.index("POOL_REPORT="))
        self.assertIn("chunked_run.py fingerprint", sb)
        fp = cr.fingerprint()
        self.assertTrue(all(fp.values()), [k for k, v in fp.items() if not v])           # every runtime file exists here
        out = subprocess.run([sys.executable, "-W", "ignore", "src/chunked_run.py", "fingerprint"], capture_output=True, text=True, cwd=ROOT).stdout
        self.assertEqual(len(out.strip().splitlines()), len(cr.RUNTIME_FILES))


class RecoveryBase(ChainBase):
    """An unsealed chunk exactly as job 24044333 left it: finished build, state written, original calls log, Slurm log."""

    def setUp(self):
        super().setUp()
        os.environ["SLURM_JOB_ID"] = JOB; self.addCleanup(os.environ.pop, "SLURM_JOB_ID", None)
        self.full = os.path.join(self.co, "data_scale_only", "full"); os.makedirs(self.full, exist_ok=True)
        self.chunk = self.run_chunk(1, NAME, seal=False)
        self.calls = os.path.join(self.full, f"{NAME}_calls.jsonl"); shutil.copy(self.calls_log, self.calls)
        self.slurm_log = os.path.join(self.tmp.name, "benchson-full-24044333.out")
        self.write_slurm_log()
        mf = os.path.join(self.chunk, "generation_manifest.json"); m = json.load(open(mf))
        m["context_guard"] = {"enabled": True, "max_model_len": 8192, "checked": 120, "overflow": 0}
        json.dump(m, open(mf, "w"))
        self.manifest = m

    def write_slurm_log(self, r1_lines=("build_benchmark.py exit code: 0 (2 = chunk quota not met, no checkpoint; 124 = generation wall-clock limit)",)):
        lines = [f"=== Preflight (job) job={JOB} attempt=0 run={JOB}-r0 ===", "preempted before the build ended",
                 f"=== Preflight (job) job={JOB} attempt=1 run={JOB}-r1 ===", "model ready", *r1_lines,
                 "=== Validating chunk 1 and the chain before it ===", "TypeError: '>' not supported between instances of 'int' and 'dict'"]
        open(self.slurm_log, "w").write("\n".join(lines) + "\n")

    def recover(self, **over):
        args = dict(chunk_dir=self.chunk, anchor_file=self.anchor_file, reserved_file=self.reserved, calls_log=self.calls,
                    slurm_log=self.slurm_log, config_file=self.cfg_path, require_context_guard=8192, confirm=NAME)
        args.update(over)
        return cr.recover_seal(args["chunk_dir"], args["anchor_file"], args["reserved_file"], args["calls_log"], args["slurm_log"],
                               args["config_file"], args["require_context_guard"], args["confirm"])

    def assert_refused(self, why, **over):
        before = tree_hash(self.chunk)
        res = self.recover(**over)
        self.assertNotEqual(res["code"], 0, why); self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)), why)
        self.assertEqual(before, tree_hash(self.chunk), why)
        return res


class RecoveryTest(RecoveryBase):
    def test_recovers_without_touching_any_existing_file_and_the_chain_continues(self):
        before = {k: tree_hash(p) if os.path.isdir(p) else read(p) for k, p in (("chunk", self.chunk), ("calls", self.calls), ("slurm", self.slurm_log))}
        state_before = read(os.path.join(self.chunk, chunking.STATE_FILE))
        res = self.recover()
        self.assertEqual(res["code"], 0, res); self.assertTrue(res["sealed"] and res["recovered"])
        self.assertEqual(read(self.calls), before["calls"]); self.assertEqual(read(self.slurm_log), before["slurm"])
        self.assertEqual(read(os.path.join(self.chunk, chunking.STATE_FILE)), state_before)
        seal = json.load(open(os.path.join(self.chunk, chunking.SEAL_FILE)))
        self.assertEqual(seal["calls_log"]["sha256"], chunking.file_sha256(self.calls)); self.assertEqual(seal["calls_log"]["n_failed"], 0)
        self.assertEqual(seal["recovery"]["build_exit_code_lines"], ["0"]); self.assertEqual(seal["recovery"]["slurm_log_sha256"], chunking.file_sha256(self.slurm_log))
        self.assertEqual(seal["context_guard_required"], 8192); self.assertEqual(seal["checker"]["verdict"], "SUCCESS")
        chunking.load_seal(self.chunk, chunking.load_state(self.chunk))                   # verified like any other marker
        code, name, _, _ = cr.resolve_chunk(self.root, 1, self.anchor_file, self.reserved, self.CHUNKS, {t: 12 for t in TASKS})
        self.assertEqual((code, name), (0, NAME))
        r = self.chain_job(2, 222)                                                        # the next link finds it by itself
        self.assertEqual(r.returncode, 0, r.stderr[-600:]); self.assertIn(f"PARENT_CHUNK={NAME}", self.inner_calls()[0])
        c2 = self.sealed(2, 222, 0, NAME)                                                 # and the next chunk really builds on it
        self.assertEqual(cr.validate_chunk(os.path.join(self.root, c2), self.anchor_file)["code"], 0)

    def test_recovery_is_one_time(self):
        self.assertEqual(self.recover()["code"], 0)
        marker = read(os.path.join(self.chunk, chunking.SEAL_FILE))
        res = self.recover(); self.assertEqual(res["code"], 3); self.assertIn("already exists", " ".join(res["technical"]))
        self.assertEqual(marker, read(os.path.join(self.chunk, chunking.SEAL_FILE)))

    def test_cli(self):
        argv = ["recover-seal", self.chunk, "--anchor-file", self.anchor_file, "--reserved-file", self.reserved, "--calls-log", self.calls,
                "--slurm-log", self.slurm_log, "--config-file", self.cfg_path, "--require-context-guard", "8192"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cr.main(argv + ["--confirm-recovery", "wrong"]), 3)
            self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)))
            self.assertEqual(cr.main(argv + ["--confirm-recovery", NAME]), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):     # every evidence option is mandatory
            cr.main(argv[:argv.index("--slurm-log")] + ["--confirm-recovery", NAME])


class RecoveryFailsClosedTest(RecoveryBase):
    def test_missing_or_wrong_evidence_never_produces_a_marker(self):
        self.assert_refused("confirmation", confirm="c01_24044333-r0")
        self.assert_refused("no calls log", calls_log=os.path.join(self.full, "missing.jsonl"))
        other = os.path.join(self.full, "c01_24044333-r0_calls.jsonl"); shutil.copy(self.calls, other)
        self.assert_refused("calls log of another attempt", calls_log=other)
        self.assert_refused("no slurm log", slurm_log=os.path.join(self.tmp.name, "nope.out"))
        self.assert_refused("no config", config_file=os.path.join(self.tmp.name, "nope.json"))
        changed = os.path.join(self.tmp.name, "cfg2.json"); open(changed, "w").write(read(self.cfg_path) + " ")
        self.assert_refused("config differs from the generation config", config_file=changed)
        self.assert_refused("guard window differs", require_context_guard=16384)
        self.assert_refused("not a chunk dir name", chunk_dir=os.path.join(self.tmp.name, "somewhere"), confirm="somewhere")

    def test_slurm_log_must_show_exactly_one_zero_exit_for_this_attempt(self):
        for lines in ((), ("build_benchmark.py exit code: 2 (2 = chunk quota not met)",), ("build_benchmark.py exit code: 1",),
                      ("build_benchmark.py exit code: 0", "build_benchmark.py exit code: 0")):
            self.write_slurm_log(lines); self.assert_refused(f"exit lines {lines}")
        open(self.slurm_log, "w").write(f"=== Preflight (job) job={JOB} attempt=0 run={JOB}-r0 ===\nbuild_benchmark.py exit code: 0\n")
        self.assert_refused("only the OTHER attempt (r0) reports exit 0")
        open(self.slurm_log, "w").write("")
        self.assert_refused("empty slurm log")
        self.write_slurm_log(); self.assertEqual(self.recover()["code"], 0)             # the intact evidence works

    def test_calls_log_must_be_the_original_complete_and_error_free(self):
        lines = read(self.calls).splitlines()
        original = read(self.calls)
        rows = [json.loads(l) for l in lines]
        rows[0]["error"] = "Maximum context length is 8192 tokens"
        open(self.calls, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        res = self.assert_refused("a failed call"); self.assertTrue(any("model calls failed" in t for t in res["technical"]))
        open(self.calls, "w").write("\n".join(lines[:-1]) + "\n"); self.assert_refused("a line missing")
        open(self.calls, "w").write(original + lines[-1] + "\n"); self.assert_refused("a line added")
        open(self.calls, "w").write(""); self.assert_refused("empty")
        open(self.calls, "w").write(original); self.assertEqual(self.recover()["code"], 0)

    def test_manifest_and_state_evidence(self):
        mf = os.path.join(self.chunk, "generation_manifest.json"); good = json.load(open(mf))
        for key, value in (("status", "stopped: KeyboardInterrupt"), ("slurm_job_id", "999"), ("finished", None), ("llm_calls_used", 1),
                           ("context_guard", {"enabled": False})):
            m = dict(good); m[key] = value; json.dump(m, open(mf, "w")); self.assert_refused(key)
        m = dict(good); m["context_guard"] = dict(good["context_guard"], checked=0); json.dump(m, open(mf, "w")); self.assert_refused("guard counted nothing")
        json.dump(good, open(mf, "w"))
        interrupted = self.preempted(1, 24044333, 5)                                       # no state file at all
        self.assert_refused("no checkpoint", chunk_dir=os.path.join(self.root, interrupted), confirm=interrupted)

    def test_any_modified_added_or_removed_example_or_state_blocks_recovery(self):
        victim = os.path.join(self.chunk, "benchmark_fix", "train"); f = sorted(os.listdir(victim))[0]
        p = os.path.join(victim, f); original = read(p)
        for edit in (lambda: open(p, "w").write(original.replace("{", "{ ", 1)), lambda: shutil.copy(p, os.path.join(victim, "extra__a9.json")),
                     lambda: os.remove(p)):
            edit(); res = self.recover(); self.assertEqual(res["code"], 3)
            self.assertFalse(os.path.exists(os.path.join(self.chunk, chunking.SEAL_FILE)))
            for extra in os.listdir(victim):
                if extra.startswith("extra"):
                    os.remove(os.path.join(victim, extra))
            open(p, "w").write(original)
        sp = os.path.join(self.chunk, chunking.STATE_FILE); st = read(sp)
        open(sp, "w").write(st.replace('"seed": 0', '"seed": 7') if '"seed": 0' in st else st + " "); self.assertEqual(self.recover()["code"], 3)
        open(sp, "w").write(st); self.assertEqual(self.recover()["code"], 0)

    def test_an_unsealed_parent_blocks_recovery_of_its_child(self):
        a = self.chunk                                                                      # chunk 1 stays unsealed
        c2 = self.run_chunk_with_unsealed_parent()
        os.environ["SLURM_JOB_ID"] = "555"
        calls2 = os.path.join(self.full, "c02_555-r0_calls.jsonl"); shutil.copy(self.calls_log, calls2)
        open(self.slurm_log, "a").write("=== Preflight (job) job=555 attempt=0 run=555-r0 ===\nbuild_benchmark.py exit code: 0\n")
        res = cr.recover_seal(c2, self.anchor_file, self.reserved, calls2, self.slurm_log, self.cfg_path, 8192, "c02_555-r0")
        self.assertNotEqual(res["code"], 0); self.assertFalse(os.path.exists(os.path.join(c2, chunking.SEAL_FILE)))
        self.assertIsNotNone(a)

    def run_chunk_with_unsealed_parent(self):
        # the builder itself refuses an unsealed parent, so build chunk 2 on a sealed copy of the parent, then remove the marker
        self.assertEqual(self.recover()["code"], 0)
        c2 = self.run_chunk(2, "c02_555-r0", NAME, seal=False)
        os.environ["SLURM_JOB_ID"] = JOB
        os.rename(os.path.join(self.chunk, chunking.SEAL_FILE), os.path.join(self.tmp.name, "parent_seal.json"))
        mf = os.path.join(c2, "generation_manifest.json"); m = json.load(open(mf)); m["slurm_job_id"] = "555"
        m["context_guard"] = {"enabled": True, "max_model_len": 8192, "checked": 5, "overflow": 0}; json.dump(m, open(mf, "w"))
        return c2


if __name__ == "__main__":
    unittest.main()
