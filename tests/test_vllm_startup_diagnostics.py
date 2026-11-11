"""CPU-only tests of the vLLM startup diagnostics and readiness loop of slurm/full_chunk_granite_v2.sbatch.
The real code is cut out of the script and run in bash with a fake `curl` and fake processes: no GPU, no network."""
import os, re, stat, subprocess, sys, tempfile, time, unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, "slurm", "full_chunk_granite_v2.sbatch")
SB = Path(FULL).read_text()
KEY = "KEYSECRET-abcdef0123456789"


def cut(start, end):
    a = SB.index(start); return SB[a:SB.index(end, a)]


FUNCTIONS = cut("# BEGIN vllm-diagnostics", "# END vllm-diagnostics")
LOOP = cut("READY=0; PROBES=0", "[ \"$READY\" -eq 1 ] ||")


class StaticTest(unittest.TestCase):
    def test_server_is_unbuffered_logs_to_the_same_file_and_keeps_every_model_parameter(self):
        launch = cut("PYTHONUNBUFFERED=1 setsid", "VLLM_PID=$!")
        for needle in ['"$VLLM_BIN" serve "$MODEL_PATH"', '--served-model-name "$GRANITE_MODEL_ID"', "--host 127.0.0.1", '--port "$SERVER_PORT"',
                       "--tensor-parallel-size 1", '--max-model-len "$MAX_MODEL_LEN"', "--gpu-memory-utilization 0.90", '> "$VLLM_LOG" 2>&1 &']:
            self.assertIn(needle, launch)
        self.assertEqual(launch.count("--"), 6)                       # no flag was added or removed
        self.assertIn('VLLM_LOG="logs/full-vllm-${RUN_ID}.log"', SB)
        self.assertNotIn("PYTHONUNBUFFERED", SB.replace("PYTHONUNBUFFERED=1 setsid", "").replace("# PYTHONUNBUFFERED applies", ""))  # not exported globally

    def test_deadline_is_the_old_one_and_endpoint_and_auth_are_unchanged(self):
        self.assertIn("READY_TIMEOUT_SEC=1800", SB)                    # 180 polls x 10 s, as before: not increased
        self.assertIn('"$VLLM_BASE_URL/models"', LOOP); self.assertIn("curl -qsf --noproxy 127.0.0.1", LOOP)
        self.assertIn("--connect-timeout 3 --max-time 10", LOOP)
        self.assertIn("header = \"Authorization: Bearer %s\"", LOOP)   # the key goes through stdin, never an argument
        self.assertNotIn("set -x", SB); self.assertNotIn("/proc/$pid/cmdline", FUNCTIONS); self.assertNotIn("environ", FUNCTIONS)
        self.assertNotIn("args", re.sub(r"nlwp=,comm=", "", re.findall(r"ps -o [^\n]*", FUNCTIONS)[0]))

    def test_cleanup_and_scratch_handling_are_untouched(self):
        for needle in ["umask 077", "kill -TERM --", "-O \"$JOB_SCRATCH\"", "SCRATCH_INODE", "trap cleanup EXIT"]:
            self.assertIn(needle, SB)
        self.assertEqual(SB.count("rm -rf"), 1)


def run_bash(script, env=None, timeout=60):
    e = {"PATH": os.environ["PATH"], "VLLM_API_KEY": KEY, **(env or {})}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=timeout)


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "vllm.log")

    def test_redaction_removes_the_key_and_bearer_tokens_and_bounds_line_length(self):
        text = f"Authorization: Bearer {KEY}\nplain {KEY} twice {KEY}\nBearer other.token-123\n" + "x" * 1000 + "\n"
        r = subprocess.run(
            ["bash", "-c", FUNCTIONS + "\nvllm_redact"], input=text, capture_output=True, text=True, env={"PATH": os.environ["PATH"], "VLLM_API_KEY": KEY})
        self.assertNotIn(KEY, r.stdout); self.assertNotIn("other.token-123", r.stdout); self.assertIn("[REDACTED]", r.stdout)
        self.assertTrue(all(len(l) <= 300 for l in r.stdout.splitlines()))

    def test_a_zombie_is_not_alive_although_kill_zero_says_so(self):
        zombie = subprocess.Popen(["true"]); time.sleep(0.5)            # exited, deliberately not reaped
        live = subprocess.Popen(["sleep", "30"]); self.addCleanup(live.wait); self.addCleanup(live.kill)
        try:
            r = run_bash(f'{FUNCTIONS}\nkill -0 {zombie.pid} && echo killzero-ok\n'
                         f'echo "zombie_state=$(vllm_proc_state {zombie.pid})"; vllm_alive {zombie.pid} && echo zombie-alive || echo zombie-dead\n'
                         f'echo "live_state=$(vllm_proc_state {live.pid})"; vllm_alive {live.pid} && echo live-alive || echo live-dead\n'
                         'echo "gone_state=$(vllm_proc_state 999999)"; vllm_alive 999999 && echo gone-alive || echo gone-dead')
        finally:
            zombie.wait()
        out = r.stdout
        self.assertIn("killzero-ok", out); self.assertIn("zombie_state=Z", out); self.assertIn("zombie-dead", out)
        self.assertIn("live-alive", out); self.assertIn("gone_state=gone", out); self.assertIn("gone-dead", out)

    def diag(self, log_text, secret_arg="SECRET-CMDLINE-ARG-777"):
        if log_text is not None:
            open(self.log, "w").write(log_text)
        live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", secret_arg]); self.addCleanup(live.wait); self.addCleanup(live.kill)
        r = run_bash(f'VLLM_PID={live.pid}; VLLM_LOG={self.log}; READY_START=$SECONDS; PROBES=7; CURL_RC=28\n{FUNCTIONS}\nvllm_diag "test reason"')
        return r.stdout + r.stderr

    def test_diagnostics_are_bounded_redacted_and_never_show_a_command_line_or_the_environment(self):
        lines = [f"line {i}" for i in range(500)] + [f"token {KEY}", "final line"]
        out = self.diag("\n".join(lines) + "\n")
        self.assertIn("test reason", out); self.assertIn("probes=7 last_curl_rc=28", out); self.assertIn("final line", out)
        self.assertNotIn("line 0\n", out); self.assertNotIn(KEY, out); self.assertNotIn("SECRET-CMDLINE-ARG-777", out)
        self.assertLessEqual(len(out.splitlines()), 90); self.assertLess(len(out), 12000)
        self.assertRegex(out, r"state=[SR]"); self.assertIn("vLLM log:", out)

    def test_an_empty_or_missing_log_is_reported_as_such(self):
        self.assertIn("vLLM log: 0 bytes", self.diag(""))
        os.remove(self.log); self.assertIn("vLLM log missing", self.diag(None))

    def test_a_huge_log_costs_a_bounded_tail(self):
        out = self.diag("a" * 5_000_000 + "\nend\n")
        self.assertLess(len(out), 12000); self.assertIn("5000005 bytes", out)


class ReadinessLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.bin = os.path.join(self.tmp.name, "bin"); os.makedirs(self.bin)
        self.log = os.path.join(self.tmp.name, "vllm.log"); open(self.log, "w").write("starting\n")
        self.record = os.path.join(self.tmp.name, "curl_args")

    def fake_curl(self, rc, sleep=0):
        path = os.path.join(self.bin, "curl")
        Path(path).write_text(f'#!/bin/bash\necho "$@" >> "{self.record}"\ncat > /dev/null\nsleep {sleep}\nexit {rc}\n')
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    def run_loop(self, server, timeout_sec=3, poll=1, report=1):
        script = (f'export PATH="{self.bin}:$PATH"\nVLLM_BASE_URL=http://127.0.0.1:1/v1; VLLM_LOG={self.log}\n'
                  f'READY_TIMEOUT_SEC={timeout_sec}; READY_POLL_SEC={poll}; READY_REPORT_EVERY_SEC={report}\n{FUNCTIONS}\n{server}\n{LOOP}\n'
                  'echo LOOP-FINISHED')
        t0 = time.time(); r = run_bash(script, env={"HOME": self.tmp.name}); return r, time.time() - t0

    def test_ready_server_ends_the_loop_without_diagnostics(self):
        self.fake_curl(0)
        r, _ = self.run_loop('sleep 30 >/dev/null 2>&1 & VLLM_PID=$!')
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("vLLM ready after", r.stdout); self.assertNotIn("diagnostics", r.stdout)
        args = open(self.record).read()
        self.assertIn("--noproxy 127.0.0.1", args); self.assertIn("--connect-timeout 3 --max-time 10", args); self.assertIn("-K -", args)
        self.assertNotIn(KEY, args + r.stdout + r.stderr)                 # the key reaches curl only through stdin

    def test_alive_but_not_ready_hits_the_deadline_and_is_reported_as_alive(self):
        self.fake_curl(7)
        r, elapsed = self.run_loop('sleep 30 >/dev/null 2>&1 & VLLM_PID=$!', timeout_sec=3)
        self.assertEqual(r.returncode, 11); self.assertNotIn("LOOP-FINISHED", r.stdout)
        self.assertIn("still ALIVE but not answering", r.stdout); self.assertIn("alive but not ready", r.stdout)
        self.assertIn("vLLM waiting:", r.stdout); self.assertIn("last_curl_rc=7", r.stdout); self.assertNotIn(KEY, r.stdout + r.stderr)
        self.assertLess(elapsed, 3 + 1 + 3)

    def test_slow_probes_cannot_push_the_loop_far_past_its_deadline(self):
        self.fake_curl(28, sleep=2)                                       # every probe runs into its --max-time
        r, elapsed = self.run_loop('sleep 30 >/dev/null 2>&1 & VLLM_PID=$!', timeout_sec=4, poll=1)
        self.assertEqual(r.returncode, 11); self.assertIn("did not become ready within 4s", r.stdout)
        self.assertLess(elapsed, 4 + 2 + 1 + 2)                           # deadline + at most one probe and one poll
        # the old loop (180 x (max-time 10 s + sleep 10 s)) could take up to 3600 s; the new worst case is 1800 s + one probe + one poll

    def test_an_exited_server_is_reported_as_exited_with_its_status(self):
        self.fake_curl(7)
        r, elapsed = self.run_loop('sh -c "echo boom; exit 7" > /dev/null & VLLM_PID=$!; sleep 0.5', timeout_sec=30)
        self.assertEqual(r.returncode, 11); self.assertIn("EXITED before it became ready (wait status 7", r.stdout)
        self.assertIn("server process exited", r.stdout); self.assertNotIn("ALIVE", r.stdout); self.assertLess(elapsed, 10)

    def test_a_finished_server_counts_as_exited(self):
        self.fake_curl(7)
        r, _ = self.run_loop('sleep 0.2 & VLLM_PID=$!; sleep 0.6', timeout_sec=30)   # exited, as far as the loop can tell
        self.assertEqual(r.returncode, 11); self.assertIn("EXITED", r.stdout)

    def test_a_missing_log_does_not_break_the_report(self):
        self.fake_curl(7); os.remove(self.log)
        r, _ = self.run_loop('sleep 30 >/dev/null 2>&1 & VLLM_PID=$!', timeout_sec=2)
        self.assertEqual(r.returncode, 11); self.assertIn("log_bytes=0", r.stdout); self.assertIn("vLLM log missing", r.stdout)


if __name__ == "__main__":
    unittest.main()
