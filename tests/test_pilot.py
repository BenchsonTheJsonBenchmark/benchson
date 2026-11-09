"""CPU-only tests for the 60-example pilot (20 Create / 20 Fix / 20 Modify): script, preflight, checker, report,
requeue safety, output isolation, anchor/leakage checks and the resume limitations found in the generator."""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
import pilot_report as pr
import test_smoke_preflight as T
from build_benchmark import main as build_main
from data_generation import anchor as A
from data_generation.generation_mix import split_total
from data_generation.published import TASKS, schema_hash
from test_anchor_pool import _builder_config, write_baseline

PILOT = os.path.join(ROOT, "slurm", "pilot_granite_v2.sbatch")
SMOKE = os.path.join(ROOT, "slurm", "smoke_granite_v2.sbatch")


class PilotScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot, cls.smoke = Path(PILOT).read_text(), Path(SMOKE).read_text()

    def test_valid_bash_and_smoke_script_is_independent(self):
        for path in (PILOT, SMOKE):
            self.assertEqual(subprocess.run(["bash", "-n", path], capture_output=True).returncode, 0)
        self.assertIn("PER_TASK=2\n", self.smoke)           # smoke script keeps its own 2/2/2 target
        self.assertNotIn("pilot", self.smoke.lower())

    def test_exactly_twenty_per_task(self):
        self.assertIn("PILOT_TOTAL=60", self.pilot); self.assertIn("PER_TASK=20", self.pilot)
        self.assertEqual(split_total(60, ["create", "fix", "modify"]), {"create": 20, "fix": 20, "modify": 20})
        self.assertIn('--smoke-total "$PILOT_TOTAL"', self.pilot)
        self.assertIn("--require-smoke-total", self.pilot)
        self.assertIn("--per-task \"$PER_TASK\"", self.pilot)

    def test_scientific_and_safety_settings_are_identical_to_smoke(self):
        for needle in ['CONFIG="configs/scale_only_vllm_granite.json"', "GRANITE_REVISION:-b8c0982bab7fde4eb48110f5a069527c008fab39",
                       "GRANITE_MODEL_ID:-ibm-granite/granite-4.0-h-small", "PUBLISHED_REVISION=7962b0947f7d96eb3a0e6b9b928d939c7f3061a6",
                       "ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6", "--smoke-openai-timeout 180",
                       "--max-model-len 8192", "--gpu-memory-utilization 0.90", "--tensor-parallel-size 1", "--gpus=rtx_pro_6000:1",
                       "umask 077", "SLURM_RESTART_COUNT", "--open-mode=append", "secrets.token_urlsafe", "--host 127.0.0.1",
                       'RUN_ID="${JOB}-r${ATTEMPT}"', "[[ ! -e \"$path\" && ! -L \"$path\" ]]", "--anchor-file \"$ANCHOR\"",
                       "published_train_schemas.json", "reuse.pool_source=published_train", "kill -TERM --", "curl -qsf --noproxy"]:
            self.assertIn(needle, self.pilot, needle)
        self.assertEqual(self.pilot.count("--config"), self.smoke.count("--config"))
        for forbidden in ("--temperature", "--top-p", "--max-tokens", "standard_ratio", "complex_ratio", "max_retries"):
            self.assertNotIn(forbidden, self.pilot)         # sampling/acceptance settings live only in the config

    def test_operational_limits(self):
        self.assertIn('MAX_LLM_CALLS="${MAX_LLM_CALLS:-600}"', self.pilot)
        self.assertIn("-le 800", self.pilot)
        self.assertIn('GEN_TIMEOUT_SEC="${GEN_TIMEOUT_SEC:-3600}"', self.pilot)
        self.assertIn("-le 4800", self.pilot)
        self.assertIn("--time=02:00:00", self.pilot)
        self.assertIn("timeout --kill-after=60", self.pilot)

    def test_every_output_path_is_pilot_specific(self):
        for needle in ['RUN_NAME="pilot_${RUN_ID}"', "logs/pilot-vllm-${RUN_ID}.log", "logs/pilot-pool-check-${RUN_ID}.json",
                       "benchson-pilot-$RUN_ID.XXXXXXXX", "benchson-pilot-%j.out", "${RUN_NAME}_report.json"]:
            self.assertIn(needle, self.pilot, needle)
        for stale in ("logs/vllm-", "logs/pool-check-", "benchson-smoke-", 'smoke_${RUN_ID}'):
            self.assertNotIn(stale, self.pilot)
        self.assertIn('"$REPORT_JSON"', self.pilot.split("for path in")[1].split("done")[0])   # report path checked before the run

    def test_report_does_not_change_the_verdict(self):
        tail = self.pilot.split("pilot_report.py")[1]
        self.assertIn("|| echo", tail); self.assertTrue(tail.rstrip().endswith('exit "$CHECK_RC"'))
        self.assertIn("GEN_START=$(date +%s)", self.pilot); self.assertIn("GEN_END=$(date +%s)", self.pilot)
        self.assertLess(self.pilot.index("GEN_START"), self.pilot.index("setsid timeout"))   # after download and vLLM startup
        self.assertGreater(self.pilot.index("download"), 0)
        self.assertGreater(self.pilot.index("GEN_START"), self.pilot.index("Waiting for vLLM"))


class PilotSlurmTest(T.SlurmCleanupTest):
    """The whole cleanup / preflight / requeue suite, run against the pilot script with 20/20/20 targets."""
    SCRIPT = "slurm/pilot_granite_v2.sbatch"
    LOG, POOLLOG, TARGET, SCRATCH = "pilot-vllm", "pilot-pool-check", 20, "benchson-pilot"

    def extra_stubs(self):
        self.stub("timeout", "args = sys.argv[1:]\nassert args[0].startswith('--kill-after=')\nassert args[1].isdigit()\nos.execvp(args[2], args[2:])")

    def test_report_is_written_under_a_pilot_name_and_never_overwritten(self):
        code, output = self.run_job("success")
        self.assertEqual(code, 0, output)
        self.assertTrue((self.root / "data_scale_only/pilot_1234-r0_report.json").exists())
        self.assertFalse(list((self.root / "data_scale_only").glob("smoke_*")))
        code, output = self.run_job("success", SLURM_RESTART_COUNT="1")
        self.assertEqual(code, 0, output)
        self.assertTrue((self.root / "data_scale_only/pilot_1234-r1_report.json").exists())

    def test_call_and_time_limits_are_bounded(self):
        for var, bad in (("MAX_LLM_CALLS", "801"), ("MAX_LLM_CALLS", "0"), ("GEN_TIMEOUT_SEC", "4801"), ("GEN_TIMEOUT_SEC", "abc")):
            code, output = self.run_job("success", **{var: bad})
            self.assertEqual(code, 10, output); self.assertIn(var, output)

    def test_pool_that_cannot_supply_twenty_per_task_is_refused(self):
        py = self.bin / "python"
        py.write_text(py.read_text().replace('"upper_bound_if_every_attempt_succeeds": 20 + 1', '"upper_bound_if_every_attempt_succeeds": 19'))
        code, output = self.run_job("success")
        self.assertEqual(code, 10, output); self.assertIn("pilot requirements", output)

    def test_wrong_target_in_pool_report_is_refused(self):
        py = self.bin / "python"
        py.write_text(py.read_text().replace('"target": 20,', '"target": 2,'))
        code, output = self.run_job("success")
        self.assertEqual(code, 10, output)


class PilotCheckerTest(unittest.TestCase):
    """A real mock-model build of 60 examples on a synthetic anchored baseline (no model, no network)."""

    @classmethod
    def setUpClass(cls):
        src = os.path.join(ROOT, "data", "schemas", "test")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.skip = None
        if not os.path.isdir(src):
            cls.skip = "data/schemas not present"; return
        files = [f for f in sorted(os.listdir(src)) if f.endswith(".json") and os.path.getsize(os.path.join(src, f)) < 4000][:60]
        if len(files) < 50:
            cls.skip = "not enough small schemas"; return
        schemas = {f: json.load(open(os.path.join(src, f))) for f in files}
        cls.train = dict(list(schemas.items())[:45]); cls.test = dict(list(schemas.items())[45:])
        cls.base = os.path.join(cls.tmp.name, "pub"); os.makedirs(cls.base)
        write_baseline(cls.base, cls.train, cls.test, tasks_per_id={n: TASKS for n in schemas})
        A.write_anchor(cls.base)
        cls.anchor_file = os.path.join(cls.base, A.ANCHOR_FILENAME)
        cfg = _builder_config(cls.tmp.name, cls.base, max_new_references_per_schema=2)
        cls.cfg_path = os.path.join(cls.tmp.name, "cfg.json"); json.dump(cfg, open(cls.cfg_path, "w"))
        cls.out, cls.calls = os.path.join(cls.tmp.name, "pilot"), os.path.join(cls.tmp.name, "calls.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            build_main(["--config", cls.cfg_path, "--smoke-total", "60", "--out-dir", cls.out,
                        "--record-calls", cls.calls, "--max-llm-calls", "600"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if self.skip:
            self.skipTest(self.skip)
        self.work = tempfile.TemporaryDirectory(); self.addCleanup(self.work.cleanup)
        self.out = os.path.join(self.work.name, "o"); shutil.copytree(self.out_src, self.out)

    out_src = property(lambda self: self.__class__.out)

    def check(self, **kw):
        return csr.check(self.out, per_task=20, calls_log=self.calls, build_exit_code=0, anchor_file=self.anchor_file,
                         require_smoke_total=60, **kw)

    def test_pilot_builds_exactly_20_20_20_from_published_train_only(self):
        r = self.check()
        self.assertEqual((r["verdict"], r["counts"]), ("SUCCESS", {"create": 20, "fix": 20, "modify": 20}), r)
        anchor = A.load_anchor(self.anchor_file, os.path.join(self.base, "reserved_schemas.json"))
        train = {e["schema_hash"] for e in anchor["train"]}; test = {e["schema_hash"] for e in anchor["test"]}
        for task in TASKS:
            d = os.path.join(self.out, f"benchmark_{task}", "train")
            for fn in os.listdir(d):
                h = schema_hash(json.load(open(os.path.join(d, fn)))["schema"])
                self.assertIn(h, train); self.assertNotIn(h, test)
        self.assertEqual(os.listdir(os.path.join(self.out, "benchmark_create", "test")), [])

    def test_nineteen_is_a_shortfall_and_twenty_one_is_invalid(self):
        d = os.path.join(self.out, "benchmark_fix", "train"); names = sorted(os.listdir(d))
        shutil.copy(os.path.join(d, names[0]), os.path.join(d, "extra__a9.json"))
        self.assertEqual(self.check()["verdict"], "INVALID_OUTPUT")
        os.remove(os.path.join(d, "extra__a9.json")); os.remove(os.path.join(d, names[0]))
        r = self.check()
        self.assertEqual((r["verdict"], r["code"]), ("DATA_SHORTFALL", 2))

    def test_smoke_sized_output_is_rejected_by_the_pilot_check(self):
        mf = os.path.join(self.out, "generation_manifest.json"); m = json.load(open(mf)); m["smoke_total"] = 6
        json.dump(m, open(mf, "w"))
        r = self.check()
        self.assertEqual(r["verdict"], "TECHNICAL_FAILURE"); self.assertTrue(any("smoke_total" in t for t in r["technical"]))

    def test_leakage_and_foreign_schema_still_detected(self):
        d = os.path.join(self.out, "benchmark_create", "train"); name = sorted(os.listdir(d))[0]
        rec = json.load(open(os.path.join(d, name))); test_schema = next(iter(self.test.values()))
        rec["schema"] = test_schema; json.dump(rec, open(os.path.join(d, name), "w"))
        r = self.check()
        self.assertEqual((r["verdict"], r["code"]), ("INVALID_OUTPUT", 4))
        self.assertTrue(any("test set" in i for i in r["invalid"]))

    def test_metrics_report_uses_recorded_events(self):
        rep = pr.build_report(self.out, self.calls, 20, 0, 1000, 1180, 600, 0)
        self.assertTrue(rep["all_tasks_reached_target"])
        n = len(pr._jsonl(self.calls)[0])
        self.assertEqual(rep["llm"]["total_calls"]["value"], n)
        self.assertEqual(rep["duration"]["generation_seconds"]["value"], 180)
        for task in TASKS:
            row = rep["tasks"][task]
            self.assertEqual(row["accepted"]["value"], 20)
            self.assertEqual(row["attempts"]["value"], 20 + row["failed_attempts"]["value"])
            self.assertAlmostEqual(row["acceptance_rate"]["value"], 20 / row["attempts"]["value"], places=3)
            self.assertEqual(row["llm_calls"]["status"], "unavailable")      # not tagged per task: never invented
        self.assertIn("value", rep["failure_reasons"]); self.assertFalse(rep["llm"]["budget"]["exhausted"])

    def test_metrics_report_marks_missing_data_unavailable_instead_of_guessing(self):
        shutil.rmtree(os.path.join(self.out, "_generation_reports"))
        rep = pr.build_report(self.out, None, 20, 1)
        self.assertEqual(rep["duration"]["generation_seconds"]["status"], "unavailable")
        self.assertEqual(rep["llm"]["total_calls"]["status"], "unavailable")
        self.assertEqual(rep["tasks"]["fix"]["attempts"]["status"], "unavailable")
        self.assertEqual(rep["failure_reasons"]["status"], "unavailable")
        self.assertTrue(rep["tasks"]["fix"]["reached_target"])               # files on disk are still counted
        rep = pr.build_report(self.out, self.calls, 20, 0, max_llm_calls=3)
        self.assertTrue(rep["llm"]["budget"]["exhausted"])

    def test_report_file_is_never_overwritten(self):
        target = os.path.join(self.work.name, "r.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pr.main(["--out-dir", self.out, "--calls-log", self.calls, "--write", target]), 0)
            first = open(target).read()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(pr.main(["--out-dir", self.out, "--write", target]), 1)
        self.assertEqual(open(target).read(), first)

    def test_scan_of_a_truncated_file_stops_resume(self):
        """Resume limitation: examples are written non-atomically, so a kill during a write leaves a broken file and a
        second build into the same directory fails instead of continuing. The pilot never reuses a directory."""
        out = os.path.join(self.work.name, "again"); shutil.copytree(self.out_src, out)
        victim = os.path.join(out, "benchmark_modify", "train", sorted(os.listdir(os.path.join(out, "benchmark_modify", "train")))[0])
        data = open(victim).read(); open(victim, "w").write(data[: len(data) // 2])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            build_main(["--config", self.cfg_path, "--smoke-total", "60", "--out-dir", out])


class RealAnchorPilotTest(unittest.TestCase):
    def test_real_pool_supplies_20_per_task(self):
        anchor = os.path.join(ROOT, "data_published", "7962b0947f7d", A.ANCHOR_FILENAME)
        if not os.path.isfile(anchor):
            self.skipTest("real anchor not present in this checkout")
        env = {**os.environ, "PUBLISHED_SHA12": "7962b0947f7d", "GRANITE_MODEL_ID": "m", "VLLM_BASE_URL": "http://127.0.0.1:1/v1",
               "VLLM_API_KEY": "x", "RUN_NAME": "t"}
        r = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "src/build_benchmark.py"), "--config",
                            os.path.join(ROOT, "configs/scale_only_vllm_granite.json"), "--smoke-total", "60", "--out-dir",
                            os.path.join(tempfile.gettempdir(), "never_created_pilot_check"), "--check-pool"],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.loads(r.stdout)
        self.assertEqual(rep["pool"]["pool_source"], "published_train")
        self.assertEqual({t: v["target"] for t, v in rep["per_task"].items()}, {"create": 20, "fix": 20, "modify": 20})
        self.assertTrue(all(v["upper_bound_if_every_attempt_succeeds"] >= 20 for v in rep["per_task"].values()))


if __name__ == "__main__":
    unittest.main()
