"""CPU-only tests for slurm/full_chunk_granite_v2.sbatch: chunk arguments, limits, path isolation, requeue, model cache."""
import json, os, re, stat, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_smoke_preflight as T

FULL = os.path.join(ROOT, "slurm", "full_chunk_granite_v2.sbatch")
SMOKE = os.path.join(ROOT, "slurm", "smoke_granite_v2.sbatch")
PILOT = os.path.join(ROOT, "slurm", "pilot_granite_v2.sbatch")


def heredoc(text, marker):
    m = re.search(r"<<'%s'[^\n]*\n(.*?)\n%s\n" % (marker, marker), text, re.S)
    assert m, marker
    return m.group(1)


class FullScriptStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sb, cls.smoke, cls.pilot = Path(FULL).read_text(), Path(SMOKE).read_text(), Path(PILOT).read_text()

    def test_valid_bash_and_other_scripts_untouched_by_name(self):
        for path in (FULL, SMOKE, PILOT):
            self.assertEqual(subprocess.run(["bash", "-n", path], capture_output=True).returncode, 0)
        for other in (self.smoke, self.pilot):
            self.assertNotIn("full_c", other); self.assertNotIn("--chunks", other)

    def test_chunk_plan_and_limits(self):
        for needle in ["N_CHUNKS=8", 'TOTALS="create=1334,fix=1333,modify=1333"', "--time=03:00:00",
                       'MAX_LLM_CALLS="${MAX_LLM_CALLS:-3000}"', "-le 4000", 'GEN_TIMEOUT_SEC="${GEN_TIMEOUT_SEC:-5400}"', "-le 7200",
                       "MAX_MODEL_LEN=8192", '--max-model-len "$MAX_MODEL_LEN"', '--context-guard "$MAX_MODEL_LEN"',
                       '--chunks "$N_CHUNKS" --chunk-index "$CHUNK_INDEX"', "--smoke-openai-timeout 180",
                       "timeout --kill-after=60", 'CHUNK_INDEX" =~ ^[1-8]$', '--require-context-guard "$MAX_MODEL_LEN"']:
            self.assertIn(needle, self.sb, needle)

    def test_scientific_and_safety_settings_equal_smoke(self):
        for needle in ['CONFIG="configs/scale_only_vllm_granite.json"', "GRANITE_REVISION:-b8c0982bab7fde4eb48110f5a069527c008fab39",
                       "GRANITE_MODEL_ID:-ibm-granite/granite-4.0-h-small", "PUBLISHED_REVISION=7962b0947f7d96eb3a0e6b9b928d939c7f3061a6",
                       "ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6", "--gpu-memory-utilization 0.90",
                       "--tensor-parallel-size 1", "--gpus=rtx_pro_6000:1", "umask 077", "SLURM_RESTART_COUNT", "--open-mode=append",
                       "secrets.token_urlsafe", "--host 127.0.0.1", 'RUN_ID="${JOB}-r${ATTEMPT}"', '[[ ! -e "$path" && ! -L "$path" ]]',
                       "reuse.pool_source=published_train", "kill -TERM --", "curl -qsf --noproxy"]:
            self.assertIn(needle, self.sb, needle)
        for forbidden in ("--temperature", "--top-p", "--max-tokens", "standard_ratio", "complex_ratio", "rm -rf /", "rm -rf \"$MODEL_CACHE"):
            self.assertNotIn(forbidden, self.sb)
        self.assertEqual(self.sb.count("rm -rf"), self.smoke.count("rm -rf"))     # only the private job scratch is ever removed

    def test_every_path_is_chunk_and_attempt_specific_and_never_the_published_baseline(self):
        for needle in ['RUN_NAME="full_c${IDX2}_${RUN_ID}"', 'OUT_DIR="$CHUNKS_ROOT/c${IDX2}_${RUN_ID}"', "logs/full-vllm-${RUN_ID}.log",
                       "logs/full-pool-check-${RUN_ID}.json", "benchson-full-$RUN_ID.XXXXXXXX", "benchson-full-%j.out"]:
            self.assertIn(needle, self.sb, needle)
        for stale in ("logs/vllm-", "logs/pilot-", "benchson-smoke-", "benchson-pilot-", "data_published/${PUBLISHED_SHA12}/benchmark"):
            self.assertNotIn(stale, self.sb)
        self.assertIn('"$OUT_DIR" "$CALLS_LOG" "$REPORT_JSON"', self.sb)

    def test_parent_must_be_explicit_and_is_revalidated_before_the_gpu_is_used(self):
        self.assertIn("PARENT_CHUNK must be the directory name", self.sb)
        self.assertLess(self.sb.index("PYPARENT"), self.sb.index("nvidia-smi"))
        self.assertIn("chunked_run.validate_chunk", self.sb)
        self.assertNotIn("ls -t", self.sb); self.assertNotIn("find data_scale_only", self.sb)    # never picks a parent by itself
        self.assertLess(self.sb.index("src/chunked_run.py seal"), self.sb.index("src/pilot_report.py"))

    def test_pool_preflight_checks_each_task_against_the_full_target(self):
        body = self.sb.split("POOL_REPORT=")[1].split("PYPARENT")[0]
        self.assertNotIn("--chunks", body.split("fail \"pool check failed\"")[0])
        self.assertIn('row["target"] != totals[name]', body); self.assertIn("feasible_at_assumed_acceptance", body)


class FullScriptSlurmTest(T.SlurmCleanupTest):
    """The whole cleanup / preflight / requeue suite, on the chunk script (chunk 1)."""
    SCRIPT = "slurm/full_chunk_granite_v2.sbatch"
    LOG, POOLLOG, TARGET, SCRATCH = "full-vllm", "full-pool-check", 20, "benchson-full"

    def extra_stubs(self):
        py = self.bin / "python"
        py.write_text(py.read_text().replace(
            '{"target": 20, "upper_bound_if_every_attempt_succeeds": 20 + 1}',
            '{"target": {"create": 1334, "fix": 1333, "modify": 1333}[task], "upper_bound_if_every_attempt_succeeds": 3228, '
            '"feasible_at_assumed_acceptance": True}'))
        self.stub("timeout", "args = sys.argv[1:]\nassert args[0].startswith('--kill-after=')\nassert args[1].isdigit()\nos.execvp(args[2], args[2:])")
        os.environ["CHUNK_INDEX"] = "1"; self.addCleanup(os.environ.pop, "CHUNK_INDEX", None)
        os.environ.pop("PARENT_CHUNK", None)

    def test_report_and_isolated_outputs_for_every_attempt(self):
        code, output = self.run_job("success")
        self.assertEqual(code, 0, output); self.assertIn("quota create=167,fix=166,modify=166", output)
        self.assertTrue((self.root / "data_scale_only/full/c01_1234-r0_report.json").exists())
        code, output = self.run_job("success", SLURM_RESTART_COUNT="1")
        self.assertEqual(code, 0, output)
        self.assertTrue((self.root / "data_scale_only/full/c01_1234-r1_report.json").exists())

    def test_chunk_arguments_are_validated(self):
        cases = [({"CHUNK_INDEX": "0"}, "CHUNK_INDEX"), ({"CHUNK_INDEX": "9"}, "CHUNK_INDEX"), ({"CHUNK_INDEX": "x"}, "CHUNK_INDEX"),
                 ({"CHUNK_INDEX": "1", "PARENT_CHUNK": "c01_1-r0"}, "no parent"),
                 ({"CHUNK_INDEX": "2"}, "PARENT_CHUNK"),
                 ({"CHUNK_INDEX": "2", "PARENT_CHUNK": "../c01_1-r0"}, "PARENT_CHUNK"),
                 ({"CHUNK_INDEX": "2", "PARENT_CHUNK": "/etc/c01_1-r0"}, "PARENT_CHUNK"),
                 ({"CHUNK_INDEX": "3", "PARENT_CHUNK": "c01_1-r0"}, "needs chunk 2"),
                 ({"MAX_LLM_CALLS": "4001"}, "MAX_LLM_CALLS"), ({"GEN_TIMEOUT_SEC": "7201"}, "GEN_TIMEOUT_SEC")]
        for env, text in cases:
            code, output = self.run_job("success", **env)
            self.assertEqual(code, 10, (env, output)); self.assertIn(text, output)

    def test_chunk_two_gets_its_parent_and_quota(self):
        code, output = self.run_job("success", CHUNK_INDEX="2", PARENT_CHUNK="c01_1234-r0")
        self.assertEqual(code, 0, output)
        self.assertTrue((self.root / "data_scale_only/full/c02_1234-r0_report.json").exists())


class ModelCacheScriptTest(unittest.TestCase):
    """The two embedded Python snippets run for real: the cache directory check and the download/integrity check."""

    @classmethod
    def setUpClass(cls):
        cls.sb = Path(FULL).read_text()
        cls.check = heredoc(cls.sb, "PYCACHE"); cls.download = heredoc(cls.sb, "PY")

    def run_check(self, path):
        return subprocess.run([sys.executable, "-c", self.check, path], capture_output=True, text=True)

    def test_cache_directory_must_be_an_existing_private_real_directory_you_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "cache"); os.mkdir(good); os.chmod(good, 0o700)
            self.assertEqual(self.run_check(good).returncode, 0)
            self.assertNotEqual(self.run_check(os.path.join(tmp, "missing")).returncode, 0)     # never created
            self.assertFalse(os.path.exists(os.path.join(tmp, "missing")))
            link = os.path.join(tmp, "link"); os.symlink(good, link)
            self.assertNotEqual(self.run_check(link).returncode, 0)
            shared = os.path.join(tmp, "shared"); os.mkdir(shared); os.chmod(shared, 0o770)
            self.assertNotEqual(self.run_check(shared).returncode, 0)
            self.assertNotEqual(self.run_check("relative/path").returncode, 0)
            f = os.path.join(tmp, "file"); open(f, "w").close()
            self.assertNotEqual(self.run_check(f).returncode, 0)

    def fake_hub(self, tmp, calls_file, complete_local, complete_after):
        """A stand-in huggingface_hub that records each call and builds a snapshot directory."""
        rev = "b8c0982bab7fde4eb48110f5a069527c008fab39"
        snap = os.path.join(tmp, "cache", "models--ibm-granite--granite-4.0-h-small", "snapshots", rev)
        os.makedirs(os.path.join(tmp, "pkg"), exist_ok=True)
        Path(tmp, "pkg", "huggingface_hub.py").write_text(textwrap.dedent(f'''
            import json, os
            SNAP = {snap!r}
            def _build(complete):
                os.makedirs(SNAP, exist_ok=True)
                open(os.path.join(SNAP, "config.json"), "w").write("{{}}")
                json.dump({{"weight_map": {{"a": "w1.safetensors", "b": "w2.safetensors"}}}}, open(os.path.join(SNAP, "model.safetensors.index.json"), "w"))
                open(os.path.join(SNAP, "w1.safetensors"), "w").write("x")
                open(os.path.join(SNAP, "w2.safetensors"), "w").write("x" if complete else "")
            def snapshot_download(repo_id, revision, cache_dir, local_files_only=False):
                open({calls_file!r}, "a").write(json.dumps({{"local_only": local_files_only, "cache_dir": cache_dir, "revision": revision}}) + "\\n")
                if local_files_only:
                    if not {complete_local!r} and not os.path.isdir(SNAP):
                        raise RuntimeError("not cached")
                    if {complete_local!r}: _build(True)
                    return SNAP
                _build({complete_after!r})
                return SNAP
        '''))
        return snap

    def run_download(self, tmp, **env):
        e = {**os.environ, "PYTHONPATH": os.path.join(tmp, "pkg"), "GRANITE_MODEL_ID": "ibm-granite/granite-4.0-h-small",
             "GRANITE_REVISION": "b8c0982bab7fde4eb48110f5a069527c008fab39", "HF_HOME": os.path.join(tmp, "hf"), **env}
        return subprocess.run([sys.executable, "-c", self.download], capture_output=True, text=True, env=e)

    def calls(self, path):
        return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

    def test_complete_cache_is_used_without_any_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "calls"); snap = self.fake_hub(tmp, log, True, True)
            r = self.run_download(tmp, MODEL_CACHE_DIR=os.path.join(tmp, "cache"))
            self.assertEqual((r.returncode, r.stdout.strip()), (0, snap), r.stderr)
            self.assertEqual([c["local_only"] for c in self.calls(log)], [True])
            self.assertEqual(self.calls(log)[0]["cache_dir"], os.path.join(tmp, "cache"))
            self.assertEqual(self.calls(log)[0]["revision"], "b8c0982bab7fde4eb48110f5a069527c008fab39")

    def test_missing_cache_downloads_the_pinned_revision_and_verifies_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "calls"); self.fake_hub(tmp, log, False, True)
            r = self.run_download(tmp, MODEL_CACHE_DIR=os.path.join(tmp, "cache"))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual([c["local_only"] for c in self.calls(log)], [True, False])

    def test_default_without_cache_dir_stays_in_private_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "calls"); self.fake_hub(tmp, log, False, True)
            self.assertEqual(self.run_download(tmp).returncode, 0)
            self.assertEqual(self.calls(log)[0]["cache_dir"], os.path.join(tmp, "hf", "hub"))

    def test_incomplete_snapshot_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "calls"); self.fake_hub(tmp, log, False, False)    # a weight file stays empty
            r = self.run_download(tmp, MODEL_CACHE_DIR=os.path.join(tmp, "cache"))
            self.assertNotEqual(r.returncode, 0); self.assertIn("incomplete", r.stderr)
            self.assertEqual(r.stdout.strip(), "")

    def test_wrong_revision_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "calls"); self.fake_hub(tmp, log, True, True)
            r = self.run_download(tmp, MODEL_CACHE_DIR=os.path.join(tmp, "cache"), GRANITE_REVISION="0" * 40)
            self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
