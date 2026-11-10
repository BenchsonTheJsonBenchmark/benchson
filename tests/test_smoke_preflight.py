"""Offline tests for the Scale-Only smoke preflight: jsb_schemas preparation, the smoke result judge,
the OpenAI-compatible provider and the Slurm script (static checks only; nothing is submitted)."""
import io, json, os, re, shutil, signal, stat, subprocess, sys, tempfile, time, types, unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

import check_smoke_result as csr
import prepare_jsb_schemas as pjs
from build_benchmark import main as build_main


# ---------------------------------------------------------------- prepare_jsb_schemas
class PrepareJsbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.repo = os.path.join(self.tmp.name, "repo"); os.makedirs(self.repo)
        self._git("init", "-q"); self._git("config", "user.email", "t@t"); self._git("config", "user.name", "t")
        os.makedirs(os.path.join(self.repo, "data", "jsb_schemas", "test"))
        for i in range(3):
            with open(os.path.join(self.repo, "data", "jsb_schemas", "test", f"s{i}.json"), "w") as f:
                json.dump({"type": "object", "title": f"s{i}"}, f)
        self._git("add", "-A"); self._git("commit", "-q", "-m", "x")
        self.commit = self._git("rev-parse", "HEAD").stdout.decode().strip()
        self.tree = self._git("rev-parse", "HEAD:data/jsb_schemas").stdout.decode().strip()

    def _git(self, *a):
        return subprocess.run(["git", *a], cwd=self.repo, capture_output=True, check=True)

    def test_tree_oid_matches_git(self):
        self.assertEqual(pjs.tree_oid(os.path.join(self.repo, "data", "jsb_schemas")), self.tree)

    def test_prepare_creates_verifies_and_is_idempotent(self):
        target = os.path.join(self.tmp.name, "dest")
        status, _ = pjs.prepare(target, self.commit, self.tree, 3, repo=self.repo)
        self.assertEqual(status, "created")
        self.assertEqual(pjs.prepare(target, self.commit, self.tree, 3, repo=self.repo)[0], "present")
        self.assertFalse([d for d in os.listdir(self.tmp.name) if ".partial-" in d])

    def test_wrong_pin_leaves_nothing_behind(self):
        target = os.path.join(self.tmp.name, "dest")
        with self.assertRaises(pjs.PrepareError):
            pjs.prepare(target, self.commit, "0" * 40, 3, repo=self.repo)
        self.assertFalse(os.path.exists(target))
        self.assertFalse([d for d in os.listdir(self.tmp.name) if ".partial-" in d])

    def test_unknown_commit_has_clear_message(self):
        with self.assertRaises(pjs.PrepareError) as cm:
            pjs.prepare(os.path.join(self.tmp.name, "dest"), "1" * 40, self.tree, 3, repo=self.repo)
        self.assertIn("git fetch", str(cm.exception))

    def test_existing_modified_dir_is_never_touched(self):
        target = os.path.join(self.tmp.name, "dest")
        pjs.prepare(target, self.commit, self.tree, 3, repo=self.repo)
        extra = os.path.join(target, "test", "extra.json")
        open(extra, "w").write("{}")
        with self.assertRaises(pjs.PrepareError):
            pjs.prepare(target, self.commit, self.tree, 3, repo=self.repo)
        self.assertTrue(os.path.exists(extra))

    def test_pinned_constants_are_plausible(self):
        self.assertRegex(pjs.PINNED_COMMIT, r"^[0-9a-f]{40}$")
        self.assertRegex(pjs.PINNED_TREE_OID, r"^[0-9a-f]{40}$")
        self.assertEqual(pjs.PINNED_FILE_COUNT, 1308)

    def test_real_corpus_if_present(self):
        target = pjs.DEFAULT_TARGET
        if not os.path.isdir(target):
            self.skipTest("data/jsb_schemas not present here")
        ok, msg = pjs.verify(target)
        self.assertTrue(ok, msg)

    def test_missing_source_error_names_the_fix(self):
        from data_generation.tasks.base_task import BaseDataTask
        src = open(os.path.join(ROOT, "src", "data_generation", "tasks", "base_task.py")).read()
        self.assertIn("prepare_jsb_schemas.py", src)


# ---------------------------------------------------------------- smoke result judge
def _mock_smoke(tmp, total=6):
    base = os.path.join(tmp, "base"); os.makedirs(base)
    reserved = os.path.join(base, "reserved_schemas.json")
    json.dump({"dataset": "x", "revision": "a" * 40, "ids": {"train": [], "test": []},
               "schema_hashes": {"train": [], "test": []}, "reference_hashes_by_schema": {},
               "task_keys": {t: {"exact": [], "content": []} for t in ("create", "fix", "modify")}}, open(reserved, "w"))
    cfg = {"scale_only": True, "llm_provider": {"module": "src.llm.mock.mock_provider", "class": "MockProvider", "params": {}},
           "sources": [{"name": "schemas", "max_schema_kb": 10}], "exclusions": {"file": reserved},
           "output_root": os.path.join(tmp, "out"), "seed": 0, "split": {"test_ratio": 0.15},
           "reuse": {"max_new_references_per_schema": 2}, "counts": {"train": {"create": 3, "fix": 3, "modify": 3}},
           "validation": {"train": {"round_trip": False}},
           "generation": {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"}}
    cfg_path = os.path.join(tmp, "cfg.json"); json.dump(cfg, open(cfg_path, "w"))
    out, calls = os.path.join(tmp, "o"), os.path.join(tmp, "calls.jsonl")
    build_main(["--config", cfg_path, "--smoke-total", str(total), "--out-dir", out,
                "--record-calls", calls, "--max-llm-calls", "500"])
    return out, calls


class SmokeJudgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with mock.patch("sys.stdout", new=open(os.devnull, "w")):
            cls.out, cls.calls = _mock_smoke(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory(); self.addCleanup(self.work.cleanup)
        self.out = os.path.join(self.work.name, "o"); shutil.copytree(self.out_src(), self.out)

    def out_src(self):
        return self.__class__.out

    def _train(self, task):
        return os.path.join(self.out, f"benchmark_{task}", "train")

    def test_exactly_two_per_task_is_success(self):
        r = csr.check(self.out, calls_log=self.__class__.calls, build_exit_code=0)
        self.assertEqual(r["verdict"], "SUCCESS", r)
        self.assertEqual(r["counts"], {"create": 2, "fix": 2, "modify": 2})

    def test_missing_example_is_shortfall_not_technical(self):
        d = self._train("fix"); os.remove(os.path.join(d, sorted(os.listdir(d))[0]))
        r = csr.check(self.out, build_exit_code=0)
        self.assertEqual((r["verdict"], r["code"]), ("DATA_SHORTFALL", 2))
        self.assertEqual(r["technical"], [])

    def _anchor(self, train_hashes, test_hashes=(), test_ids=()):
        entries = lambda hs, ids=(): [{"id": (list(ids) + [f"x{i}" for i in range(len(hs))])[i], "schema_hash": h} for i, h in enumerate(hs)]
        return {"train": entries(train_hashes), "test": entries(test_hashes, test_ids)}

    def _hashes(self):
        from data_generation.published import schema_hash
        hs = set()
        for task in ("create", "fix", "modify"):
            d = self._train(task)
            for f in os.listdir(d):
                hs.add(schema_hash(json.load(open(os.path.join(d, f)))["schema"]))
        return hs

    def test_anchor_leakage_check_passes_for_published_train_schemas(self):
        with mock.patch("data_generation.anchor.load_anchor", return_value=self._anchor(self._hashes())):
            r = csr.check(self.out, build_exit_code=0, anchor_file="a.json")
        self.assertEqual(r["verdict"], "SUCCESS", r)

    def test_schema_outside_the_published_train_set_is_invalid(self):
        with mock.patch("data_generation.anchor.load_anchor", return_value=self._anchor(["nope"])):
            r = csr.check(self.out, build_exit_code=0, anchor_file="a.json")
        self.assertEqual((r["verdict"], r["code"]), ("INVALID_OUTPUT", 4))

    def test_schema_matching_the_published_test_set_is_invalid(self):
        hs = self._hashes()
        with mock.patch("data_generation.anchor.load_anchor", return_value=self._anchor(hs, test_hashes=[next(iter(hs))])):
            r = csr.check(self.out, build_exit_code=0, anchor_file="a.json")
        self.assertEqual((r["verdict"], r["code"]), ("INVALID_OUTPUT", 4))

    def test_unusable_anchor_is_a_technical_failure(self):
        r = csr.check(self.out, build_exit_code=0, anchor_file=os.path.join(self.work.name, "missing.json"))
        self.assertEqual((r["verdict"], r["code"]), ("TECHNICAL_FAILURE", 3))

    def test_extra_example_is_invalid_output(self):
        d = self._train("create"); f = sorted(os.listdir(d))[0]
        shutil.copy(os.path.join(d, f), os.path.join(d, "dup__a9.json"))
        self.assertEqual(csr.check(self.out, build_exit_code=0)["verdict"], "INVALID_OUTPUT")

    def test_invalid_content_is_invalid_output(self):
        d = self._train("fix"); p = os.path.join(d, sorted(os.listdir(d))[0])
        rec = json.load(open(p)); rec["erroneous_json"] = rec["valid_json"]
        json.dump(rec, open(p, "w"))
        r = csr.check(self.out, build_exit_code=0)
        self.assertEqual(r["verdict"], "INVALID_OUTPUT"); self.assertTrue(any("equals" in x or "valid" in x for x in r["invalid"]))

    def test_corrupt_file_is_invalid_output(self):
        d = self._train("modify"); open(os.path.join(d, sorted(os.listdir(d))[0]), "w").write("{not json")
        self.assertEqual(csr.check(self.out, build_exit_code=0)["verdict"], "INVALID_OUTPUT")

    def test_nonzero_build_exit_is_technical_even_with_full_output(self):
        r = csr.check(self.out, build_exit_code=1)
        self.assertEqual((r["verdict"], r["code"]), ("TECHNICAL_FAILURE", 3))

    def test_technical_outranks_shortfall(self):
        d = self._train("fix"); os.remove(os.path.join(d, sorted(os.listdir(d))[0]))
        self.assertEqual(csr.check(self.out, build_exit_code=2)["verdict"], "TECHNICAL_FAILURE")

    def test_stopped_manifest_is_technical(self):
        mp = os.path.join(self.out, "generation_manifest.json"); m = json.load(open(mp))
        m["status"] = "stopped: CallBudgetExceeded"; json.dump(m, open(mp, "w"))
        self.assertEqual(csr.check(self.out, build_exit_code=0)["verdict"], "TECHNICAL_FAILURE")

    def test_failed_model_calls_are_technical(self):
        log = os.path.join(self.work.name, "calls.jsonl")
        open(log, "w").write(json.dumps({"call": 1, "error": "ConnectionError: refused"}) + "\n")
        self.assertEqual(csr.check(self.out, calls_log=log, build_exit_code=0)["verdict"], "TECHNICAL_FAILURE")

    def test_empty_call_log_is_technical(self):
        log = os.path.join(self.work.name, "calls.jsonl"); open(log, "w").close()
        self.assertEqual(csr.check(self.out, calls_log=log, build_exit_code=0)["verdict"], "TECHNICAL_FAILURE")

    def test_malformed_or_non_object_call_log_is_technical(self):
        log = Path(self.work.name) / "calls.jsonl"
        for content in ('{"call":1,"error":null}\n{', '[]\n', '{}\n'):
            with self.subTest(content=content):
                log.write_text(content)
                self.assertEqual(csr.check(self.out, calls_log=log)["code"], csr.TECHNICAL)

    def test_corrupt_manifest_and_summary_are_technical(self):
        for path in (Path(self.out) / "generation_manifest.json",
                     Path(csr.latest_report_dir(self.out)) / "generation_summary.json"):
            original = path.read_bytes()
            for content in (b"{", b"[]", b"\xff"):
                with self.subTest(path=path, content=content):
                    path.write_bytes(content)
                    self.assertEqual(csr.check(self.out)["code"], csr.TECHNICAL)
            path.write_bytes(original)

    def test_missing_or_corrupt_required_jsonl_is_technical(self):
        for name in ("generation_failures.jsonl", "generation_events.jsonl"):
            path = Path(csr.latest_report_dir(self.out)) / name
            original = path.read_bytes()
            path.unlink()
            self.assertEqual(csr.check(self.out)["code"], csr.TECHNICAL)
            path.write_text('{"truncated":')
            self.assertEqual(csr.check(self.out)["code"], csr.TECHNICAL)
            path.write_bytes(original)

    def test_missing_output_dir_is_technical(self):
        self.assertEqual(csr.check(os.path.join(self.work.name, "nope"))["verdict"], "TECHNICAL_FAILURE")

    def test_provider_error_event_is_technical(self):
        rd = csr.latest_report_dir(self.out)
        with open(os.path.join(rd, "generation_events.jsonl"), "a") as f:
            f.write(json.dumps({"kind": "provider_error", "schema_id": "x", "detail": "boom"}) + "\n")
        self.assertEqual(csr.check(self.out, build_exit_code=0)["verdict"], "TECHNICAL_FAILURE")

    def test_cli_exit_codes(self):
        self.assertEqual(csr.main(["--out-dir", self.out, "--build-exit-code", "0"]), 0)
        self.assertEqual(csr.main(["--out-dir", self.out, "--build-exit-code", "1"]), 3)

    def test_smoke_total_6_means_two_per_task(self):
        from data_generation.generation_mix import split_total
        self.assertEqual(split_total(6, ["create", "fix", "modify"]), {"create": 2, "fix": 2, "modify": 2})


# ---------------------------------------------------------------- OpenAI-compatible provider
class OpenAIProviderTest(unittest.TestCase):
    def _provider(self, content, **kw):
        fake_openai = types.SimpleNamespace()
        created = {}
        class Client:
            def __init__(self, **k):
                created.update(k)
                msg = types.SimpleNamespace(content=content)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                    create=lambda **p: (created.setdefault("request", p), types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)]))[1]))
        fake_openai.OpenAI = Client
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            from src.llm.openai.openai_provider import OpenAIProvider
            return OpenAIProvider(api_key="k", model="m", base_url="http://127.0.0.1:1/v1", **kw), created

    def test_none_content_becomes_empty_string(self):
        p, _ = self._provider(None)
        self.assertEqual(p.generate([{"role": "user", "content": "x"}])[0], "")

    def test_client_receives_base_url_key_timeout_and_params(self):
        p, created = self._provider("ok", timeout=30, max_retries=1, model_params={"temperature": 0.7, "max_tokens": 4096, "top_p": 0.95})
        self.assertEqual(p.generate([{"role": "user", "content": "x"}])[0], "ok")
        self.assertEqual((created["base_url"], created["api_key"], created["timeout"], created["max_retries"]),
                         ("http://127.0.0.1:1/v1", "k", 30, 1))
        self.assertEqual(created["request"]["temperature"], 0.7)
        self.assertEqual(created["request"]["max_tokens"], 4096)

    def test_defaults_do_not_set_timeout(self):
        _, created = self._provider("ok")
        self.assertNotIn("timeout", created); self.assertNotIn("max_retries", created)

    def test_empty_assistant_turn_dropped_only_when_enabled(self):
        msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]
        p, created = self._provider("ok", drop_empty_assistant_turn=True)
        p.generate(msgs); self.assertEqual(len(created["request"]["messages"]), 1)
        p, created = self._provider("ok")
        p.generate(msgs); self.assertEqual(len(created["request"]["messages"]), 2)


# ---------------------------------------------------------------- config + Slurm script (static)
class ConfigAndSlurmTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.load(open(os.path.join(ROOT, "configs", "scale_only_vllm_granite.json")))
        self.sb = open(os.path.join(ROOT, "slurm", "smoke_granite_v2.sbatch")).read()

    def test_config_keeps_sources_and_scientific_settings(self):
        self.assertEqual([s["name"] for s in self.cfg["sources"]], ["schemas", "jsb_schemas"])
        p = self.cfg["llm_provider"]["params"]
        self.assertEqual(p["model_params"], {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.95})
        self.assertEqual(self.cfg["generation"], {"standard_ratio": 1.0, "complex_ratio": 0.0, "standard_modify_strategy": "original"})
        self.assertEqual(self.cfg["split"]["test_ratio"], 0.15)
        self.assertEqual(self.cfg["seed"], 0)

    def test_config_credentials_are_placeholders(self):
        p = self.cfg["llm_provider"]["params"]
        self.assertTrue(p["api_key"].startswith("${VLLM_API_KEY"))
        self.assertTrue(p["base_url"].startswith("${VLLM_BASE_URL"))
        self.assertTrue(p["model"].startswith("${GRANITE_MODEL_ID"))

    def test_sbatch_is_valid_bash(self):
        r = subprocess.run(["bash", "-n", os.path.join(ROOT, "slurm", "smoke_granite_v2.sbatch")], capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_sbatch_requirements(self):
        sb = self.sb
        self.assertIn("--smoke-total \"$SMOKE_TOTAL\"", sb); self.assertIn("SMOKE_TOTAL=6", sb); self.assertIn("PER_TASK=2", sb)
        self.assertIn("check_smoke_result.py", sb)
        self.assertNotIn("--check-only", sb)                             # published_train pool does not need data/jsb_schemas
        self.assertIn("reuse.pool_source=published_train", sb)
        self.assertIn("published_train_schemas.json", sb)                # anchor verified before the GPU is used
        self.assertIn("ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6", sb)
        self.assertIn("--anchor-file", sb)                               # leakage check on the produced examples
        self.assertIn("data_published/", sb)                             # baseline exclusion file verified
        self.assertRegex(sb, r'export PATH="\$\{VLLM_ENV\}/bin')         # ninja
        self.assertIn("for tool in ninja setsid curl", sb)
        self.assertIn("VLLM_API_KEY", sb); self.assertIn("secrets.token_urlsafe", sb)
        self.assertNotRegex(sb, r'VLLM_API_KEY="EMPTY"')
        self.assertNotRegex(sb, r"--api-key")                            # key never on a command line
        self.assertNotRegex(sb, r"echo[^\n]*\$VLLM_API_KEY")
        self.assertIn("--host 127.0.0.1", sb)
        self.assertIn('rm -rf -- "$JOB_SCRATCH"', sb)
        self.assertIn("trap cleanup EXIT", sb)
        self.assertIn("umask 077", sb)
        self.assertIn("SCRATCH_DIR", sb); self.assertIn("/scratch/", sb)
        self.assertIn('OUT_DIR="data_scale_only/${RUN_NAME}"', sb)       # per-job output directory
        self.assertNotRegex(sb, r"set -[a-z]*e")                         # build failure must reach the checker
        self.assertNotIn("temperature", sb)                              # no scientific setting in the job script

    def test_sbatch_pins_model_revision(self):
        self.assertRegex(self.sb, r'GRANITE_REVISION:-[0-9a-f]{40}')


class SmokeSdkOptionsTest(unittest.TestCase):
    def test_sdk_overrides_are_opt_in_and_keep_builder_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config = json.loads((Path(ROOT) / "configs/scale_only_vllm_granite.json").read_text())
            config["scale_only"] = False
            config["exclusions"]["file"] = "unused"
            config["llm_provider"]["params"]["model"] = "granite"
            config_path.write_text(json.dumps(config))
            for options in ([], ["--smoke-total", "6", "--smoke-openai-timeout", "180"]):
                with self.subTest(options=options):
                    with mock.patch("build_benchmark.LLMProvider.from_config", side_effect=RuntimeError("stop")) as factory:
                        with self.assertRaisesRegex(RuntimeError, "stop"):
                            build_main(["--config", str(config_path), *options])
                    params = factory.call_args.args[0]["params"]
                    self.assertEqual(params["model_params"], config["llm_provider"]["params"]["model_params"])
                    self.assertEqual(params.get("max_retries"), 0 if options else None)
                    self.assertEqual(params.get("timeout"), 180 if options else None)
            with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                build_main(["--config", str(config_path), "--smoke-openai-timeout", "180"])
            self.assertEqual(error.exception.code, 2)


class SlurmCleanupTest(unittest.TestCase):
    SCRIPT = "slurm/smoke_granite_v2.sbatch"   # PilotSlurmTest (tests/test_pilot.py) re-runs everything on the pilot script
    LOG = "vllm"
    POOLLOG = "pool-check"
    TARGET = 2
    SCRATCH = "benchson-smoke"

    def extra_stubs(self):
        pass

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bin = self.root / "env/bin"
        self.bin.mkdir(parents=True)
        self.scratch = self.root / "existing-scratch"
        self.scratch.mkdir()
        self.keep = self.scratch / "existing-data"
        self.keep.write_text("keep")
        (self.root / "configs").mkdir()
        (self.root / "configs/scale_only_vllm_granite.json").write_text(json.dumps({"reuse": {"pool_source": "published_train"}}))
        baseline = self.root / "data_published/7962b0947f7d/reserved_schemas.json"
        baseline.parent.mkdir(parents=True)
        baseline.write_text(json.dumps({
            "revision": "7962b0947f7d96eb3a0e6b9b928d939c7f3061a6",
            "ids": {"train": ["id"], "test": ["id"]},
            "schema_hashes": {"train": ["hash"], "test": ["hash"]},
        }))
        self.stub("git", 'print("improve-data-generation")')
        self.stub("df", 'print("header\\nfs blocks used 99999999 capacity mount")')
        self.stub("ninja", "pass")
        self.stub("nvidia-smi", "pass")
        self.stub("setsid", "os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])")
        self.stub("vllm", "Path('server-pid').write_text(str(os.getpid())); time.sleep(30)")
        self.stub("curl", """assert sys.stdin.read().startswith('header = "Authorization: Bearer ')
deadline = time.monotonic() + 5
while not Path('server-pid').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
assert Path('server-pid').exists()
""")
        self.stub("python", """
args = sys.argv[1:]
if args == ["-"]:
    print(os.environ["HF_HOME"])
elif args[0] == "-" and len(args) > 1:
    pass   # anchor integrity check (covered by test_anchor_pool); the real script is exercised there
elif args[0] == "src/build_benchmark.py" and "--check-pool" in args:
    print(json.dumps({"pool": {"pool_source": "published_train", "usable_schemas": 1076,
                               "anchor_sha256": "ab09def994a78920d1df6bfa2d0c11f6e135294a46ebc33baca9307aee0327f6"},
                      "per_task": {task: {"target": __TARGET__, "upper_bound_if_every_attempt_succeeds": __TARGET__ + 1}
                                 for task in ("create", "fix", "modify")}}))
elif args[0] == "src/build_benchmark.py" and "--chunks" in args:
    assert args[args.index("--chunks") + 1] == "8" and args[args.index("--context-guard") + 1] == "8192"
    assert args[args.index("--chunk-index") + 1] == os.environ["CHUNK_INDEX"]
    assert ("--parent-chunk" in args) == (os.environ["CHUNK_INDEX"] != "1")
    assert args[args.index("--smoke-openai-timeout") + 1] == "180"
    Path("building").touch()
    if os.environ["SMOKE_SCENARIO"] == "cancel":
        time.sleep(30)
    sys.exit(1 if os.environ["SMOKE_SCENARIO"] == "error" else 0)
elif args[0] == "src/build_benchmark.py":
    assert args[args.index("--smoke-total") + 1] == str(3 * __TARGET__)
    assert args[args.index("--smoke-openai-timeout") + 1] == "180"
    Path("building").touch()
    if os.environ["SMOKE_SCENARIO"] == "cancel":
        time.sleep(30)
    sys.exit(1 if os.environ["SMOKE_SCENARIO"] == "error" else 0)
elif args[0] == "src/chunked_run.py" and args[1] in ("selfcheck", "fingerprint"):
    pass
elif args[0] == "src/chunked_run.py" and args[1] == "increments":
    print(json.dumps({"chunks": [{"create": 167, "fix": 166, "modify": 166}] * 8}))
elif args[0] == "src/chunked_run.py":
    sys.exit(3 if args[-1] != "0" else 0)   # validate ... --build-exit-code N
elif args[0] == "src/pilot_report.py":
    open(args[args.index("--write") + 1], "x").write("{}")   # like the real tool: never overwrites
elif args[0] == "src/check_smoke_result.py":
    sys.exit(3 if args[-1] != "0" else 0)
elif args[0] == "src/prepare_jsb_schemas.py" or args[-1] in ("import deepdiff, jsonschema", "import openai", "import huggingface_hub, vllm"):
    pass
else:
    os.execv(sys.executable, [sys.executable, *args])
""".replace("__TARGET__", str(self.TARGET)))
        self.extra_stubs()

    def stub(self, name, body):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\nimport json, os, sys, time\nfrom pathlib import Path\n{body}\n")
        path.chmod(0o700)

    def run_job(self, scenario, **overrides):
        env = {**os.environ, "VLLM_ENV": str(self.bin.parent), "BENCHSON_PYTHON": str(self.bin / "python"),
               "SLURM_JOB_ID": "1234", "SLURM_SUBMIT_DIR": str(self.root), "SCRATCH_DIR": str(self.scratch),
               "SMOKE_SCENARIO": scenario, "MAX_LLM_CALLS": "60", **overrides}
        cmd = ["bash", os.path.join(ROOT, self.SCRIPT)]
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as job:
            try:
                if scenario == "cancel":
                    deadline = time.monotonic() + 8
                    while not (self.root / "building").exists() and job.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue((self.root / "building").exists())
                    private, = self.scratch.glob(self.SCRATCH + "-*")
                    self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o700)
                    job.send_signal(signal.SIGTERM)
                output, _ = job.communicate(timeout=12)
            finally:
                if job.poll() is None:
                    job.terminate()
                    job.communicate(timeout=12)
        self.assertEqual(self.keep.read_text(), "keep")
        self.assertEqual(list(self.scratch.iterdir()), [self.keep])
        server_pid = self.root / "server-pid"
        if server_pid.exists():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(server_pid.read_text()), 0)
            self.assertEqual(stat.S_IMODE((self.root / f"logs/{self.LOG}-1234-r0.log").stat().st_mode), 0o600)
        return job.returncode, output

    def test_cleanup_on_success(self):
        code, output = self.run_job("success")
        self.assertEqual(code, 0, output)

    def test_cleanup_on_build_error(self):
        code, output = self.run_job("error")
        self.assertEqual(code, 3, output)

    def test_cleanup_on_term(self):
        code, output = self.run_job("cancel")
        self.assertEqual(code, 143, output)

    def test_invalid_scratch_parent_is_rejected(self):
        code, output = self.run_job("success", SCRATCH_DIR="/")
        self.assertEqual(code, 10, output)
        self.assertIn("scratch parent", output)

    # ---- Slurm requeue / preemption: the same job ID runs again with a higher SLURM_RESTART_COUNT
    def _paths(self, attempt):
        return [self.root / f"logs/{self.LOG}-1234-r{attempt}.log", self.root / f"logs/{self.POOLLOG}-1234-r{attempt}.json"]

    def test_requeued_attempts_use_distinct_paths_and_keep_old_ones(self):
        code, output = self.run_job("success")                          # first execution (no restart count set)
        self.assertEqual(code, 0, output)
        first = {p: p.read_bytes() for p in self._paths(0)}
        self.assertTrue(all(first))
        for attempt in ("1", "2"):                                       # rapid successive requeues
            code, output = self.run_job("success", SLURM_RESTART_COUNT=attempt)
            self.assertEqual(code, 0, output)
            self.assertIn(f"run=1234-r{attempt}", output)
            self.assertTrue(all(p.exists() for p in self._paths(attempt)))
        for p, data in first.items():                                    # earlier attempt left untouched
            self.assertEqual(p.read_bytes(), data)
        self.assertEqual(len({p for a in (0, 1, 2) for p in self._paths(a)}), 6)

    def test_requeue_after_a_killed_attempt_does_not_collide(self):
        code, output = self.run_job("cancel")                            # attempt 0 terminated mid-build
        self.assertEqual(code, 143, output)
        code, output = self.run_job("success", SLURM_RESTART_COUNT="1")
        self.assertEqual(code, 0, output)
        self.assertTrue(all(p.exists() for p in self._paths(0) + self._paths(1)))

    def test_same_attempt_paths_are_still_never_reused(self):
        (self.root / "logs").mkdir(exist_ok=True)
        old = self.root / f"logs/{self.LOG}-1234-r1.log"
        old.write_text("previous attempt")
        code, output = self.run_job("success", SLURM_RESTART_COUNT="1")
        self.assertEqual(code, 10, output)
        self.assertIn("job path already exists", output)
        self.assertEqual(old.read_text(), "previous attempt")

    def test_invalid_restart_count_is_rejected(self):
        for bad in ("-1", "x", "01", ""):
            if bad == "":
                continue
            code, output = self.run_job("success", SLURM_RESTART_COUNT=bad)
            self.assertEqual(code, 10, output)
            self.assertIn("SLURM_RESTART_COUNT", output)

    def test_sbatch_names_every_attempt_path_by_run_id(self):
        sb = (Path(ROOT) / "slurm/smoke_granite_v2.sbatch").read_text()
        self.assertIn("SLURM_RESTART_COUNT", sb)
        self.assertIn("--open-mode=append", sb)                          # Slurm's own output file keeps earlier attempts
        self.assertNotRegex(sb, r"logs/[a-z-]+-\$\{JOB\}")               # no log path named by the job ID alone
        self.assertNotIn('smoke_${JOB}', sb)
        self.assertIn('benchson-smoke-$RUN_ID.XXXXXXXX', sb)
        self.assertIn('[[ ! -e "$path" && ! -L "$path" ]]', sb)           # path-existence check not weakened


if __name__ == "__main__":
    unittest.main()
