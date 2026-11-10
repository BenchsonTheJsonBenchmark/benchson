"""CPU-only tests: context-window guard, error classification, and what the checker does with the result."""
import contextlib, http.server, io, json, os, shutil, sys, tempfile, threading, types, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
from data_generation import anchor as A
from data_generation.benchmark_builder import BenchmarkBuilder
from llm.context_guard import ContextGuardProvider, ContextOverflow, TokenCountError, is_context_overflow
from llm.llm_provider import LLMProvider
from llm.mock.mock_provider import MockProvider
from llm.recording_provider import CallBudget, ProviderFailing, RecordingProvider
from src.llm.openai.openai_provider import OpenAIProvider
from test_anchor_pool import _builder_config, _schema, write_baseline


class Echo(LLMProvider):
    def __init__(self):
        super().__init__(); self.model = "echo"; self.seen = []

    def _generate(self, messages, parameters=None):
        self.seen.append((messages, parameters)); return "{}"


class FixedCounter:
    def __init__(self, tokens, limit=None):
        self.tokens, self.limit, self.calls = tokens, limit, 0

    def count_prompt_tokens(self, messages):
        self.calls += 1
        if isinstance(self.tokens, Exception):
            raise self.tokens
        return self.tokens, self.limit


class GuardUnitTest(unittest.TestCase):
    def guard(self, tokens, limit=None, max_tokens=4096):
        inner = Echo()
        return inner, ContextGuardProvider(inner, FixedCounter(tokens, limit), 8192, max_tokens)

    def test_exact_boundary_matches_the_server_rule(self):
        inner, g = self.guard(4096)                       # 4096 + 4096 == 8192: the server accepts it
        self.assertEqual(g.generate([{"role": "user", "content": "x"}])[0], "{}")
        inner, g = self.guard(4097)                       # the pilot error: "prompt contains at least 4097 tokens"
        with self.assertRaises(ContextOverflow) as ctx:
            g.generate([{"role": "user", "content": "x"}])
        self.assertTrue(is_context_overflow(ctx.exception)); self.assertEqual(inner.seen, [])   # never sent
        self.assertEqual((ctx.exception.prompt_tokens, ctx.exception.max_tokens, ctx.exception.max_model_len), (4097, 4096, 8192))
        self.assertIn("context_overflow", str(ctx.exception))

    def test_requests_that_fit_are_forwarded_unchanged(self):
        inner, g = self.guard(100)
        msgs, params = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], {"seed": 3}
        g.generate(msgs, params)
        self.assertEqual(inner.seen, [(msgs, params)])
        self.assertEqual(g.stats["checked"], 1); self.assertEqual(g.stats["overflow"], 0)

    def test_overflow_is_neither_a_model_call_nor_a_failed_call(self):
        log = os.path.join(tempfile.mkdtemp(), "calls.jsonl"); self.addCleanup(shutil.rmtree, os.path.dirname(log))
        budget, echo = CallBudget(5), Echo()
        g = ContextGuardProvider(RecordingProvider(echo, log, "builder", budget), FixedCounter(5000), 8192, 4096)
        for _ in range(10):                               # more refusals than the consecutive-error limit
            with self.assertRaises(ContextOverflow):
                g.generate([{"role": "user", "content": "x"}])
        self.assertEqual((budget.used, budget.consecutive_errors), (0, 0)); self.assertFalse(os.path.exists(log))

    def test_window_mismatch_and_broken_counting_stop_the_run(self):
        _, g = self.guard(10, limit=16384)
        with self.assertRaises(ProviderFailing):
            g.generate([{"role": "user", "content": "x"}])
        _, g = self.guard(RuntimeError("tokenize down"))
        for _ in range(2):
            with self.assertRaises(TokenCountError):      # reported (typed), never silently skipped
                g.generate([{"role": "user", "content": "x"}])
        with self.assertRaises(ProviderFailing):
            g.generate([{"role": "user", "content": "x"}])

    def test_configuration_is_validated(self):
        for bad in ((0, 10), (8192, 8192), (8192, None), (8192, True)):
            with self.assertRaises(ValueError):
                ContextGuardProvider(Echo(), FixedCounter(1), bad[0], bad[1])


class TokenizeEndpointTest(unittest.TestCase):
    def test_count_uses_the_servers_tokenizer_with_auth_and_same_filtering(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                data = json.dumps({"count": 4321, "max_model_len": 8192, "tokens": []}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.client = types.SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}/v1/", api_key="k3y")
        provider.model, provider.drop_empty_assistant_turn = "ibm/granite", True
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": ""}]
        self.assertEqual(provider.count_prompt_tokens(msgs), (4321, 8192))
        self.assertEqual(seen["path"], "/tokenize"); self.assertEqual(seen["auth"], "Bearer k3y")
        self.assertEqual(seen["body"]["messages"], msgs[:2]); self.assertEqual(seen["body"]["model"], "ibm/granite")


class ByteCounter:
    """Stand-in tokenizer: one token per 3 characters."""
    def __init__(self):
        self.max_seen = 0

    def count_prompt_tokens(self, messages):
        return sum(len(m["content"]) for m in messages if m.get("content")) // 3, None


class OverflowInTheBuilderTest(unittest.TestCase):
    """Mock-model runs where one schema's own prompt cannot fit and the others can."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        train = {f"t{i}.json": _schema(f"t{i}") for i in range(14)}
        train["huge.json"] = _schema("huge", pad=9000)       # the guard window below is sized to refuse only this one
        test = {f"e{i}.json": _schema(f"e{i}") for i in range(3)}
        cls.base = os.path.join(cls.tmp.name, "pub"); os.makedirs(cls.base)
        write_baseline(cls.base, train, test); A.write_anchor(cls.base)
        cfg = _builder_config(cls.tmp.name, cls.base, max_new_references_per_schema=2)
        cfg["counts"] = {"train": {"create": 8, "fix": 8, "modify": 8}}
        cls.cfg = cfg
        cls.out, cls.calls = os.path.join(cls.tmp.name, "out"), os.path.join(cls.tmp.name, "calls.jsonl")
        cfg["output_root"] = cls.out
        mock = MockProvider()
        budget = CallBudget(2000)
        rec = RecordingProvider(mock, cls.calls, "builder", budget)
        cls.guard = ContextGuardProvider(rec, ByteCounter(), 4096 + 1000, 4096)   # prompt must be <= 1000 "tokens" = 3000 chars
        cls.builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.builder.build(cls.guard, None)
        cls.summary = cls.builder.last_summary

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_overflow_is_classified_separately_and_never_a_worker_error(self):
        reasons = self.summary["failure_reasons"]
        self.assertTrue(any(k.endswith("/context_overflow") for k in reasons), reasons)
        self.assertFalse(any("worker_error" in k for k in reasons), reasons)
        self.assertFalse(any(e["kind"] == "provider_error" for e in self.builder._events))
        self.assertTrue(any(e["kind"] == "context_overflow" for e in self.builder._events))
        ex = self.summary["capacity_exclusions"]["context_overflow"]
        self.assertTrue(ex and all("huge" in x["schema_id"] for x in ex))
        self.assertEqual(ex[0]["detail"]["max_model_len"], 5096)

    def test_the_unfittable_schema_is_retired_without_retrying(self):
        rows = {json.loads(l)["schema_id"]: json.loads(l) for l in
                open(os.path.join(self.summary["outputs"]["report_dir"], "schema_ledger.jsonl"))}
        huge = next(v for k, v in rows.items() if "huge" in k)
        self.assertEqual(huge["status"], "retired"); self.assertIn("context_overflow", huge["reason"])
        self.assertEqual(huge["attempts"], 1)                 # one look, no retry of an identical request
        self.assertEqual(huge["accepted_references"], 0)

    def test_nothing_was_sent_for_the_refused_requests_and_no_call_failed(self):
        calls = [json.loads(l) for l in open(self.calls)]
        self.assertTrue(calls and all(c["error"] is None for c in calls))
        self.assertFalse(any("x" * 8000 in json.dumps(c["messages"]) for c in calls))   # the huge schema never reached the model

    def test_other_schemas_still_reach_the_targets(self):
        self.assertEqual({t: v["standard"] for t, v in self.summary["produced_on_disk"]["train"].items()},
                         {"create": 8, "fix": 8, "modify": 8})

    def test_checker_stays_strict_about_provider_errors_but_not_about_refusals(self):
        out = self.cfg["output_root"]
        report = self.summary["outputs"]["report_dir"]
        events = os.path.join(report, "generation_events.jsonl")
        r = csr.check(out, per_task=8, calls_log=self.calls)
        self.assertFalse(any("context_overflow" in t for t in r["technical"]), r["technical"])
        self.assertTrue(any("context_overflow" in w for w in r["warnings"]), r["warnings"])
        work = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, work)
        copy = os.path.join(work, "o"); shutil.copytree(out, copy)
        ev = os.path.join(copy, os.path.relpath(events, out))
        with open(ev, "a") as f:
            f.write(json.dumps({"kind": "provider_error", "schema_id": "s", "detail": "BadRequestError: boom"}) + "\n")
        r = csr.check(copy, per_task=8, calls_log=self.calls)
        self.assertEqual(r["verdict"], "TECHNICAL_FAILURE")
        bad_calls = os.path.join(work, "calls.jsonl")
        open(bad_calls, "w").write(open(self.calls).read() + json.dumps(
            {"call": 99999, "error": "BadRequestError: Maximum context length is 8192 tokens", "messages": []}) + "\n")
        r = csr.check(out, per_task=8, calls_log=bad_calls)   # a server-side overflow that slipped through is still technical
        self.assertTrue(any("model calls failed" in t for t in r["technical"]))

    def test_without_the_guard_the_same_request_is_a_provider_error(self):
        """A server-side overflow (guard off or missed) keeps its technical classification."""
        class Overflowing(LLMProvider):
            def __init__(self):
                super().__init__(); self.model = "o"

            def _generate(self, messages, parameters=None):
                raise RuntimeError("Maximum context length is 8192 tokens. Requested output tokens: 4096.")

        b = BenchmarkBuilder({**self.cfg, "output_root": os.path.join(self.tmp.name, "noguard")})
        with self.assertRaises(RuntimeError):
            b._process_schema(_schema("x"), "jsb", "x", "x.json", {"create": "standard"}, Overflowing(),
                              __import__("random").Random(0), b.gates_by_split["train"], "train", "0-train-x-a1", "x__a1")
        self.assertEqual([e["kind"] for e in b._events], ["provider_error"])


class AnalyzeFailuresTest(unittest.TestCase):
    def test_failed_calls_are_classified_from_the_log(self):
        import analyze_context_failures as acf
        err = "Error: Maximum context length is 8192 tokens. Requested output tokens: 4096. Prompt contains at least 4097 input tokens."
        first = [{"role": "system", "content": "s"}, {"role": "user", "content": "Generate a JSON object that conforms to the following schema: {}"}]
        repair = first + [{"role": "assistant", "content": "x" * 50}, {"role": "user", "content": "The JSON is invalid: boom. Fix it"}]
        rows = [{"call": 1, "error": err, "messages": first}, {"call": 2, "error": err, "messages": repair},
                {"call": 3, "error": None, "messages": first}, {"call": 4, "error": "Timeout", "messages": first}]
        path = os.path.join(tempfile.mkdtemp(), "c.jsonl"); self.addCleanup(shutil.rmtree, os.path.dirname(path))
        open(path, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
        out = acf.analyze(path)
        self.assertEqual([r["call"] for r in out], [1, 2])
        self.assertTrue(out[0]["kind"].startswith("initial_schema_prompt")); self.assertTrue(out[1]["kind"].startswith("repair_prompt"))
        self.assertEqual(out[0]["reported_prompt_tokens_at_least"], 4097); self.assertEqual(out[0]["allowed_prompt_tokens"], 4096)


class GuardWiringTest(unittest.TestCase):
    def test_guard_needs_a_provider_that_can_count_tokens(self):
        from build_benchmark import main as build_main
        tmp = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, tmp)
        base = os.path.join(tmp, "pub"); os.makedirs(base)
        write_baseline(base, {"t.json": _schema("t")}, {"e.json": _schema("e")}); A.write_anchor(base)
        cfg = _builder_config(tmp, base)
        path = os.path.join(tmp, "c.json"); json.dump(cfg, open(path, "w"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            build_main(["--config", path, "--smoke-total", "3", "--out-dir", os.path.join(tmp, "o"), "--context-guard", "8192"])

    def test_chunk_validation_requires_the_guard_to_have_run(self):
        import chunked_run as cr
        from test_chunking import ChunkBase
        case = ChunkBase("run_chunk"); case.setUp(); self.addCleanup(case.tmp.cleanup)
        out = case.run_chunk(1, "c01")
        self.assertEqual(cr.validate_chunk(out, case.anchor_file)["code"], 0)                       # not required: fine
        res = cr.validate_chunk(out, case.anchor_file, require_context_guard=8192)                 # mock run had no guard
        self.assertEqual(res["code"], 3); self.assertTrue(any("context guard" in t for t in res["technical"]))
        mf = os.path.join(out, "generation_manifest.json"); m = json.load(open(mf))
        m["context_guard"] = {"enabled": True, "max_model_len": 8192, "checked": 120, "overflow": 2}
        json.dump(m, open(mf, "w"))
        res = cr.validate_chunk(out, case.anchor_file, require_context_guard=8192)
        self.assertEqual(res["code"], 0, res); self.assertTrue(any("refused 2" in w for w in res["warnings"]))
        m["context_guard"]["max_model_len"] = 16384; json.dump(m, open(mf, "w"))
        self.assertEqual(cr.validate_chunk(out, case.anchor_file, require_context_guard=8192)["code"], 3)


if __name__ == "__main__":
    unittest.main()
