"""CPU-only tests: a timed-out /tokenize request (token count) is a typed, bounded, tolerated-in-small-numbers event that is
never confused with a model-generation failure; everything else stays a technical failure."""
import contextlib, http.server, io, json, os, shutil, socket, sys, tempfile, threading, time, types, unittest, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
from data_generation import anchor as A
from data_generation.benchmark_builder import BenchmarkBuilder
from llm.context_guard import ContextGuardProvider, TokenCountError, is_count_timeout, is_token_count_error
from llm.llm_provider import LLMProvider
from llm.mock.mock_provider import MockProvider
from llm.recording_provider import CallBudget, ProviderFailing, RecordingProvider
from src.llm.openai.openai_provider import OpenAIProvider
from test_anchor_pool import _builder_config, _schema, write_baseline
from test_context_guard import ByteCounter, Echo

MSG = [{"role": "user", "content": "x"}]


class Scripted:
    """Counter that raises the scripted errors in order, then answers (tokens, None)."""
    def __init__(self, *script, tokens=10):
        self.script, self.tokens, self.calls = list(script), tokens, 0

    def count_prompt_tokens(self, messages):
        self.calls += 1
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
        return self.tokens, None


def guard(counter, inner=None, attempts=3):
    sleeps = []
    inner = inner or Echo()
    return inner, ContextGuardProvider(inner, counter, 8192, 4096, count_attempts=attempts, sleep=sleeps.append), sleeps


class ClassificationTest(unittest.TestCase):
    def test_only_a_timeout_of_the_count_request_is_a_count_timeout(self):
        self.assertTrue(is_count_timeout(TimeoutError("timed out")))
        self.assertTrue(is_count_timeout(socket.timeout("timed out")))
        self.assertTrue(is_count_timeout(urllib.error.URLError(TimeoutError("timed out"))))
        for other in (RuntimeError("timed out"), ConnectionError("reset"), urllib.error.URLError(ConnectionRefusedError()),
                      ValueError("bad json"), KeyError("count"), urllib.error.HTTPError("u", 500, "boom", {}, None)):
            self.assertFalse(is_count_timeout(other), other)

    def test_the_typed_error_is_not_a_context_overflow_and_keeps_its_cause(self):
        _, g, _ = guard(Scripted(*[TimeoutError("timed out")] * 3))
        with self.assertRaises(TokenCountError) as ctx:
            g.generate(MSG)
        e = ctx.exception
        self.assertTrue(is_token_count_error(e)); self.assertEqual((e.kind, e.attempts), ("timeout", 3))
        self.assertIsInstance(e.__cause__, TimeoutError); self.assertIn("token_count_timeout", str(e))
        self.assertFalse(getattr(e, "is_context_overflow", False))


class RetryTest(unittest.TestCase):
    def test_a_timeout_is_retried_with_backoff_and_then_succeeds(self):
        counter = Scripted(TimeoutError("timed out"), TimeoutError("timed out"))
        inner, g, sleeps = guard(counter)
        self.assertEqual(g.generate(MSG)[0], "{}")
        self.assertEqual(counter.calls, 3); self.assertEqual(sleeps, [2.0, 5.0]); self.assertEqual(len(inner.seen), 1)
        self.assertEqual((g.stats["checked"], g.stats["count_retries"], g.stats["count_timeouts"]), (1, 2, 0))

    def test_exhaustion_is_bounded_and_nothing_is_sent(self):
        counter = Scripted(*[TimeoutError("timed out")] * 10)
        inner, g, sleeps = guard(counter)
        with self.assertRaises(TokenCountError) as ctx:
            g.generate(MSG)
        self.assertEqual(ctx.exception.kind, "timeout"); self.assertEqual(counter.calls, 3)   # never more than the bound
        self.assertEqual(inner.seen, []); self.assertEqual(sleeps, [2.0, 5.0])
        self.assertEqual((g.stats["checked"], g.stats["count_timeouts"], g.stats["count_retries"]), (0, 1, 2))

    def test_a_non_timeout_count_error_is_not_retried_and_is_a_technical_kind(self):
        counter = Scripted(ConnectionError("reset"), RuntimeError("never reached"))
        inner, g, sleeps = guard(counter)
        with self.assertRaises(TokenCountError) as ctx:
            g.generate(MSG)
        self.assertEqual((ctx.exception.kind, ctx.exception.attempts), ("error", 1)); self.assertEqual(counter.calls, 1)
        self.assertEqual(sleeps, []); self.assertEqual(g.stats["count_errors"], 1)

    def test_repeated_exhausted_counts_stop_the_run_fail_closed_and_a_success_resets(self):
        counter = Scripted(*[TimeoutError("timed out")] * 6, tokens=5)
        _, g, _ = guard(counter, attempts=1)
        for _ in range(2):
            with self.assertRaises(TokenCountError):
                g.generate(MSG)
        with self.assertRaises(ProviderFailing):                       # the third in a row: the guard cannot protect the run
            g.generate(MSG)
        counter = Scripted(TimeoutError("t"), TimeoutError("t"), None, TimeoutError("t"), TimeoutError("t"), tokens=5)
        _, g, _ = guard(counter, attempts=1)
        for expect_error in (True, True, False, True, True):
            if expect_error:
                with self.assertRaises(TokenCountError):
                    g.generate(MSG)
            else:
                g.generate(MSG)                                         # a success in between resets the streak


class NotAModelFailureTest(unittest.TestCase):
    def test_a_model_generation_timeout_passes_through_untouched_and_is_never_retried_or_retyped(self):
        class Slow(Echo):
            def _generate(self, messages, parameters=None):
                self.seen.append(messages); raise TimeoutError("Request timed out.")
        inner = Slow(); counter = Scripted()
        g = ContextGuardProvider(inner, counter, 8192, 4096, sleep=lambda s: self.fail("slept"))
        with self.assertRaises(TimeoutError) as ctx:
            g.generate(MSG)
        self.assertNotIsInstance(ctx.exception, TokenCountError)
        self.assertEqual((counter.calls, len(inner.seen)), (1, 1))      # the count ran once, the model once: no retry here
        self.assertEqual(g.stats["count_timeouts"], 0)

    def test_count_failures_are_not_model_calls_and_never_touch_the_budget_or_the_calls_log(self):
        log = os.path.join(tempfile.mkdtemp(), "calls.jsonl"); self.addCleanup(shutil.rmtree, os.path.dirname(log))
        budget = CallBudget(5)
        g = ContextGuardProvider(RecordingProvider(Echo(), log, "builder", budget), Scripted(*[TimeoutError("t")] * 3), 8192, 4096,
                                 sleep=lambda s: None)
        with self.assertRaises(TokenCountError):
            g.generate(MSG)
        self.assertEqual(budget.used, 0); self.assertEqual(budget.consecutive_errors, 0)
        self.assertFalse(os.path.exists(log) and open(log).read().strip())
        g.generate(MSG)                                                  # the next request counts and calls normally
        self.assertEqual(budget.used, 1); self.assertEqual(len(open(log).read().splitlines()), 1)

    def test_a_real_model_timeout_is_logged_as_a_failed_call_and_counts_towards_the_budget(self):
        class Slow(LLMProvider):
            model = "slow"
            def _generate(self, messages, parameters=None):
                raise TimeoutError("Request timed out.")
        log = os.path.join(tempfile.mkdtemp(), "calls.jsonl"); self.addCleanup(shutil.rmtree, os.path.dirname(log))
        budget = CallBudget(5)
        g = ContextGuardProvider(RecordingProvider(Slow(), log, "builder", budget), Scripted(), 8192, 4096)
        with self.assertRaises(TimeoutError):
            g.generate(MSG)
        rows = [json.loads(l) for l in open(log)]
        self.assertEqual((budget.used, budget.consecutive_errors, len(rows)), (1, 1, 1)); self.assertTrue(rows[0]["error"])


class CountRequestTimeoutTest(unittest.TestCase):
    def test_the_count_request_has_its_own_timeout_and_really_times_out(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(1.5)
            def log_message(self, *a):
                pass
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        self.assertEqual(OpenAIProvider.COUNT_TIMEOUT_SECONDS, 60)
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.client = types.SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}/v1/", api_key="k")
        provider.model, provider.drop_empty_assistant_turn = "m", True
        provider.COUNT_TIMEOUT_SECONDS = 0.3
        t0 = time.time()
        with self.assertRaises(Exception) as ctx:
            provider.count_prompt_tokens(MSG)
        self.assertTrue(is_count_timeout(ctx.exception), repr(ctx.exception)); self.assertLess(time.time() - t0, 1.4)


class TimeoutInTheBuilderTest(unittest.TestCase):
    """Mock-model run where the token count of ONE schema's prompts always times out."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        train = {f"t{i}.json": _schema(f"t{i}") for i in range(14)}
        train["slow.json"] = _schema("slow")
        test = {f"e{i}.json": _schema(f"e{i}") for i in range(3)}
        cls.base = os.path.join(cls.tmp.name, "pub"); os.makedirs(cls.base)
        write_baseline(cls.base, train, test); A.write_anchor(cls.base)
        cfg = _builder_config(cls.tmp.name, cls.base, max_new_references_per_schema=2)
        cfg["counts"] = {"train": {"create": 8, "fix": 8, "modify": 8}}
        cls.out, cls.calls = os.path.join(cls.tmp.name, "out"), os.path.join(cls.tmp.name, "calls.jsonl")
        cfg["output_root"] = cls.out

        class SlowCounter(ByteCounter):
            def count_prompt_tokens(self, messages):
                if "f_slow" in json.dumps(messages):
                    raise TimeoutError("timed out")
                return super().count_prompt_tokens(messages)

        rec = RecordingProvider(MockProvider(), cls.calls, "builder", CallBudget(2000))
        cls.guard = ContextGuardProvider(rec, SlowCounter(), 8192, 4096, count_attempts=2, sleep=lambda s: None,
                                         max_consecutive_count_errors=1000)
        cls.builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.builder.build(cls.guard, None)
        cls.summary = cls.builder.last_summary
        cls.report = cls.summary["outputs"]["report_dir"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_failed_attempt_has_its_own_stage_and_is_never_a_worker_error_or_provider_error(self):
        reasons = self.summary["failure_reasons"]
        self.assertTrue(any(k.endswith("/token_count_timeout") for k in reasons), reasons)
        self.assertFalse(any("worker_error" in k for k in reasons), reasons)
        kinds = {e["kind"] for e in self.builder._events}
        self.assertIn("token_count_timeout", kinds); self.assertNotIn("provider_error", kinds)
        self.assertTrue(all("slow" in e["schema_id"] for e in self.builder._events if e["kind"] == "token_count_timeout"))

    def test_a_failed_attempt_leaves_no_example_no_sidecar_and_no_call(self):
        slow_ids = {e["schema_id"] for e in self.builder._events if e["kind"] == "token_count_timeout"}
        self.assertTrue(slow_ids)
        for root, _, names in os.walk(self.out):
            for n in names:
                self.assertFalse(any(n.startswith(i) for i in slow_ids), os.path.join(root, n))
        calls = [json.loads(l) for l in open(self.calls)]
        self.assertTrue(calls and all(c["error"] is None for c in calls))
        self.assertFalse(any("f_slow" in json.dumps(c["messages"]) for c in calls))

    def test_the_ledger_counts_every_failed_attempt(self):
        rows = {json.loads(l)["schema_id"]: json.loads(l) for l in open(os.path.join(self.report, "schema_ledger.jsonl"))}
        slow = next(v for k, v in rows.items() if "slow" in k)
        n_events = sum(1 for e in self.builder._events if e["kind"] == "token_count_timeout")
        self.assertEqual(slow["accepted_references"], 0)
        self.assertEqual(slow["failed_attempts"], n_events); self.assertGreaterEqual(slow["attempts"], slow["failed_attempts"])
        failures = [json.loads(l) for l in open(os.path.join(self.report, "generation_failures.jsonl"))]
        self.assertEqual(len([f for f in failures if f["stage"] == "token_count_timeout"]), 3 * n_events)

    def test_other_schemas_still_reach_the_targets_and_the_guard_accounts_for_it(self):
        self.assertEqual({t: v["standard"] for t, v in self.summary["produced_on_disk"]["train"].items()},
                         {"create": 8, "fix": 8, "modify": 8})
        n_events = sum(1 for e in self.builder._events if e["kind"] == "token_count_timeout")
        self.assertEqual(self.guard.stats["count_timeouts"], n_events); self.assertGreater(self.guard.stats["count_retries"], 0)
        self.assertEqual(self.guard.stats["checked"], len(open(self.calls).read().splitlines()) + self.guard.stats["overflow"])


class ModelFailureStaysTechnicalInTheBuilderTest(unittest.TestCase):
    def test_a_model_timeout_in_a_worker_is_a_worker_error_and_a_provider_error_event(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        train = {f"t{i}.json": _schema(f"t{i}") for i in range(6)}
        base = os.path.join(tmp.name, "pub"); os.makedirs(base)
        write_baseline(base, train, {f"e{i}.json": _schema(f"e{i}") for i in range(2)}); A.write_anchor(base)
        cfg = _builder_config(tmp.name, base, max_new_references_per_schema=1)
        cfg["counts"] = {"train": {"create": 2, "fix": 2, "modify": 2}}

        class Timeouts(MockProvider):
            def _generate(self, messages, parameters=None):
                raise TimeoutError("timed out")                          # the very same text, but from the MODEL call
        g = ContextGuardProvider(Timeouts(), ByteCounter(), 8192, 4096, sleep=lambda s: None)
        builder = BenchmarkBuilder(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                builder.build(g, None)
            except BaseException:
                pass                                                     # ProviderFailing after 3 consecutive model errors is fine
        kinds = {e["kind"] for e in builder._events}
        self.assertIn("provider_error", kinds); self.assertFalse(any(k.startswith("token_count") for k in kinds))


def make_out(tmp, events=(), failures=(), guard=None, examples=(), sidecars=()):
    out = os.path.join(tmp, "out"); rd = os.path.join(out, "_generation_reports", "20260101_000000"); os.makedirs(rd)
    json.dump({"status": "completed", "finished": True, "config": {}, **({"context_guard": guard} if guard is not None else {})},
              open(os.path.join(out, "generation_manifest.json"), "w"))
    json.dump({"completed": True}, open(os.path.join(rd, "generation_summary.json"), "w"))
    for name, rows in (("generation_events.jsonl", events), ("generation_failures.jsonl", failures)):
        open(os.path.join(rd, name), "w").write("".join(json.dumps(r) + "\n" for r in rows))
    for t in csr.TASKS:
        os.makedirs(os.path.join(out, csr.DATASET_DIRS[t], "train")); os.makedirs(os.path.join(out, csr.DATASET_DIRS[t], "_metadata", "train"))
    for rel in examples + sidecars:
        open(os.path.join(out, rel), "w").write("{}")
    return out


def timeout_rows(ids):
    ev = [{"split": "train", "schema_id": i, "kind": "token_count_timeout", "detail": "x"} for i in ids]
    fl = [{"task": t, "mode": "standard", "schema_id": i, "stage": "token_count_timeout", "detail": "x"} for i in ids for t in csr.TASKS]
    return ev, fl


class CheckerToleranceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def check(self, **kw):
        return csr.check(make_out(self.tmp.name, **kw), per_task=0)

    def test_a_small_fully_accounted_number_of_timeouts_is_tolerated_and_reported(self):
        ev, fl = timeout_rows(["s1__a1"])
        res = self.check(events=ev, failures=fl, guard={"checked": 500, "count_timeouts": 1})
        self.assertEqual(res["technical"], [], res); self.assertEqual(res["tolerated"]["token_count_timeouts"]["count"], 1)
        self.assertTrue(any("token_count_timeout" in w for w in res["warnings"]))

    def test_default_is_strict_for_everything_else(self):
        for event, stage in (("token_count_error", "token_count_error"), ("provider_error", "worker_error")):
            ev = [{"split": "train", "schema_id": "s1__a1", "kind": event, "detail": "TimeoutError: timed out"}]
            fl = [{"task": "create", "mode": "standard", "schema_id": "s1__a1", "stage": stage, "detail": "timed out"}]
            res = self.check(events=ev, failures=fl, guard={"checked": 500, "count_timeouts": 0})
            self.assertTrue(res["technical"] and not res["tolerated"], (event, res)); self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory()

    def test_the_old_historical_shape_is_not_tolerated_without_the_explicit_policy(self):
        ev = [{"split": "train", "schema_id": "s1__a1", "kind": "provider_error", "detail": "TimeoutError: timed out"}]
        fl = [{"task": t, "mode": "standard", "schema_id": "s1__a1", "stage": "worker_error", "detail": "timed out"} for t in csr.TASKS]
        res = self.check(events=ev, failures=fl, guard={"checked": 500})
        self.assertEqual(res["code"], 3)

    def test_exceeding_the_count_limit_or_the_rate_is_technical(self):
        ids = [f"s{i}__a1" for i in range(csr.MAX_TOLERATED_TOKEN_COUNT_TIMEOUTS + 1)]
        ev, fl = timeout_rows(ids)
        res = self.check(events=ev, failures=fl, guard={"checked": 100000, "count_timeouts": len(ids)})
        self.assertTrue(any("exceed the limit" in t for t in res["technical"]), res)
        self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory()
        ev, fl = timeout_rows(["a__a1", "b__a1", "c__a1"])
        res = self.check(events=ev, failures=fl, guard={"checked": 100, "count_timeouts": 3})        # 3/103 > 1 %
        self.assertTrue(any("rate" in t for t in res["technical"]), res)
        self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory()
        ev, fl = timeout_rows(["a__a1"])
        self.assertEqual(self.check(events=ev, failures=fl, guard={"checked": 100, "count_timeouts": 1})["technical"], [])   # 1/101 < 1 %

    def test_unaccounted_timeouts_are_technical(self):
        ev, fl = timeout_rows(["s1__a1", "s2__a1"])
        cases = {"manifest has no guard counters": dict(guard=None),
                 "manifest counts another number": dict(guard={"checked": 500, "count_timeouts": 1}),
                 "event without failure": dict(guard={"checked": 500, "count_timeouts": 2}, failures=fl[:3])}
        for why, kw in cases.items():
            self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory()
            args = dict(events=ev, failures=fl); args.update(kw)
            res = self.check(**args); self.assertTrue(res["technical"], why); self.assertFalse(res["tolerated"], why)

    def test_a_failed_attempt_with_an_example_or_sidecar_is_technical(self):
        ev, fl = timeout_rows(["s1__a1"])
        for kw in (dict(examples=("benchmark_create/train/s1__a1.json",)), dict(sidecars=("benchmark_fix/_metadata/train/s1__a1.json",))):
            self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory()
            res = self.check(events=ev, failures=fl, guard={"checked": 500, "count_timeouts": 1}, **kw)
            self.assertTrue(any("although attempt" in t for t in res["technical"]), res)


if __name__ == "__main__":
    unittest.main()
