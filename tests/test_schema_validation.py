"""CPU-only tests: schema validation distinguishes VALID / INVALID / TECHNICAL, never treats a technical failure as proof of
invalidity or as a pass, and a transient failure never damages or re-writes a sealed chunk. No test uses the real internet:
external `$ref` documents are served by a local HTTP server."""
import contextlib, glob, hashlib, http.server, io, json, os, socket, subprocess, sys, tempfile, threading, time, unittest, warnings
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))

import check_smoke_result as csr
import chunked_run as cr
from data_generation import anchor as A
from data_generation import chunking
from data_generation.published import TASKS
from evaluations import metrics
from evaluations import schema_validation as sv
from test_anchor_pool import _builder_config, _schema, write_baseline
from test_chunking import ChunkBase, tree_hash

N1 = "c01_111-r0"
LOCAL = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


class RemoteServer:
    """Serves one JSON document per path; every behaviour can be changed while the test runs."""

    def __init__(self):
        self.docs, self.status, self.delay, self.raw, self.hits = {}, {}, {}, {}, {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits[self.path] = outer.hits.get(self.path, 0) + 1
                time.sleep(outer.delay.get(self.path, 0))
                code = outer.status.get(self.path, 200)
                body = outer.raw.get(self.path) or json.dumps(outer.docs.get(self.path, {})).encode()
                try:
                    self.send_response(code); self.end_headers(); self.wfile.write(body if code == 200 else b"x")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def url(self, path):
        return self.base + path

    def close(self):
        self.server.shutdown(); self.server.server_close()


class SvBase(unittest.TestCase):
    def setUp(self):
        self.srv = RemoteServer(); self.addCleanup(self.srv.close)
        self.sleeps = []
        for name, value in (("_sleep", self.sleeps.append), ("REMOTE_TIMEOUT_SECONDS", 1.0)):
            p = mock.patch.object(sv, name, value); p.start(); self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}); env.start(); self.addCleanup(env.stop)
        sv.reset_caches(); self.addCleanup(sv.reset_caches)

    def schema_with_ref(self, path="/doc", **extra):
        return {"type": "object", "properties": {"a": {"$ref": self.srv.url(path)}}, "required": ["a"], **extra}


class LocalSchemaTest(SvBase):
    def test_valid_instance_against_a_valid_local_schema(self):
        self.assertEqual(sv.validate_instance({"a": 1}, LOCAL), (sv.VALID, "valid"))

    def test_a_genuine_violation_is_invalid_with_the_message(self):
        state, detail = sv.validate_instance({"a": "x"}, LOCAL)
        self.assertEqual(state, sv.INVALID); self.assertIn("integer", detail)
        self.assertEqual(sv.validate_instance({}, LOCAL)[0], sv.INVALID)

    def test_same_verdict_as_the_evaluation_helper_for_decidable_cases(self):
        for obj in ({"a": 1}, {"a": "x"}, {}, [], None):
            old = metrics.schema_compliance(obj, LOCAL)[0]
            self.assertEqual(old == 1, sv.validate_instance(obj, LOCAL)[0] == sv.VALID, obj)

    def test_a_self_referencing_schema_resolves_locally_without_any_network(self):
        schema = {"$id": "https://example.invalid/self.json", "type": "object",
                  "properties": {"a": {"$ref": "https://example.invalid/self.json#/definitions/n"}},
                  "definitions": {"n": {"type": "integer"}}}
        with mock.patch.object(sv, "_fetch_once", side_effect=AssertionError("network used")):
            self.assertEqual(sv.validate_instance({"a": 1}, schema)[0], sv.VALID)
            self.assertEqual(sv.validate_instance({"a": "x"}, schema)[0], sv.INVALID)
            draft4 = {"$schema": "http://json-schema.org/draft-04/schema#", "id": "https://example.invalid/d4.json", "type": "object",
                      "properties": {"a": {"$ref": "https://example.invalid/d4.json#/definitions/n"}}, "definitions": {"n": {"type": "integer"}}}
            self.assertEqual(sv.validate_instance({"a": 1}, draft4)[0], sv.VALID)      # draft-04 "id" resolves locally too


class ExternalRefTest(SvBase):
    def test_a_resolved_external_ref_validates_and_is_fetched_once_per_process(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        schema = self.schema_with_ref()
        self.assertEqual(sv.validate_instance({"a": 1}, schema)[0], sv.VALID)
        self.assertEqual(sv.validate_instance({"a": "x"}, schema)[0], sv.INVALID)
        self.assertEqual(self.srv.hits["/doc"], 1)                       # one fixed copy of the document for the whole process

    def test_unavailable_ref_404_is_technical_and_not_retried(self):
        self.srv.status["/doc"] = 404
        state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref())
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("http_404", detail)
        self.assertEqual(self.srv.hits["/doc"], 1); self.assertEqual(self.sleeps, [])

    def test_unreachable_server_is_technical_after_a_bounded_number_of_tries(self):
        self.srv.close = lambda: None; self.srv.server.shutdown(); self.srv.server.server_close()    # nothing listens any more
        state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref())
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("connection", detail)
        self.assertEqual(self.sleeps, list(sv.REMOTE_BACKOFF_SECONDS[:sv.REMOTE_ATTEMPTS - 1]))

    def test_a_timing_out_ref_is_technical_bounded_and_never_a_pass(self):
        self.srv.delay["/doc"] = 2.0
        with mock.patch.object(sv, "REMOTE_TIMEOUT_SECONDS", 0.2):
            t0 = time.time(); state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref()); elapsed = time.time() - t0
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("timeout", detail)
        self.assertLess(elapsed, sv.REMOTE_ATTEMPTS * 0.2 + 2.5)         # at most 3 timeouts, never a hang
        self.assertEqual(self.srv.hits["/doc"], sv.REMOTE_ATTEMPTS)

    def test_a_transient_failure_is_retried_and_then_succeeds(self):
        self.srv.docs["/doc"] = {"type": "integer"}; self.srv.status["/doc"] = 503
        real = sv._fetch_once; calls = []
        def flaky(uri):
            calls.append(uri)
            if len(calls) == 3:
                self.srv.status["/doc"] = 200
            return real(uri)
        with mock.patch.object(sv, "_fetch_once", flaky):
            self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.VALID)
        self.assertEqual(len(calls), 3); self.assertEqual(self.sleeps, [1.0, 3.0])

    def test_a_failed_document_is_not_retried_for_a_while_and_is_retried_afterwards(self):
        self.srv.status["/doc"] = 500
        now = [1000.0]
        with mock.patch.object(sv, "_clock", lambda: now[0]):
            self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.TECHNICAL)
            hits = self.srv.hits["/doc"]; self.assertEqual(hits, sv.REMOTE_ATTEMPTS)
            self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.TECHNICAL)
            self.assertEqual(self.srv.hits["/doc"], hits)                  # an outage cannot stall every later validation
            self.srv.status["/doc"] = 200; self.srv.docs["/doc"] = {"type": "integer"}
            self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.TECHNICAL)   # still inside the memo window
            now[0] += sv.FAILURE_MEMO_SECONDS + 1
            self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.VALID)       # recovered, nothing stuck

    def test_malformed_unresolvable_or_hostile_references_are_technical(self):
        self.srv.raw["/html"] = b"<html>not json</html>"
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref("/html"))[0], sv.TECHNICAL)
        self.srv.raw["/big"] = b"[" + b"1," * 50 + b"1]"
        with mock.patch.object(sv, "MAX_REMOTE_BYTES", 20):
            state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref("/big"))
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("too_large", detail)
        for ref in ("file:///etc/passwd", "ftp://example.invalid/x", "other-file.json", "#/definitions/missing"):
            state, _ = sv.validate_instance({"a": 1}, {"type": "object", "properties": {"a": {"$ref": ref}}, "required": ["a"]})
            self.assertEqual(state, sv.TECHNICAL, ref)

    def test_malformed_schema_is_technical_not_invalid(self):
        for schema in ({"type": "nonsense"}, {"properties": {"a": {"minimum": "x"}}}, {"required": "a"}):
            self.assertEqual(sv.validate_instance({"a": 1}, schema)[0], sv.TECHNICAL, schema)

    def test_the_outcome_does_not_depend_on_which_keyword_is_visited_first(self):
        self.srv.status["/doc"] = 500
        broken_first = {"properties": {"a": {"$ref": self.srv.url("/doc")}, "b": {"type": "integer"}}}
        broken_last = {"properties": {"b": {"type": "integer"}, "a": {"$ref": self.srv.url("/doc")}}}
        for schema in (broken_first, broken_last):
            sv.reset_caches()
            self.assertEqual(sv.validate_instance({"a": 1, "b": "violation"}, schema)[0], sv.TECHNICAL)

    def test_repeated_and_cross_process_validation_is_consistent(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        results = {sv.validate_instance({"a": 1}, self.schema_with_ref()) for _ in range(5)}
        self.assertEqual(results, {(sv.VALID, "valid")})
        code = ("import sys, json, warnings; warnings.simplefilter('ignore'); sys.path.insert(0, %r)\n"
                "from evaluations import schema_validation as sv\n"
                "s = json.loads(sys.argv[1]); print(sv.validate_instance({'a': 1}, s)[0], sv.validate_instance({'a': 'x'}, s)[0])\n"
                % os.path.join(ROOT, "src"))
        env = {**os.environ, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
        outs = {subprocess.run([sys.executable, "-W", "ignore", "-c", code, json.dumps(self.schema_with_ref())], capture_output=True,
                               text=True, env=env).stdout.strip() for _ in range(2)}
        self.assertEqual(outs, {"valid invalid"})
        self.srv.status["/doc"] = 500; sv.reset_caches()
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.TECHNICAL)   # same inputs, no network: never "valid"

    def test_the_evaluation_helper_is_unchanged_it_still_cannot_tell_the_cases_apart(self):
        self.srv.status["/doc"] = 404
        score, message = metrics.schema_compliance({"a": 1}, self.schema_with_ref())
        self.assertEqual(score, 0); self.assertIn("schema/validation error", message)            # documents why the checker must not use it


class CheckerFindingsTest(SvBase):
    def fix_record(self, valid, erroneous, schema=None):
        return {"name": "x", "source": "s", "schema": schema or self.schema_with_ref(), "valid_json": valid, "erroneous_json": erroneous}

    def test_fix_requires_valid_json_valid_and_erroneous_json_invalid(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        self.assertEqual(csr.example_findings("fix", self.fix_record({"a": 1}, {"a": "x"})), ([], [], []))
        invalid, tech, _ = csr.example_findings("fix", self.fix_record({"a": "x"}, {"a": "y"}))
        self.assertEqual((len(invalid), tech), (1, [])); self.assertTrue(any("valid_json does not satisfy" in i for i in invalid))
        invalid, tech, _ = csr.example_findings("fix", self.fix_record({"a": 1}, {"a": 2}))
        self.assertTrue(any("actually valid" in i for i in invalid)); self.assertEqual(tech, [])

    def test_technical_error_on_valid_json_blocks_and_is_not_reported_as_invalid(self):
        self.srv.status["/doc"] = 500
        invalid, tech, _ = csr.example_findings("fix", self.fix_record({"a": 1}, {"a": "x"}))
        self.assertEqual(invalid, []); self.assertEqual(len(tech), 2)
        self.assertTrue(all("not evidence of invalidity" in t for t in tech))

    def test_technical_error_on_erroneous_json_is_not_evidence_that_it_is_invalid(self):
        # valid_json is decided locally (no ref involved); only erroneous_json touches the unreachable document
        schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"$ref": self.srv.url("/doc")}}, "required": ["a"]}
        self.srv.status["/doc"] = 500
        invalid, tech, _ = csr.example_findings("fix", self.fix_record({"a": 1}, {"a": 1, "b": 5}, schema))
        self.assertEqual(invalid, [])                                    # NOT "valid_json invalid", NOT silently accepted as broken
        self.assertEqual([t.split(" ")[0] for t in tech], ["erroneous_json"])
        sv.reset_caches()
        self.assertEqual(csr.example_problems("fix", self.fix_record({"a": 1}, {"a": 1, "b": 5}, schema))[0], tech)   # wrapper keeps the reasons

    def test_old_helper_would_have_accepted_that_erroneous_json(self):
        schema = {"type": "object", "properties": {"b": {"$ref": self.srv.url("/doc")}}}
        self.srv.status["/doc"] = 500
        self.assertEqual(metrics.schema_compliance({"b": 5}, schema)[0], 0)      # "invalid" - the false accept this change closes

    def test_create_and_modify_technical_results(self):
        self.srv.status["/doc"] = 500
        create = {"name": "x", "source": "s", "schema": self.schema_with_ref(), "description": "a is 1", "reference_json": {"a": 1}}
        invalid, tech, _ = csr.example_findings("create", create)
        self.assertEqual((invalid, len(tech)), ([], 1))
        modify = {"name": "x", "source": "s", "schema": self.schema_with_ref(), "data": {"a": 1}, "ground_truth": {"a": 2},
                  "instructions": "change a", "modification": {"op": "x"}}
        invalid, tech, warns = csr.example_findings("modify", modify)
        self.assertEqual(invalid, []); self.assertEqual([t.split(" ")[0] for t in tech], ["data"])
        self.assertTrue(any(w.startswith("ground_truth could not be validated") for w in warns))   # Modify's allowed-invalid side only warns

    def test_modify_ground_truth_may_still_violate_the_schema(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        modify = {"name": "x", "source": "s", "schema": self.schema_with_ref(), "data": {"a": 1}, "ground_truth": {"a": "x"},
                  "instructions": "change a", "modification": {"op": "x"}}
        invalid, tech, warns = csr.example_findings("modify", modify)
        self.assertEqual((invalid, tech), ([], [])); self.assertTrue(any("allowed by the original Modify task" in w for w in warns))


DEFAULT_TOTAL, DEFAULT_ATTEMPTS, DEFAULT_BACKOFF = sv.REMOTE_TOTAL_SECONDS, sv.REMOTE_ATTEMPTS, sv.REMOTE_BACKOFF_SECONDS
D4 = "http://json-schema.org/draft-04/schema#"
D7 = "http://json-schema.org/draft-07/schema#"


class DialectCompatibilityTest(SvBase):
    """On fully resolvable schemas the new validator must agree with the original `jsonschema.validate()` (the call the data was
    generated and checked with), including external documents that omit `$schema` or are written for older drafts."""
    DOCS = {
        "d4_noschema_excl": {"properties": {"n": {"type": "number", "minimum": 0, "exclusiveMinimum": True}}},
        "d4_schema_id_ref": {"$schema": D4, "id": "ID/root.json", "definitions": {"p": {"type": "integer"}},
                             "properties": {"n": {"$ref": "#/definitions/p"}}},
        "d4_format_required": {"$schema": D4, "required": ["n"], "properties": {"n": {"type": "integer", "maximum": 5, "exclusiveMaximum": True}}},
        "noschema_defs": {"definitions": {"p": {"type": "integer"}}, "properties": {"n": {"$ref": "#/definitions/p"}}},
        "tuple_items": {"items": [{"type": "integer"}, {"type": "string"}]},
        "dependencies": {"dependencies": {"n": ["m"]}},
        "d7_schema": {"$schema": D7, "properties": {"n": {"type": "integer"}}},
        "ref_siblings": {"definitions": {"p": {"type": "integer"}}, "properties": {"n": {"$ref": "#/definitions/p", "minimum": 10}}},
        "nested_remote_rel": {"properties": {"n": {"$ref": "other"}}},
        "type_union": {"properties": {"n": {"type": ["integer", "null"]}}},
    }
    INSTANCES = [{"n": 1}, {"n": 0}, {"n": 5}, {"n": "x"}, {"n": None}, {}, {"n": 12}, [1, "a"], [1, 2]]

    @staticmethod
    def stock(instance, schema):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import jsonschema
                jsonschema.validate(instance, schema); return sv.VALID
            except jsonschema.ValidationError:
                return sv.INVALID
            except Exception:
                return sv.TECHNICAL

    def test_same_verdict_as_the_original_validate_for_every_dialect_combination(self):
        for name, doc in self.DOCS.items():
            self.srv.docs["/" + name] = json.loads(json.dumps(doc).replace("ID/", self.srv.base + "/ids/"))
        self.srv.docs["/other"] = {"type": "integer", "minimum": 3}
        seen = set(); compared = 0
        for parent in ({}, {"$schema": D4}, {"$schema": D7}):
            for name in self.DOCS:
                schema = {**parent, "$ref": self.srv.url("/" + name)}
                for instance in self.INSTANCES:
                    sv.reset_caches(); expected = self.stock(instance, schema)
                    sv.reset_caches(); actual = sv.validate_instance(instance, schema)[0]
                    self.assertEqual(actual, expected, (parent, name, instance)); seen.add(expected); compared += 1
        self.assertEqual(compared, 3 * len(self.DOCS) * len(self.INSTANCES))
        self.assertTrue({sv.VALID, sv.INVALID} <= seen)                  # the comparison really exercised both outcomes

    def test_documents_without_schema_follow_the_referring_schema_exactly_like_the_original(self):
        # boolean exclusiveMinimum is a draft-04 idiom: its meaning comes from the validator class chosen by the REFERRING schema
        self.srv.docs["/d"] = self.DOCS["d4_noschema_excl"]
        for parent in ({"$schema": D4}, {"$schema": D7}, {}):
            schema = {**parent, "$ref": self.srv.url("/d")}
            for instance in ({"n": 0}, {"n": 1}, {"n": -1}):
                sv.reset_caches(); expected = self.stock(instance, schema); sv.reset_caches()
                self.assertEqual(sv.validate_instance(instance, schema)[0], expected, (parent, instance))

    def test_matches_the_evaluation_helper_on_published_schemas_that_need_no_network(self):
        files = sorted(glob.glob(os.path.join(ROOT, "data_published", "*", "benchmark_fix", "train", "*.json")))[:150]
        if not files:
            self.skipTest("published data not present")
        compared = 0
        for path in files:
            record = json.load(open(path))
            for field in ("valid_json", "erroneous_json"):
                state, _ = sv.validate_instance(record[field], record["schema"])
                if state == sv.TECHNICAL:
                    continue                                              # needs a remote document (offline here): decided elsewhere
                self.assertEqual(state == sv.VALID, metrics.schema_compliance(record[field], record["schema"])[0] == 1, path)
                compared += 1
        self.assertGreater(compared, 100)


class SlowServer:
    """Raw socket server that answers valid JSON but only a few bytes at a time (a 'slow loris' style response)."""

    def __init__(self, mode, interval=0.3, pieces=200):
        self.mode, self.interval, self.pieces = mode, interval, pieces
        self.sock = socket.socket(); self.sock.bind(("127.0.0.1", 0)); self.sock.listen(8)
        self.port = self.sock.getsockname()[1]; self.connections = 0
        self.stop, self.conns = threading.Event(), []
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1; self.conns.append(conn)
            threading.Thread(target=self.serve, args=(conn,), daemon=True).start()

    def serve(self, conn):
        try:
            conn.recv(4096)
            if self.mode == "body":
                conn.sendall(b"HTTP/1.0 200 OK\r\n\r\n{\"x\":[")
                for _ in range(self.pieces):
                    if self.stop.wait(self.interval):
                        break
                    conn.sendall(b"1,")
                conn.sendall(b"1]}")
            elif self.mode == "headers":
                conn.sendall(b"HTTP/1.0 200 OK\r\n")
                for i in range(90):
                    if self.stop.wait(self.interval):
                        break
                    conn.sendall(b"X-%d: y\r\n" % i)
                conn.sendall(b"\r\n{}")
            else:                                                         # "silent": accepts and never answers
                self.stop.wait(self.interval * self.pieces)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self.stop.set(); self.sock.close()
        for conn in self.conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class TotalTimeLimitTest(SvBase):
    """The per-operation socket timeout does not bound a response that keeps dripping; the total limit must."""

    def drip(self, mode):
        server = SlowServer(mode); self.addCleanup(server.close)
        sv_total = mock.patch.object(sv, "REMOTE_TOTAL_SECONDS", 1.0); sv_total.start(); self.addCleanup(sv_total.stop)
        sv_op = mock.patch.object(sv, "REMOTE_TIMEOUT_SECONDS", 5.0); sv_op.start(); self.addCleanup(sv_op.stop)  # a drip never trips this
        schema = {"type": "object", "properties": {"a": {"$ref": f"http://127.0.0.1:{server.port}/d"}}, "required": ["a"]}
        t0 = time.time(); result = sv.validate_instance({"a": 1}, schema); return result, time.time() - t0, server

    def test_a_dripping_body_is_cut_off_at_the_total_limit_and_is_technical(self):
        (state, detail), elapsed, server = self.drip("body")
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("timeout", detail)
        self.assertLess(elapsed, sv.REMOTE_ATTEMPTS * 1.0 + 3)           # 3 attempts x 1 s, not 3 x (200 x 0.3 s)
        self.assertEqual(server.connections, sv.REMOTE_ATTEMPTS)

    def test_dripping_headers_and_a_silent_server_are_cut_off_too(self):
        for mode in ("headers", "silent"):
            sv.reset_caches()
            (state, _), elapsed, _ = self.drip(mode)
            self.assertEqual(state, sv.TECHNICAL, mode); self.assertLess(elapsed, sv.REMOTE_ATTEMPTS * 1.0 + 3, mode)

    def test_a_response_that_is_slow_but_finishes_within_the_limit_is_accepted(self):
        server = SlowServer("body", interval=0.05, pieces=6); self.addCleanup(server.close)
        with mock.patch.object(sv, "REMOTE_TOTAL_SECONDS", 5.0):
            state, _ = sv.validate_instance({}, {"$ref": f"http://127.0.0.1:{server.port}/d"})
        self.assertEqual(state, sv.VALID)

    def test_a_giant_response_is_refused_by_size_before_it_can_run_long(self):
        self.srv.raw["/big"] = b"[" + b"1," * 200000 + b"1]"
        with mock.patch.object(sv, "MAX_REMOTE_BYTES", 1000):
            state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref("/big"))
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("too_large", detail); self.assertEqual(self.srv.hits["/big"], 1)


class RemoteContentDriftTest(SvBase):
    """What changes when the remote document changes between runs, and what is reported so that it can be noticed."""

    def test_documents_used_are_reported_with_their_sha256_and_never_persisted(self):
        body = json.dumps({"type": "integer"}).encode(); self.srv.raw["/doc"] = body
        self.assertEqual(sv.used_documents(), {})
        sv.validate_instance({"a": 1}, self.schema_with_ref())
        self.assertEqual(sv.used_documents(), {self.srv.url("/doc"): hashlib.sha256(body).hexdigest()})
        sv.reset_caches(); self.assertEqual(sv.used_documents(), {})

    def test_within_one_process_a_changing_document_cannot_change_the_verdict(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.VALID)
        self.srv.docs["/doc"] = {"type": "string"}                        # the server changes its mind mid-run
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.VALID)
        self.assertEqual(self.srv.hits["/doc"], 1)

    def test_across_runs_a_changed_document_changes_the_verdict_and_the_digest_shows_why(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref())[0], sv.VALID); first = sv.used_documents()
        sv.reset_caches()                                                  # a new process / a later run
        self.srv.docs["/doc"] = {"type": "string"}
        state, _ = sv.validate_instance({"a": 1}, self.schema_with_ref())
        self.assertEqual(state, sv.INVALID)                               # genuine, blocking: never silently accepted ...
        self.assertNotEqual(first, sv.used_documents())                   # ... and the drift is visible in the report


class RemoteRefChunkBase(ChunkBase):
    """A sealed chunk whose schemas all point at a document on the local server."""

    def setUp(self):
        self.srv = RemoteServer(); self.addCleanup(self.srv.close); self.srv.docs["/doc"] = {}
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}); env.start(); self.addCleanup(env.stop)
        sv.reset_caches(); self.addCleanup(sv.reset_caches)
        for name, value in (("_sleep", lambda s: None),):
            p = mock.patch.object(sv, name, value); p.start(); self.addCleanup(p.stop)
        super().setUp()
        def with_ref(tag):
            s = _schema(tag); s["properties"]["r"] = {"$ref": self.srv.url("/doc")}; s["required"] = [f"f_{tag}", "r"]; return s
        train = {f"t{i}.json": with_ref(f"t{i}") for i in range(self.N_TRAIN)}
        test = {f"e{i}.json": _schema(f"e{i}") for i in range(3)}
        shutil_rmtree = __import__("shutil").rmtree; shutil_rmtree(self.base); os.makedirs(self.base)
        write_baseline(self.base, train, test); A.write_anchor(self.base)
        sv.reset_caches()

    def validate(self, chunk, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return cr.validate_chunk(chunk, self.anchor_file, self.reserved, **kw)

    def sealed_chunk(self):
        out = self.run_chunk(1, N1)
        self.assertTrue(os.path.isfile(os.path.join(out, chunking.SEAL_FILE)))
        return out


class ChunkResolutionTest(RemoteRefChunkBase):
    def resolve(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cr.resolve_chunk(self.root, 1, self.anchor_file, self.reserved, self.CHUNKS, {t: self.TOTAL for t in TASKS})

    def test_chunk_is_accepted_when_every_reference_resolves(self):
        out = self.sealed_chunk(); sv.reset_caches()
        res = self.validate(out); self.assertEqual(res["code"], 0, res)
        self.assertGreater(self.srv.hits["/doc"], 0)                      # the check really went through the external reference
        self.assertEqual(list(res["remote_documents"]), [self.srv.url("/doc")])   # and the report names the document it used
        self.assertEqual(self.resolve()[:2], (0, N1))

    def test_unresolved_references_refuse_the_chunk_as_technical_without_touching_it_and_it_recovers_later(self):
        out = self.sealed_chunk(); before = tree_hash(out)
        for path in ("/doc",):
            self.srv.status[path] = 500
        sv.reset_caches()
        res = self.validate(out)
        self.assertEqual((res["code"], res["verdict"]), (3, "TECHNICAL_FAILURE")); self.assertEqual(res["invalid"], [])
        self.assertTrue(any("not evidence of invalidity" in t for t in res["technical"]))
        code, name, valid, ignored = self.resolve()
        self.assertEqual((code, name, valid), (cr.RESOLVE_NONE, None, []))
        self.assertIn("could not be validated", ignored[0][1])           # the reason is visible in the resolver output
        self.assertEqual(before, tree_hash(out))                         # seal, state, examples and reports are byte-identical
        self.srv.status["/doc"] = 200; sv.reset_caches()                  # the outage ends: nothing was damaged, nothing is re-sealed
        self.assertEqual(self.validate(out)["code"], 0); self.assertEqual(self.resolve()[:2], (0, N1)); self.assertEqual(before, tree_hash(out))

    def test_an_ancestor_with_unresolved_references_blocks_the_next_chunk_and_leaves_both_seals_alone(self):
        names = self.chain([N1, "c02_222-r0"]); a, b = (os.path.join(self.root, n) for n in names)
        before = (tree_hash(a), tree_hash(b))
        self.srv.status["/doc"] = 503; sv.reset_caches()
        res = self.validate(b)
        self.assertEqual(res["code"], 3); self.assertTrue(any(os.path.basename(a) in t for t in res["technical"]))
        self.assertEqual(before, (tree_hash(a), tree_hash(b)))

    def test_genuinely_invalid_examples_are_refused_as_invalid_not_technical(self):
        out = self.sealed_chunk(); before = tree_hash(out)
        self.srv.docs["/doc"] = {"not": {}}                              # the reference now rejects everything: a real violation
        sv.reset_caches()
        res = self.validate(out)
        self.assertEqual((res["code"], res["verdict"]), (4, "INVALID_OUTPUT")); self.assertEqual(res["technical"], [])
        self.assertTrue(any("valid_json does not satisfy the schema" in i for i in res["invalid"]))
        self.assertEqual(self.resolve()[0], cr.RESOLVE_NONE); self.assertEqual(before, tree_hash(out))

    def test_sealing_a_chunk_is_refused_while_a_reference_cannot_be_resolved(self):
        out = self.run_chunk(1, N1, seal=False)
        self.srv.status["/doc"] = 500; sv.reset_caches()
        res = cr.seal_chunk(out, self.anchor_file, self.reserved, self.calls_log, 0)
        self.assertEqual(res["code"], 3); self.assertFalse(os.path.exists(os.path.join(out, chunking.SEAL_FILE)))
        self.srv.status["/doc"] = 200; sv.reset_caches()
        res = cr.seal_chunk(out, self.anchor_file, self.reserved, self.calls_log, 0)
        self.assertEqual(res["code"], 0, res); self.assertTrue(res["sealed"])                   # then the normal seal works


class FailClosedTest(SvBase):
    def test_a_validator_crash_is_technical(self):
        with mock.patch.object(sv, "_validator", side_effect=RecursionError("deep")):
            self.assertEqual(sv.validate_instance({}, {})[0], sv.TECHNICAL)
        class Boom:
            def iter_errors(self, obj):
                raise RuntimeError("boom")
        with mock.patch.object(sv, "_validator", return_value=Boom()):
            state, detail = sv.validate_instance({}, {})
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("RuntimeError", detail)

    def test_nothing_is_written_to_disk_by_fetching(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd(); os.chdir(d)
            try:
                sv.validate_instance({"a": 1}, self.schema_with_ref())
                self.assertEqual(os.listdir(d), [])
            finally:
                os.chdir(old)


def wait_until(predicate, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class WorkerThreadSafetyTest(SvBase):
    """The download thread behind the total time limit must not accumulate or live on after its attempt."""

    def setUp(self):
        super().setUp()
        for name, value in (("REMOTE_TOTAL_SECONDS", 0.5), ("REMOTE_TIMEOUT_SECONDS", 5.0)):
            p = mock.patch.object(sv, name, value); p.start(); self.addCleanup(p.stop)
        self.addCleanup(lambda: self.assertTrue(wait_until(lambda: sv.stalled_workers() == 0, 10), "worker threads left behind"))

    def test_a_dripping_download_thread_ends_on_its_own_shortly_after_the_deadline(self):
        server = SlowServer("body", interval=0.1, pieces=2000); self.addCleanup(server.close)
        for i in range(3):                                            # repeated failures on different documents
            sv.reset_caches()
            schema = {"$ref": f"http://127.0.0.1:{server.port}/d{i}"}
            self.assertEqual(sv.validate_instance({}, schema)[0], sv.TECHNICAL)
            self.assertLessEqual(sv.stalled_workers(), 1)             # at most the attempt that was just abandoned
            self.assertTrue(wait_until(lambda: sv.stalled_workers() == 0, 3.0), "the abandoned download did not end")
        self.assertEqual(sum(t.name == "benchson-remote-ref" and t.is_alive() for t in threading.enumerate()), 0)

    def test_hanging_downloads_cannot_pile_up_beyond_the_cap_and_fetching_recovers_afterwards(self):
        release = threading.Event(); self.addCleanup(release.set)
        def hang(uri, deadline):
            release.wait(30); raise sv.RemoteRefError("connection", uri, retriable=True)
        with mock.patch.object(sv, "_fetch_blocking", hang), mock.patch.object(sv, "REMOTE_ATTEMPTS", 1):
            kinds = []
            for i in range(sv.MAX_STALLED_WORKERS + 4):
                with self.assertRaises(sv.RemoteRefError) as ctx:
                    sv.fetch_remote_document(f"http://127.0.0.1:1/doc{i}")
                kinds.append(ctx.exception.kind)
            self.assertEqual(sv.stalled_workers(), sv.MAX_STALLED_WORKERS)          # never more threads than the cap
            self.assertEqual(kinds[:sv.MAX_STALLED_WORKERS], ["timeout"] * sv.MAX_STALLED_WORKERS)
            self.assertEqual(set(kinds[sv.MAX_STALLED_WORKERS:]), {"too_many_stalled_requests"})   # then: fail closed, no new thread
            state, detail = sv.validate_instance({}, {"$ref": "http://127.0.0.1:1/another"})
            self.assertEqual(state, sv.TECHNICAL); self.assertIn("too_many_stalled_requests", detail)
        release.set()
        self.assertTrue(wait_until(lambda: sv.stalled_workers() == 0, 10))
        sv.reset_caches(); self.srv.docs["/ok"] = {"type": "integer"}
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref("/ok"))[0], sv.VALID)   # recovered

    def test_documented_limits_attempts_total_time_and_backoff(self):
        server = SlowServer("silent", interval=1.0, pieces=60); self.addCleanup(server.close)
        t0 = time.time()
        with self.assertRaises(sv.RemoteRefError) as ctx:
            sv.fetch_remote_document(f"http://127.0.0.1:{server.port}/d")
        elapsed = time.time() - t0
        self.assertEqual(ctx.exception.kind, "timeout"); self.assertEqual(server.connections, sv.REMOTE_ATTEMPTS)
        self.assertEqual(self.sleeps, list(sv.REMOTE_BACKOFF_SECONDS[:sv.REMOTE_ATTEMPTS - 1]))
        self.assertGreaterEqual(elapsed, sv.REMOTE_ATTEMPTS * 0.5 - 0.1); self.assertLess(elapsed, sv.REMOTE_ATTEMPTS * 0.5 + 1.5)
        # the documented worst case per failing document: attempts x total limit + the two waits
        self.assertEqual((DEFAULT_TOTAL, DEFAULT_ATTEMPTS, DEFAULT_BACKOFF), (30, 3, (1.0, 3.0)))
        self.assertEqual(DEFAULT_ATTEMPTS * DEFAULT_TOTAL + sum(DEFAULT_BACKOFF), 94)       # seconds, worst case per failing document

    def test_an_unexpected_internal_error_is_technical_and_never_retried(self):
        with mock.patch.object(sv, "_fetch_blocking", side_effect=RecursionError("deep json")):
            state, detail = sv.validate_instance({"a": 1}, self.schema_with_ref())
        self.assertEqual(state, sv.TECHNICAL); self.assertIn("internal", detail); self.assertEqual(self.sleeps, [])

    def test_the_document_cache_is_bounded_by_size_and_keeps_digests_in_step(self):
        for i in range(6):
            self.srv.raw[f"/d{i}"] = json.dumps({"type": "integer", "description": "x" * 300}).encode()
        with mock.patch.object(sv, "MAX_CACHED_BYTES", 1000):
            for i in range(6):
                self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref(f"/d{i}"))[0], sv.VALID)
                self.assertLessEqual(sum(sv._sizes.values()), 1000)
        self.assertEqual(set(sv._documents), set(sv._digests)); self.assertEqual(set(sv._documents), set(sv._sizes))
        self.assertLess(len(sv._documents), 6)
        self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref("/d0"))[0], sv.VALID)   # an evicted document is just fetched again

    def test_remembered_failures_stay_bounded(self):
        with mock.patch.object(sv, "MAX_REMEMBERED_FAILURES", 5), mock.patch.object(sv, "REMOTE_ATTEMPTS", 1):
            for i in range(20):
                self.srv.status[f"/f{i}"] = 404
                self.assertEqual(sv.validate_instance({"a": 1}, self.schema_with_ref(f"/f{i}"))[0], sv.TECHNICAL)
            self.assertLessEqual(len(sv._failures), 5)


class DigestReportSideEffectTest(RemoteRefChunkBase):
    def test_the_report_is_read_only_information_and_leaves_no_trace_in_seals_or_files(self):
        out = self.sealed_chunk(); before = tree_hash(out); sv.reset_caches()
        first = self.validate(out); second = self.validate(out)
        self.assertEqual((first["code"], second["code"]), (0, 0))
        self.assertEqual(first["remote_documents"], second["remote_documents"])
        digest = next(iter(first["remote_documents"].values()))
        self.assertEqual(before, tree_hash(out))                                    # nothing written into the chunk
        for name in os.listdir(out):
            path = os.path.join(out, name)
            if os.path.isfile(path):
                self.assertNotIn(digest, open(path, encoding="utf-8", errors="ignore").read(), name)   # not in the seal, state or manifest

    def test_the_returned_report_is_a_copy_and_urls_never_show_credentials(self):
        self.srv.docs["/doc"] = {}
        sv.validate_instance({"a": 1}, {"$ref": self.srv.url("/doc")})
        report = sv.used_documents(); report.clear()
        self.assertEqual(len(sv.used_documents()), 1)
        self.assertEqual(sv._display("https://user:secret@example.org/a/b.json?x=1#frag"), "https://example.org/a/b.json?x=1")

    def test_the_verdict_is_the_same_with_or_without_looking_at_the_report(self):
        self.srv.docs["/doc"] = {"type": "integer"}
        schema = {"$ref": self.srv.url("/doc")}
        a = sv.validate_instance(1, schema); sv.used_documents(); b = sv.validate_instance(1, schema)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
