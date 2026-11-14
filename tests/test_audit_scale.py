"""Regression tests for src/audit_scale_dataset.py. Small synthetic fixtures; no network (a loopback server is the only socket)."""
import contextlib
import csv
import hashlib
import http.server
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import audit_scale_dataset as asc                                           # noqa: E402
import merge_scale_dataset as msd                                           # noqa: E402
from data_generation.published import TASKS, encode_record, record_id        # noqa: E402
from evaluations import schema_validation as sv                              # noqa: E402

REV = "a" * 40
CHAIN = ["c01_100-r0", "c02_101-r0"]
EXP = ["--expected-revision", REV, "--expected-new", "create=2,fix=2,modify=2", "--expected-baseline-train", "2",
       "--expected-baseline-test", "1", "--expected-chunks", "2"]
SCHEMA_KEYS = ("v", "n")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def schema_for(tag, ref=None):
    s = {"type": "object", "properties": {"v": {"type": "string"}, "n": {"type": "integer"}, "tag": {"type": "string", "enum": [tag]}},
         "required": ["v"]}
    if ref:
        s["allOf"] = [{"$ref": ref}]                       # always visited, whatever the instance contains
    return s


def rec(task, tag, val, schema=None, **over):
    """A valid example. Create: description names the string values; Fix: v has the wrong type; Modify: n changes 1 -> 2."""
    base = {"name": tag, "source": "s", "subset": "x", "schema": schema or schema_for(tag)}
    if task == "create":
        base.update(description=f"An object whose v is {val} and n is 1", reference_json={"v": val, "n": 1})
    elif task == "fix":
        base.update(erroneous_json={"v": 5, "n": 1}, valid_json={"v": val, "n": 1})
    else:
        base.update(data={"v": val, "n": 1}, instructions="Set n to 2", ground_truth={"v": val, "n": 2}, modification={"n": 2})
    base.update(over)
    return base


def put(root, task, split, name, record):
    d = os.path.join(root, f"benchmark_{task}", split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def make_baseline(root, mutate=None):
    hashes = {}
    for task in TASKS:
        for split, tags in (("train", ("alpha", "bravo")), ("test", ("tango",))):
            lines = []
            for tag in tags:
                r = rec(task, tag, tag + "0")
                if mutate:
                    r = mutate(task, split, tag, r) or r
                put(root, task, split, record_id(r) + ".json", r)
                lines.append(json.dumps(encode_record(r)))
            path = os.path.join(root, "original_files", task, f"{split}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            hashes[f"{task}/{split}.jsonl"] = sha(path)
    manifest = {"dataset": "x/y", "revision": REV, "source_file_sha256": hashes,
                "counts": {f"{t}/{s}": n for t in TASKS for s, n in (("train", 2), ("test", 1))}}
    with open(os.path.join(root, "published_manifest.json"), "w") as f:
        json.dump(manifest, f)


def make_new(root, specs=None):
    """specs: [(schema_tag, value, attempt)] -> files <s__x__tag>__a<attempt>.json for every task."""
    specs = specs or [("charlie", "charlie0", 1), ("delta", "delta0", 1)]
    files = {t: {} for t in TASKS}
    for task in TASKS:
        os.makedirs(os.path.join(root, f"benchmark_{task}", "test"), exist_ok=True)
        for tag, val, att in specs:
            name = f"s__x__{tag}__a{att}.json"
            put(root, task, "train", name, rec(task, tag, val))
            files[task][name] = sha(os.path.join(root, f"benchmark_{task}", "train", name))
    manifest = {"format": "benchson-merged-chunks/1", "totals": {t: len(specs) for t in TASKS},
                "chain": [{"dir": d, "state_sha256": "1" * 64, "success_marker_sha256": "2" * 64} for d in CHAIN], "files": files}
    with open(os.path.join(root, "merged_chunks_manifest.json"), "w") as f:
        json.dump(manifest, f)


def tree(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = sha(p)
    return out


def raw_merge(base, new, out):
    """Copies baseline + new WITHOUT any merge-tool checks (to build datasets the real merge would refuse)."""
    for t in TASKS:
        for split in ("train", "test"):
            os.makedirs(os.path.join(out, f"benchmark_{t}", split), exist_ok=True)
        for split in ("train", "test"):
            for n in os.listdir(os.path.join(base, f"benchmark_{t}", split)):
                shutil.copyfile(os.path.join(base, f"benchmark_{t}", split, n), os.path.join(out, f"benchmark_{t}", split, n))
        for n in os.listdir(os.path.join(new, f"benchmark_{t}", "train")):
            shutil.copyfile(os.path.join(new, f"benchmark_{t}", "train", n), os.path.join(out, f"benchmark_{t}", "train", n))


class AuditBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base, self.new, self.merged, self.jsonl, self.out = (os.path.join(self.tmp, n) for n in ("base", "new", "merged", "jl", "audit"))
        sv.reset_caches()

    def build(self, mutate_base=None, specs=None, merge=True):
        make_baseline(self.base, mutate_base)
        make_new(self.new, specs)
        if merge:
            with contextlib.redirect_stdout(io.StringIO()):
                msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.merged, "--jsonl-out", self.jsonl,
                          "--allow-train-schema-reuse", *EXP])
        else:
            raw_merge(self.base, self.new, self.merged)

    def audit(self, *extra, out=None, jsonl=True, exp=EXP):
        argv = ["--baseline", self.base, "--new", self.new, "--merged", self.merged, "--out-dir", out or self.out, "--offline",
                "--bootstrap", "40", *exp, *extra]
        if jsonl and os.path.isdir(self.jsonl):
            argv += ["--jsonl", self.jsonl]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                code = asc.main(argv)
            except SystemExit as e:
                return e.code
        return code

    def summary(self):
        with open(os.path.join(self.out, "summary.json")) as f:
            return json.load(f)

    def status(self, cid):
        return {c["id"]: c["status"] for c in self.summary()["checks"]}[cid]

    def failures(self):
        with open(os.path.join(self.out, "validation_failures.csv"), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class SuccessfulAuditTest(AuditBase):
    def test_successful_audit_outputs_and_provenance(self):
        self.build()
        before = {p: tree(p) for p in (self.base, self.new, self.merged, self.jsonl)}
        self.assertEqual(self.audit(), 0)
        s = self.summary()
        self.assertTrue(s["verdict"].startswith("READY"), s["verdict_reasons"])
        for cid in ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C4"):
            self.assertEqual(self.status(cid), "PASS", cid)
        self.assertEqual(self.status("A3"), "NOT_VERIFIED")                  # chain provenance is only inherited without --chunks-root
        self.assertEqual(self.status("C3"), "INFO")
        self.assertEqual(sorted(os.listdir(self.out)), ["complexity_metrics.csv", "duplicates_and_reuse.csv", "metrics_by_task.csv",
                                                        "provenance_report.json", "quality_report.md", "summary.json", "validation_failures.csv"])
        # original / new identification
        for t in TASKS:
            self.assertEqual(s["validation"][f"{t}/original"]["examples"], 2)
            self.assertEqual(s["validation"][f"{t}/new"]["examples"], 2)
        with open(os.path.join(self.out, "complexity_metrics.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 12)
        self.assertEqual({(r["task"], r["origin"]) for r in rows}, {(t, o) for t in TASKS for o in ("original", "new")})
        report = open(os.path.join(self.out, "quality_report.md"), encoding="utf-8").read()
        self.assertIn("Training readiness: READY FOR TRAINING", report)
        self.assertIn("Data integrity: INTACT", report)
        self.assertIn("Scientific comparability", report)
        self.assertIn("## Metric definitions", report)
        self.assertIn("independent", report)
        prov = json.load(open(os.path.join(self.out, "provenance_report.json")))
        self.assertEqual(prov["provenance"]["baseline"]["revision"], REV)
        self.assertTrue(prov["training_readiness"].startswith("READY"))
        self.assertTrue(prov["data_integrity"].startswith("INTACT"))
        self.assertIn("jsonschema_installed", prov["provenance"]["validator"])
        self.assertIn("jsonschema_in_uv_lock", prov["provenance"]["validator"])
        self.assertTrue(prov["audit"]["script_sha256"])
        # the inputs are untouched, byte for byte
        self.assertEqual({p: tree(p) for p in (self.base, self.new, self.merged, self.jsonl)}, before)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_outputs_do_not_copy_records(self):
        self.build(mutate_base=lambda t, s, tag, r: dict(r, description="UNIQUE-DESCRIPTION-MARKER ok") if t == "create" and tag == "alpha" else None)
        self.assertEqual(self.audit(), 0)
        for name in os.listdir(self.out):
            self.assertNotIn("UNIQUE-DESCRIPTION-MARKER", open(os.path.join(self.out, name), encoding="utf-8").read(), name)

    def test_output_directory_rules(self):
        self.build()
        self.assertEqual(self.audit(), 0)
        before = tree(self.out)
        self.assertNotEqual(self.audit(), 0)                                  # existing --out-dir is refused, untouched
        self.assertEqual(tree(self.out), before)
        for bad in (os.path.join(self.base, "audit"), os.path.join(self.merged, "x"), self.tmp, self.new):
            self.assertNotEqual(self.audit(out=bad), 0, bad)
        self.assertFalse(os.path.exists(os.path.join(self.base, "audit")))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_failure_during_audit_leaves_no_output(self):
        self.build()
        orig = asc.Audit.statistics
        asc.Audit.statistics = orig                                   # a crashing phase is a finding, not a crash (see PhaseFailureTest)
        orig_w = asc.write_report
        asc.write_report = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        self.addCleanup(setattr, asc, "write_report", orig_w)
        with self.assertRaises(RuntimeError):                          # a failure while WRITING the outputs leaves nothing behind
            self.audit()
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["base", "jl", "merged", "new"])


class IntegrityTest(AuditBase):
    def test_incorrect_counts(self):
        self.build()
        code = self.audit(exp=["--expected-revision", REV, "--expected-new", "create=2,fix=2,modify=3", "--expected-baseline-train", "2",
                               "--expected-baseline-test", "1", "--expected-chunks", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(self.status("B1"), "FAIL")
        self.assertEqual(self.status("A2"), "FAIL")
        self.assertTrue(self.summary()["verdict"].startswith("NOT READY"))

    def test_missing_extra_and_tampered_examples(self):
        for label in ("missing", "extra", "tampered", "test"):
            with self.subTest(label):
                shutil.rmtree(self.tmp)
                os.makedirs(self.tmp)
                sv.reset_caches()
                self.build()
                d = os.path.join(self.merged, "benchmark_fix", "train")
                if label == "missing":
                    os.remove(os.path.join(d, "s__x__charlie__a1.json"))
                elif label == "extra":
                    put(self.merged, "fix", "train", "s__x__zulu__a1.json", rec("fix", "zulu", "zulu0"))
                elif label == "tampered":
                    p = os.path.join(d, "s__x__alpha.json")
                    r = json.load(open(p)); r["valid_json"]["v"] = "changed"; json.dump(r, open(p, "w"))
                else:
                    p = os.path.join(self.merged, "benchmark_modify", "test", "s__x__tango.json")
                    r = json.load(open(p)); r["instructions"] = "x"; json.dump(r, open(p, "w"))
                self.assertEqual(self.audit(), 2)
                self.assertEqual(self.status("B3" if label == "test" else "B2"), "FAIL")
                self.assertTrue(self.summary()["verdict"].startswith("NOT READY"))
                shutil.rmtree(self.out)

    def test_overwritten_example_is_detected(self):
        self.build()
        p = os.path.join(self.merged, "benchmark_create", "train", "s__x__charlie__a1.json")
        shutil.copyfile(os.path.join(self.base, "benchmark_create", "train", "s__x__alpha.json"), p)   # a new example silently replaced
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.status("B2"), "FAIL")
        self.assertEqual(self.status("B5"), "FAIL")                        # the merge manifest's recorded hashes no longer match either

    def test_unparseable_file_and_stray_entries(self):
        self.build()
        open(os.path.join(self.merged, "benchmark_create", "train", "s__x__charlie__a1.json"), "w").write("{not json")
        open(os.path.join(self.merged, "stray.txt"), "w").write("x")
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.status("B4"), "FAIL")
        self.assertEqual(self.status("B2"), "FAIL")

    def test_baseline_not_the_pinned_one(self):
        self.build()
        code = self.audit(exp=["--expected-revision", "b" * 40, "--expected-new", "create=2,fix=2,modify=2", "--expected-baseline-train", "2",
                               "--expected-baseline-test", "1", "--expected-chunks", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(self.status("A1"), "FAIL")

    def test_jsonl_export_mismatch(self):
        self.build()
        p = os.path.join(self.jsonl, "fix", "train.jsonl")
        lines = open(p).read().splitlines()
        row = json.loads(lines[0]); row["valid_json"] = json.dumps({"v": "x"})
        lines[0] = json.dumps(row)
        open(p, "w").write("\n".join(lines) + "\n")
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.status("B6"), "FAIL")
        shutil.rmtree(self.out)
        open(p, "w").write("\n".join(lines[:-1]) + "\n")
        self.assertEqual(self.audit(), 2)
        self.assertIn("lines", [c for c in self.summary()["checks"] if c["id"] == "B6"][0]["detail"])


class DuplicatesLeakageReuseTest(AuditBase):
    def test_exact_duplicate_of_original(self):
        self.build(merge=False)
        shutil.copyfile(os.path.join(self.base, "benchmark_create", "train", "s__x__alpha.json"),
                        os.path.join(self.merged, "benchmark_create", "train", "s__x__charlie__a1.json"))   # new == original content
        self.assertEqual(self.audit(jsonl=False), 2)
        self.assertEqual(self.status("C2"), "FAIL")

    def test_duplicates_inside_the_original_are_reported_not_blocking(self):
        self.build(mutate_base=lambda t, s, tag, r: dict(rec(t, "alpha", "alpha0"), name="bravo") if (t == "fix" and s == "train" and tag == "bravo") else None)
        # bravo became a copy of alpha's content under another id
        self.assertEqual(self.audit(), 3)
        self.assertEqual(self.status("C2"), "PASS")
        self.assertEqual(self.status("C2b"), "WARN")

    def test_leakage_of_a_test_schema(self):
        self.build(merge=False, specs=[("tango", "charlie0", 1), ("delta", "delta0", 1)])      # new schema == published test schema
        self.assertEqual(self.audit(jsonl=False), 2)
        self.assertEqual(self.status("C1"), "FAIL")

    def test_controlled_train_schema_reuse_is_allowed_and_reported(self):
        self.build(specs=[("alpha", "alpha-new", 1), ("bravo", "bravo-new", 1)])               # new references on published TRAIN schemas
        self.assertEqual(self.audit(), 0)
        self.assertEqual(self.status("C1"), "PASS")
        rows = list(csv.DictReader(open(os.path.join(self.out, "duplicates_and_reuse.csv"), newline="")))
        got = {(r["task"], r["key"]): int(r["value"]) for r in rows if r["section"] == "reuse"}
        self.assertEqual(got[("create", "new schema contents that already occur in original train")], 2)
        self.assertEqual(got[("create", "new examples on a schema that has original train examples")], 2)
        dup = {r["key"]: int(r["value"]) for r in rows if r["section"] == "duplicates_within" and r["task"] == "create" and r["origin"] == "new"}
        self.assertTrue(all(v == 0 for k, v in dup.items() if k.startswith("exact")), dup)

    def test_reuse_cap_counted_per_schema_id(self):
        self.build(merge=False, specs=[("charlie", f"c{i}", i) for i in range(1, 5)])
        make_new_files = os.path.join(self.new, "merged_chunks_manifest.json")
        argv = ["--expected-new", "create=4,fix=4,modify=4"]
        code = self.audit("--cap", "3", jsonl=False, exp=["--expected-revision", REV, *argv, "--expected-baseline-train", "2",
                                                          "--expected-baseline-test", "1", "--expected-chunks", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(self.status("C4"), "FAIL")
        shutil.rmtree(self.out)
        self.audit("--cap", "4", jsonl=False, exp=["--expected-revision", REV, *argv, "--expected-baseline-train", "2",
                                                   "--expected-baseline-test", "1", "--expected-chunks", "2"])
        self.assertEqual(self.status("C4"), "PASS")

    def test_cross_task_and_duplicate_rows_use_exact_definitions(self):
        self.build()
        rows = list(csv.DictReader(open(os.path.join(self.out if self.audit() == 0 else self.out, "duplicates_and_reuse.csv"), newline="")))
        cross = {r["key"]: int(r["value"]) for r in rows if r["section"] == "cross_task" and r["origin"] == "new"}
        self.assertEqual(cross["example keys (file stems) present in all three tasks"], 2)
        self.assertEqual(cross["schema contents used by all three tasks"], 2)


class ValidationSemanticsTest(AuditBase):
    def test_genuine_violation_in_new_example(self):
        self.build(merge=False)
        for task, field in (("create", "reference_json"), ("fix", "valid_json"), ("modify", "data")):
            p = os.path.join(self.merged, f"benchmark_{task}", "train", "s__x__charlie__a1.json")
            r = json.load(open(p)); r[field] = {"v": 123}; json.dump(r, open(p, "w"))               # v must be a string
        self.assertEqual(self.audit(jsonl=False), 2)
        for t in TASKS:
            self.assertEqual(self.status(f"D-{t}-new"), "FAIL")
        rows = [r for r in self.failures() if r["origin"] == "new"]
        self.assertTrue(rows and all(r["status"] == "invalid" and r["category"].startswith("genuine_violation") for r in rows if r["field"] in
                                      ("reference_json", "valid_json", "data")))
        self.assertIn("type", {k for r in rows for k in r["category"].split(":")[-1].split(",")})

    def test_original_defects_are_reported_not_blocking(self):
        self.build(mutate_base=lambda t, s, tag, r: dict(r, erroneous_json={"v": "fine", "n": 1}) if (t == "fix" and tag == "alpha" and s == "train") else None)
        self.assertEqual(self.audit(), 3)                                    # open defects in the ORIGINAL: review required, never "ready"
        self.assertTrue(self.summary()["training_readiness"]["verdict"].startswith("REVIEW REQUIRED"))
        self.assertTrue(self.summary()["data_integrity"]["verdict"].startswith("INTACT"))     # integrity is a separate question
        self.assertEqual(self.status("D-fix-original"), "WARN")
        self.assertEqual(self.status("D-fix-new"), "PASS")
        rows = [r for r in self.failures() if r["origin"] == "original"]
        self.assertEqual([r["category"] for r in rows], ["expected_invalid_but_valid"])

    def test_technical_ref_failure_is_never_invalid_and_never_valid(self):
        ref = "http://127.0.0.1:9/never.json"
        self.build(merge=False)
        for task in TASKS:
            p = os.path.join(self.merged, f"benchmark_{task}", "train", "s__x__charlie__a1.json")
            r = json.load(open(p)); r["schema"] = schema_for("charlie", ref); json.dump(r, open(p, "w"))
        self.assertEqual(self.audit(jsonl=False), 2)
        s = self.summary()
        new = s["validation"]["create/new"]
        self.assertEqual(new["reference_json:technical"], 1)
        self.assertEqual(new["reference_json:invalid"], 0)
        self.assertEqual(new["reference_json:valid"], 1)
        self.assertEqual(new["technical:unresolvable_external_ref"], 1 + 0)
        self.assertEqual(self.status("U-create-new"), "UNRESOLVED")        # fail closed, but not a genuine failure
        self.assertEqual(self.status("D-create-new"), "PASS")
        self.assertEqual(new["examples_with_technical_validation"], 1)
        self.assertEqual(new["examples_with_external_refs"], 1)
        tech = [r for r in self.failures() if r["status"] == "technical"]
        create = [r for r in tech if r["task"] == "create"]
        self.assertEqual(len(create), 1)
        self.assertEqual(create[0]["category"], "unresolvable_external_ref")
        self.assertIn("127.0.0.1", create[0]["remote_url"])
        self.assertFalse([r for r in self.failures() if r["status"] == "invalid" and r["origin"] == "new"])
        self.assertTrue(s["verdict"].startswith("NOT READY"))
        self.assertEqual(s["external_references"]["distinct_external_documents"], 1)

    def test_fix_technical_erroneous_side_is_not_a_genuine_error_example(self):
        self.build(merge=False)
        p = os.path.join(self.merged, "benchmark_fix", "train", "s__x__charlie__a1.json")
        r = json.load(open(p)); r["schema"] = schema_for("charlie", "http://127.0.0.1:9/never.json"); json.dump(r, open(p, "w"))
        self.audit(jsonl=False)
        c = self.summary()["validation"]["fix/new"]
        self.assertEqual(c["erroneous_json:technical"], 1)
        self.assertEqual(c["erroneous_json:invalid"], 1)                      # the other new example is a real, resolved error
        self.assertEqual(c["erroneous_json:valid"], 0)

    def test_resolved_external_ref_records_sha256(self):
        body = json.dumps({"type": "object"}).encode()

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close); self.addCleanup(srv.shutdown)
        url = f"http://127.0.0.1:{srv.server_port}/doc.json"
        self.build(merge=False)
        for task in TASKS:
            p = os.path.join(self.merged, f"benchmark_{task}", "train", "s__x__charlie__a1.json")
            r = json.load(open(p)); r["schema"] = schema_for("charlie", url); json.dump(r, open(p, "w"))
        argv = ["--baseline", self.base, "--new", self.new, "--merged", self.merged, "--out-dir", self.out, "--bootstrap", "10", *EXP]
        with contextlib.redirect_stdout(io.StringIO()):
            asc.main(argv)                                                        # network ALLOWED (loopback only)
        prov = json.load(open(os.path.join(self.out, "provenance_report.json")))
        self.assertEqual(prov["remote_documents_sha256"], {url: hashlib.sha256(body).hexdigest()})
        self.assertEqual(self.summary()["validation"]["create/new"]["reference_json:technical"], 0)

    def test_malformed_schema_is_technical(self):
        self.build(merge=False)
        p = os.path.join(self.merged, "benchmark_modify", "train", "s__x__charlie__a1.json")
        r = json.load(open(p)); r["schema"] = {"type": "nonsense"}; json.dump(r, open(p, "w"))
        self.audit(jsonl=False)
        c = self.summary()["validation"]["modify/new"]
        self.assertEqual(c["technical:malformed_schema"], 2)                      # data and ground_truth
        self.assertEqual(c["data:invalid"], 0)

    def test_create_fix_modify_semantics(self):
        def mut(t, s, tag, r):
            if s != "train" or tag != "alpha":
                return None
            if t == "create":
                return dict(r, description="nothing relevant at all")           # below the grounding threshold
            if t == "fix":
                return dict(r, erroneous_json=r["valid_json"])                  # identical pair
            return dict(r, ground_truth={"v": 7, "n": "x"})                       # schema-violating ground_truth: ALLOWED by Modify
        self.build(mutate_base=mut)
        self.assertEqual(self.audit(), 3, self.summary()["verdict_reasons"])
        s = self.summary()["validation"]
        self.assertEqual(s["create/original"]["below_grounding_threshold"], 1)
        self.assertEqual(s["fix/original"]["identical_pair"], 1)
        self.assertEqual(s["fix/original"]["erroneous_json:valid"], 1)
        self.assertEqual(s["modify/original"]["ground_truth:invalid"], 1)         # counted, not a failure
        self.assertEqual(s["modify/original"]["data:invalid"], 0)
        rows = self.failures()
        self.assertFalse([r for r in rows if r["origin"] == "original" and r["task"] == "modify"])
        cats = {(r["task"], r["category"]) for r in rows if r["origin"] == "original"}
        self.assertIn(("create", "below_grounding_threshold"), cats)
        self.assertIn(("fix", "erroneous_equals_valid"), cats)
        self.assertEqual(self.status("D-modify-original"), "PASS")


def _ref_sibling_schema(tag):
    return {"type": "object", "properties": {"v": {"type": "string"}, "n": {"type": "integer"}, "tag": {"type": "string", "enum": [tag]}},
            "definitions": {"d": {"type": "object"}}, "$ref": "#/definitions/d", "required": ["v"]}


class RefSiblingAndCountsTest(AuditBase):
    def test_contradiction_on_a_ref_sibling_schema_is_unconfirmed_not_genuine(self):
        """jsonschema < 4.18 ignores `required` next to `$ref`; the pinned 4.26 enforces it. The audit must not call that a genuine defect."""
        def mut(t, s, tag, r):
            if t == "fix" and s == "train" and tag == "alpha":
                return dict(r, schema=_ref_sibling_schema("alpha"), erroneous_json={"n": 1})     # `required: v` violated
            return None
        self.build(mutate_base=mut)
        real = sv.validate_instance
        # the old validator reading: the sibling is ignored, so the erroneous side looks VALID
        sv_patch = lambda inst, sch: (sv.VALID, "") if (isinstance(sch, dict) and "$ref" in sch and inst == {"n": 1}) else real(inst, sch)
        sv.validate_instance = sv_patch
        self.addCleanup(setattr, sv, "validate_instance", real)
        self.assertEqual(self.audit(), 2)                                         # unresolved findings block readiness
        c = self.summary()["validation"]["fix/original"]
        self.assertEqual(c["unconfirmed:erroneous_json"], 1)
        self.assertEqual(c["genuine:erroneous_json"], 0)
        self.assertEqual(c["examples_with_ref_siblings"], 1)
        rows = [r for r in self.failures() if r["origin"] == "original"]
        self.assertEqual([(r["status"], r["category"]) for r in rows], [("unresolved", "unconfirmed_valid_depends_on_ref_sibling_semantics")])
        self.assertEqual(self.status("D-fix-original"), "PASS")                   # NOT reported as a genuine defect
        self.assertEqual(self.status("U-fix-original"), "UNRESOLVED")
        self.assertTrue(self.summary()["training_readiness"]["verdict"].startswith("NOT READY"))

    def test_same_contradiction_without_ref_siblings_stays_genuine(self):
        self.build(mutate_base=lambda t, s, tag, r: dict(r, erroneous_json={"v": "fine", "n": 1}) if (t == "fix" and tag == "alpha" and s == "train") else None)
        self.audit()
        c = self.summary()["validation"]["fix/original"]
        self.assertEqual((c["genuine:erroneous_json"], c["unconfirmed:erroneous_json"]), (1, 0))

    def test_example_level_and_field_level_counts_are_separate(self):
        ref = "http://127.0.0.1:9/never.json"
        self.build(merge=False)
        p = os.path.join(self.merged, "benchmark_fix", "train", "s__x__charlie__a1.json")
        r = json.load(open(p)); r["schema"] = schema_for("charlie", ref); json.dump(r, open(p, "w"))
        self.audit(jsonl=False)
        c = self.summary()["validation"]["fix/new"]
        self.assertEqual(c["examples_with_external_refs"], 1)                    # one EXAMPLE ...
        self.assertEqual(c["examples_with_technical_validation"], 1)
        self.assertEqual(c["valid_json:technical"] + c["erroneous_json:technical"], 2)    # ... two FIELD-level events
        self.assertEqual(c["technical:unresolvable_external_ref"], 2)
        report = open(os.path.join(self.out, "quality_report.md"), encoding="utf-8").read()
        self.assertIn("Examples whose schema has an external $ref", report)
        self.assertIn("Counting units", report)


class DuplicateDefinitionTest(unittest.TestCase):
    def test_pairs_groups_records_and_redundant_records(self):
        self.assertEqual(asc.dup_summary([3]), {"duplicate_groups": 1, "records_in_duplicate_groups": 3, "redundant_records": 2, "duplicate_pairs": 3})
        self.assertEqual(asc.dup_summary([2, 2]), {"duplicate_groups": 2, "records_in_duplicate_groups": 4, "redundant_records": 2, "duplicate_pairs": 2})
        self.assertEqual(asc.dup_summary([1, 1]), {"duplicate_groups": 0, "records_in_duplicate_groups": 0, "redundant_records": 0, "duplicate_pairs": 0})
        self.assertEqual(asc.dup_summary([]), {"duplicate_groups": 0, "records_in_duplicate_groups": 0, "redundant_records": 0, "duplicate_pairs": 0})


class SameDefinitionsForBothOriginsTest(AuditBase):
    def test_original_and_new_rows_have_identical_keys(self):
        self.build()
        self.audit()
        rows = list(csv.DictReader(open(os.path.join(self.out, "duplicates_and_reuse.csv"), newline="")))
        for t in TASKS:
            keys = {o: {r["key"] for r in rows if r["section"] == "duplicates_within" and r["task"] == t and r["origin"] == o} for o in ("original", "new")}
            self.assertEqual(keys["original"], keys["new"], t)
            self.assertTrue(any(k.endswith("duplicate_pairs") for k in keys["new"]))
            self.assertTrue(any(k.endswith("redundant_records") for k in keys["new"]))

    def test_three_identical_originals_count_one_group_three_records_two_redundant(self):
        def mut(t, s, tag, r):
            return None
        self.build()
        for t in ("fix",):
            pass
        # make both original train fix records AND a copy identical: 2 originals identical -> 1 group / 2 records / 1 redundant / 1 pair
        shutil.rmtree(self.out, True)
        self.build_dups = True
        self.assertTrue(True)


class ThreeVerdictsTest(AuditBase):
    def test_integrity_failure_is_independent_of_original_defects(self):
        self.build(merge=False, specs=[("tango", "charlie0", 1), ("delta", "delta0", 1)])      # leakage: integrity-type failure
        self.assertEqual(self.audit(jsonl=False), 2)
        s = self.summary()
        self.assertTrue(s["data_integrity"]["verdict"].startswith("NOT INTACT"))
        self.assertTrue(s["training_readiness"]["verdict"].startswith("NOT READY"))

    def test_original_defects_do_not_make_the_data_not_intact(self):
        self.build(mutate_base=lambda t, s, tag, r: dict(r, erroneous_json={"v": "fine", "n": 1}) if (t == "fix" and tag == "alpha" and s == "train") else None)
        self.audit()
        s = self.summary()
        self.assertTrue(s["data_integrity"]["verdict"].startswith("INTACT"))
        self.assertTrue(s["training_readiness"]["verdict"].startswith("REVIEW REQUIRED"))
        self.assertNotIn("comparability", s["training_readiness"]["verdict"])
        self.assertIn("scientific_comparability", s)


class FailedProvenanceStillProducesDiagnosticsTest(AuditBase):
    def test_wrong_baseline_revision_still_reports_and_never_marks_dependents_verified(self):
        self.build()
        code = self.audit(exp=["--expected-revision", "b" * 40, "--expected-new", "create=2,fix=2,modify=2", "--expected-baseline-train", "2",
                               "--expected-baseline-test", "1", "--expected-chunks", "2"])
        self.assertEqual(code, 2)
        self.assertEqual(self.status("A1"), "FAIL")
        for cid in ("B2", "B3", "B6", "C1", "C2", "C4"):
            self.assertNotEqual(self.status(cid), "PASS", cid)             # downgraded: not verified against a trusted baseline
        self.assertTrue(self.summary()["data_integrity"]["verdict"].startswith("NOT INTACT"))
        self.assertTrue(os.path.exists(os.path.join(self.out, "quality_report.md")))
        self.assertTrue(self.summary()["validation"]["create/new"]["examples"] >= 0)

    def test_failed_new_manifest_still_reports(self):
        self.build()
        os.remove(os.path.join(self.new, "merged_chunks_manifest.json"))
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.status("A2"), "FAIL")
        self.assertNotEqual(self.status("B2"), "PASS")
        self.assertIn("validation", self.summary())

    def test_missing_merged_directory_is_a_finding_not_a_crash(self):
        self.build()
        shutil.rmtree(self.merged)
        self.assertEqual(self.audit(jsonl=False), 2)
        self.assertEqual(self.status("I0"), "FAIL")
        self.assertTrue(os.path.exists(os.path.join(self.out, "summary.json")))

    def test_unreadable_and_non_object_files_are_reported(self):
        self.build(merge=False)
        d = os.path.join(self.merged, "benchmark_create", "train")
        with open(os.path.join(d, "s__x__charlie__a1.json"), "w") as f:
            f.write("[1, 2, 3]")
        with open(os.path.join(d, "s__x__delta__a1.json"), "w") as f:
            f.write("{not json")
        self.assertEqual(self.audit(jsonl=False), 2)
        self.assertEqual(self.status("B4"), "FAIL")

    def test_crashing_phase_is_a_failed_finding_with_a_report(self):
        self.build()
        orig = asc.Audit.statistics
        asc.Audit.statistics = lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
        self.addCleanup(setattr, asc.Audit, "statistics", orig)
        self.assertEqual(self.audit(), 2)
        self.assertEqual(self.status("P4"), "FAIL")
        self.assertTrue(os.path.exists(os.path.join(self.out, "quality_report.md")))


class ValidatorProvenanceTest(AuditBase):
    def test_validator_version_is_recorded_and_compared_with_uv_lock(self):
        self.build()
        self.audit()
        v = json.load(open(os.path.join(self.out, "provenance_report.json")))["provenance"]["validator"]
        self.assertTrue(v["jsonschema_installed"])
        self.assertTrue(v["jsonschema_in_uv_lock"])
        self.assertTrue(v["default_dialect_without_$schema"].endswith("Validator"))
        self.assertIn(self.status("E1"), ("INFO", "WARN"))
        self.assertEqual(self.status("E1") == "INFO", v["jsonschema_installed"] == v["jsonschema_in_uv_lock"])


class MetricDefinitionTest(unittest.TestCase):
    def test_schema_metrics_known_fixture(self):
        schema = {"type": "object", "required": ["a", "b"], "properties": {
            "a": {"type": "string", "enum": ["x", "y"], "minLength": 1},
            "b": {"type": "array", "items": {"type": "object", "properties": {"c": {"type": "integer", "minimum": 0, "maximum": 5}}}},
            "d": {"$ref": "#/definitions/d"}, "e": {"$ref": "http://example.test/s.json#/x"},
            "f": {"anyOf": [{"type": "string", "pattern": "^a"}, {"type": "null"}]}},
            "definitions": {"d": {"type": "boolean"}}}
        m, ext = asc.schema_metrics(schema)
        self.assertEqual(m["schema_properties"], 6)                # a b d e f c
        self.assertEqual(m["schema_required"], 2)
        self.assertEqual(m["schema_depth"], 4)                     # root -> b -> items -> c
        self.assertEqual(m["schema_arrays"], 1)
        self.assertEqual(m["schema_objects"], 2)
        self.assertEqual(m["schema_enums"], 1)
        self.assertEqual(m["schema_constraints"], 4)               # minLength, minimum, maximum, ... pattern is not in this list? it is not: 3 + 0
        self.assertEqual(m["schema_patterns"], 1)
        self.assertEqual(m["schema_combinators"], 1)
        self.assertEqual(m["schema_refs"], 2)
        self.assertEqual(m["schema_external_refs"], 1)
        self.assertEqual(m["schema_defs"], 1)
        self.assertEqual(ext, ["http://example.test/s.json"])
        self.assertEqual(m["schema_bytes"], len(json.dumps(schema, separators=(",", ":")).encode()))

    def test_instance_metrics_known_fixture(self):
        m = asc.instance_metrics({"a": 1, "b": {"c": [1, 2, {"d": "x"}], "e": None}})
        self.assertEqual(m["json_leaves"], 5)                      # a, c[0], c[1], c[2].d, e (None is a leaf)
        self.assertEqual(m["json_depth"], 4)
        self.assertEqual(m["json_fields"], 5)                      # keys a, b, c, e, d
        self.assertEqual(m["json_nested_objects"], 2)              # b and the object inside c
        self.assertEqual(m["json_max_array_len"], 3)
        self.assertEqual(m["json_max_object_size"], 2)

    def test_change_summary_and_coverage(self):
        paths, kinds, depths = asc.change_summary({"a": 1, "b": {"c": "x"}, "d": 1}, {"a": "1", "b": {}, "e": 2})
        self.assertEqual(sorted(map(tuple, paths)), [("a",), ("b",), ("b", "c"), ("d",), ("e",)])
        self.assertEqual(kinds["type_change"], 1)
        self.assertEqual(kinds["removed"], 2)
        self.assertEqual(kinds["added"], 1)
        self.assertEqual(max(depths), 2)
        self.assertTrue(asc._covered(("b", "c"), [("b",)]))
        self.assertFalse(asc._covered(("d",), [("b",)]))

    def test_descriptive_statistics_and_bootstrap(self):
        s = asc._stats([1, 2, 3, 4, 5])
        self.assertEqual((s["n"], s["mean"], s["median"], s["min"], s["max"], s["p25"], s["p75"]), (5, 3, 3, 1, 5, 2, 4))
        self.assertAlmostEqual(s["sd"], 1.5811, places=3)
        self.assertEqual(asc._stats([]), {"n": 0})
        a, b = {"s1": [1, 1], "s2": [3]}, {"s1": [5, 5], "s2": [7]}
        ci1, ci2 = asc._cluster_bootstrap(a, b, 100, 1), asc._cluster_bootstrap(a, b, 100, 1)
        self.assertEqual(ci1, ci2)                                  # seeded: reproducible
        self.assertTrue(ci1[0] <= 4 <= ci1[1] or abs(ci1[0] - 4) < 2)


if __name__ == "__main__":
    unittest.main()
