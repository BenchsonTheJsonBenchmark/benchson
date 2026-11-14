"""Regression tests for src/merge_scale_dataset.py. Small synthetic fixtures only: no network, no Hugging Face, no cluster."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "src", "merge_scale_dataset.py")
sys.path.insert(0, os.path.join(ROOT, "src"))
import merge_scale_dataset as msd                                            # noqa: E402
from data_generation.published import TASKS, encode_record, record_id        # noqa: E402

REV = "a" * 40
CHAIN = ["c01_100-r0", "c02_101-r0"]
OPTS = ["--expected-revision", REV, "--expected-new", "create=2,fix=2,modify=2", "--expected-baseline-train", "2",
        "--expected-baseline-test", "1", "--expected-chunks", "2"]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rec(task, schema_tag, val):
    base = {"name": schema_tag, "source": "s", "subset": "x",
            "schema": {"type": "object", "properties": {schema_tag: {"type": "string"}}}}
    if task == "create":
        base.update(description=f"d {val}", reference_json={"v": val})
    elif task == "fix":
        base.update(erroneous_json={"v": val, "bad": 1}, valid_json={"v": val})
    else:
        base.update(data={"v": val}, instructions=f"i {val}", ground_truth={"v": val, "g": 1}, modification={"m": val})
    return base


def put(root, task, split, name, record):
    d = os.path.join(root, f"benchmark_{task}", split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def make_baseline(root, train=("b0", "b1"), test=("t0",)):
    rows, hashes = {}, {}
    for task in TASKS:
        for split, tags in (("train", train), ("test", test)):
            lines = []
            for tag in tags:
                r = rec(task, tag, tag)
                put(root, task, split, record_id(r) + ".json", r)
                lines.append(json.dumps(encode_record(r)))
            path = os.path.join(root, "original_files", task, f"{split}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            hashes[f"{task}/{split}.jsonl"] = sha(path)
    manifest = {"dataset": "x/y", "revision": REV, "source_file_sha256": hashes,
                "counts": {f"{t}/{s}": n for t in TASKS for s, n in (("train", len(train)), ("test", len(test)))}}
    with open(os.path.join(root, "published_manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def make_new(root, tags=("n0", "n1"), schema_of=None):
    """Merged-chunks directory. tags are the example values; schema_of maps a value to the schema tag it uses."""
    schema_of = schema_of or {}
    files = {t: {} for t in TASKS}
    for task in TASKS:
        os.makedirs(os.path.join(root, f"benchmark_{task}", "test"), exist_ok=True)
        for i, tag in enumerate(tags):
            stag = schema_of.get(tag, tag)
            r = rec(task, stag, tag)
            name = f"s__x__{stag}__a{i + 1}.json"
            put(root, task, "train", name, r)
            files[task][name] = sha(os.path.join(root, f"benchmark_{task}", "train", name))
    manifest = {"format": "benchson-merged-chunks/1", "totals": {t: len(tags) for t in TASKS},
                "chain": [{"dir": d, "state_sha256": "1" * 64, "success_marker_sha256": "2" * 64} for d in CHAIN],
                "files": files}
    write_manifest(root, manifest)
    return manifest


def write_manifest(root, manifest):
    with open(os.path.join(root, "merged_chunks_manifest.json"), "w") as f:
        json.dump(manifest, f)


def tree(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = sha(p)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base, self.new, self.out = (os.path.join(self.tmp, n) for n in ("base", "new", "out"))
        make_baseline(self.base)
        self.new_manifest = make_new(self.new)

    def cli(self, *extra, out=None, opts=OPTS):
        return subprocess.run([sys.executable, CLI, "--baseline", self.base, "--new", self.new, "--out-dir", out or self.out,
                               *opts, *extra], capture_output=True, text=True)

    def assert_refused(self, r, text=None):
        self.assertNotEqual(r.returncode, 0, r.stdout)
        if text:
            self.assertIn(text, r.stderr)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])


class SuccessTest(Base):
    def test_exact_quotas_manifest_and_byte_identity(self):
        before_base, before_new = tree(self.base), tree(self.new)
        r = self.cli("--allow-train-schema-reuse", "--jsonl-out", os.path.join(self.tmp, "jl"))
        self.assertEqual(r.returncode, 0, r.stderr)
        for t in TASKS:
            self.assertEqual(len(os.listdir(os.path.join(self.out, f"benchmark_{t}", "train"))), 4)
            self.assertEqual(len(os.listdir(os.path.join(self.out, f"benchmark_{t}", "test"))), 1)
        m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
        self.assertEqual(m["merged_counts"]["fix"], {"train": 4, "test": 1})
        self.assertEqual(m["output_counts"]["train_total"], 12)
        self.assertEqual(m["output_counts"]["test_total"], 3)
        self.assertEqual(m["baseline_provenance"]["revision"], REV)
        self.assertEqual(m["baseline_provenance"]["published_manifest_sha256"], sha(os.path.join(self.base, "published_manifest.json")))
        self.assertEqual(m["new_provenance"]["merged_chunks_manifest_sha256"], sha(os.path.join(self.new, "merged_chunks_manifest.json")))
        self.assertEqual([c["dir"] for c in m["new_provenance"]["chain"]], CHAIN)
        self.assertEqual(m["input_counts"]["new_train"], {"create": 2, "fix": 2, "modify": 2})
        self.assertIn("NOT re-read", m["new_provenance"]["chain_verification"])
        self.assertTrue(m["limitations"])
        # every recorded output hash equals the real file and its source (baseline test + train, new)
        for t in TASKS:
            for split in ("train", "test"):
                for name, digest in m["output_file_sha256"][t][split].items():
                    self.assertEqual(sha(os.path.join(self.out, f"benchmark_{t}", split, name)), digest)
        out_tree = tree(self.out)
        for src in (before_base, before_new):
            for rel, digest in src.items():
                if rel.startswith("benchmark_"):
                    self.assertEqual(out_tree[rel], digest, rel)
        # the sources were only read
        self.assertEqual(tree(self.base), before_base)
        self.assertEqual(tree(self.new), before_new)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_jsonl_counts_and_content(self):
        jl = os.path.join(self.tmp, "jl")
        r = self.cli("--allow-train-schema-reuse", "--jsonl-out", jl)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
        self.assertEqual(m["jsonl_export"]["counts"]["fix/train"], 4)
        for t in TASKS:
            train = [json.loads(l) for l in open(os.path.join(jl, t, "train.jsonl"))]
            test = [json.loads(l) for l in open(os.path.join(jl, t, "test.jsonl"))]
            self.assertEqual((len(train), len(test)), (4, 1))
            for line in train + test:                       # published format: JSON fields are strings
                self.assertTrue(all(isinstance(line[k], str) for k in ("schema",)))
            # lines of baseline examples are byte-identical to the published JSONL
            published = {l.rstrip("\n") for l in open(os.path.join(self.base, "original_files", t, "train.jsonl"))}
            ours = {l.rstrip("\n") for l in open(os.path.join(jl, t, "train.jsonl"))}
            self.assertTrue(published <= ours)
        self.assertEqual(self.cli(out=os.path.join(self.tmp, "out2"), *["--jsonl-out", jl]).returncode, 1)   # existing jsonl dir refused

    def test_chain_verification_with_chunk_directories(self):
        fake = {"code": 0, "verdict": "SUCCESS", "technical": [], "invalid": [], "shortfall": [], "cumulative": {t: 2 for t in TASKS}}
        files = self.new_manifest["files"]
        states = [{"self_sha256": "1" * 64, "files": {t: (dict(files[t]) if i == 1 else {}) for t in TASKS}} for i in range(2)]
        chain = [(os.path.join(self.tmp, d), s) for d, s in zip(CHAIN, states)]
        import chunked_run
        from data_generation import chunking
        self.chunks = os.path.join(self.tmp, "chunks"); os.makedirs(self.chunks)
        argv = ["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse",
                "--chunks-root", self.chunks, "--anchor-file", "a", "--reserved-file", "r"]
        with mock.patch.object(chunked_run, "validate_chunk", return_value=fake), \
                mock.patch.object(chunking, "load_chain", return_value=chain), \
                mock.patch.object(chunking, "load_seal", return_value={"self_sha256": "2" * 64}):
            msd.main(argv)
            m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
            self.assertTrue(m["new_provenance"]["chain_verification"]["validated"])
            shutil.rmtree(self.out)
            states[0]["self_sha256"] = "9" * 64            # a chunk state that is not the one the merge recorded
            with self.assertRaises(SystemExit) as cm:
                msd.main(argv)
            self.assertIn("chunk_state.json hash differs", str(cm.exception))
        self.assertFalse(os.path.exists(self.out))


class NewDatasetIntegrityTest(Base):
    def test_missing_new_example(self):
        os.remove(os.path.join(self.new, "benchmark_fix", "train", "s__x__n1__a2.json"))
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "files on disk")

    def test_unexpected_additional_example(self):
        put(self.new, "create", "train", "s__x__zz__a9.json", rec("create", "zz", "zz"))
        self.assert_refused(self.cli(), "unexpected")

    def test_unexpected_entries_and_test_examples(self):
        put(self.new, "modify", "test", "s__x__q.json", rec("modify", "q", "q"))
        self.assert_refused(self.cli(), "test examples")

    def test_malformed_manifest_variants(self):
        good = json.load(open(os.path.join(self.new, "merged_chunks_manifest.json")))
        cases = {"not json": None, "format": dict(good, format="other/9"), "chunks": dict(good, chain=good["chain"][:1]),
                 "order": dict(good, chain=list(reversed(good["chain"]))), "totals": dict(good, totals={"create": 2, "fix": 2, "modify": 3}),
                 "hash": dict(good, chain=[dict(good["chain"][0], state_sha256="zz"), good["chain"][1]]), "files": {k: v for k, v in good.items() if k != "files"}}
        for label, m in cases.items():
            with self.subTest(label):
                path = os.path.join(self.new, "merged_chunks_manifest.json")
                if m is None:
                    open(path, "w").write("{not json")
                else:
                    write_manifest(self.new, m)
                self.assert_refused(self.cli("--allow-train-schema-reuse"))
        os.remove(os.path.join(self.new, "merged_chunks_manifest.json"))
        self.assert_refused(self.cli(), "not found")

    def test_source_sha256_mismatch(self):
        p = os.path.join(self.new, "benchmark_create", "train", "s__x__n0__a1.json")
        with open(p, "a") as f:
            f.write(" ")
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "SHA256 differs")

    def test_wrong_quota_is_refused(self):
        r = subprocess.run([sys.executable, CLI, "--baseline", self.base, "--new", self.new, "--out-dir", self.out,
                            "--expected-revision", REV, "--expected-baseline-train", "2", "--expected-baseline-test", "1",
                            "--expected-chunks", "2"], capture_output=True, text=True)   # default quota 1334/1333/1333
        self.assert_refused(r, "expected")


class BaselineIntegrityTest(Base):
    def test_missing_baseline_example(self):
        os.remove(os.path.join(self.base, "benchmark_create", "train", "s__x__b1.json"))
        self.assert_refused(self.cli(), "Baseline integrity")

    def test_baseline_file_differs_from_published_row(self):
        p = os.path.join(self.base, "benchmark_fix", "test", "s__x__t0.json")
        d = json.load(open(p)); d["valid_json"] = {"v": "tampered"}
        json.dump(d, open(p, "w"))
        self.assert_refused(self.cli(), "differs from the published JSONL row")

    def test_published_jsonl_changed_or_wrong_revision(self):
        p = os.path.join(self.base, "original_files", "create", "train.jsonl")
        open(p, "a").write("\n")
        self.assert_refused(self.cli(), "SHA256")
        make_baseline(self.base)
        self.assert_refused(self.cli(opts=["--expected-revision", "b" * 40, *OPTS[2:]]), "pinned revision")

    def test_missing_manifest_or_wrong_counts(self):
        m = json.load(open(os.path.join(self.base, "published_manifest.json")))
        m["counts"]["create/test"] = 149
        json.dump(m, open(os.path.join(self.base, "published_manifest.json"), "w"))
        self.assert_refused(self.cli(), "counts")
        os.remove(os.path.join(self.base, "published_manifest.json"))
        self.assert_refused(self.cli(), "not found")


class ExampleChecksTest(Base):
    def test_filename_collision_with_baseline_and_case(self):
        shutil.rmtree(self.new)
        os.makedirs(self.new)
        make_new(self.new)
        old = os.path.join(self.new, "benchmark_create", "train")
        os.rename(os.path.join(old, "s__x__n0__a1.json"), os.path.join(old, "s__x__b0.json"))
        m = json.load(open(os.path.join(self.new, "merged_chunks_manifest.json")))
        m["files"]["create"]["s__x__b0.json"] = m["files"]["create"].pop("s__x__n0__a1.json")
        write_manifest(self.new, m)
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "also exist in the baseline")
        os.rename(os.path.join(old, "s__x__b0.json"), os.path.join(old, "S__X__B0.JSON".replace("JSON", "json")))
        m["files"]["create"]["S__X__B0.json"] = m["files"]["create"].pop("s__x__b0.json")
        write_manifest(self.new, m)
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "letter case")

    def test_exact_duplicates(self):
        shutil.rmtree(self.new); make_new(self.new, tags=("b0", "n1"), schema_of={"b0": "b0"})      # identical to published b0
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "duplicate")
        shutil.rmtree(self.new); make_new(self.new, tags=("n0", "n0"), schema_of={})                 # same example twice (a1, a2)
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "duplicate")

    def test_test_schema_leakage(self):
        shutil.rmtree(self.new); make_new(self.new, tags=("n0", "n1"), schema_of={"n0": "t0"})
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "TEST set")

    def test_controlled_train_schema_reuse(self):
        shutil.rmtree(self.new); make_new(self.new, tags=("n0", "n1"), schema_of={"n0": "b0"})      # new reference on a published train schema
        self.assert_refused(self.cli(), "already used in the published train set")
        r = self.cli("--allow-train-schema-reuse")
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
        self.assertEqual(m["new_schemas_also_in_published_train"], 1)
        self.assertEqual(m["duplicate_statistics"]["exact_duplicates_vs_published"], 0)
        self.assertEqual(m["policy"]["schema_reuse"], "controlled train-schema reuse")

    def test_same_reference_and_cap(self):
        shutil.rmtree(self.new); make_new(self.new, tags=("b0", "n1"), schema_of={"b0": "b0"})
        # b0 reference on the b0 schema is the published one
        self.assert_refused(self.cli("--allow-train-schema-reuse"))
        shutil.rmtree(self.new); make_new(self.new, tags=("n0", "n1"), schema_of={"n0": "n1"})       # two attempts on schema n1
        r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "1")
        self.assert_refused(r, "cap is 1")
        r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "2")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_missing_required_field(self):
        p = os.path.join(self.new, "benchmark_create", "train", "s__x__n0__a1.json")
        d = json.load(open(p)); d.pop("description"); json.dump(d, open(p, "w"))
        m = json.load(open(os.path.join(self.new, "merged_chunks_manifest.json")))
        m["files"]["create"]["s__x__n0__a1.json"] = sha(p); write_manifest(self.new, m)
        self.assert_refused(self.cli("--allow-train-schema-reuse"), "required fields")


class OutputSafetyTest(Base):
    def test_destination_exists_is_never_touched(self):
        for kind in ("empty dir", "dir with file", "file"):
            with self.subTest(kind):
                shutil.rmtree(self.out, ignore_errors=True)
                if os.path.exists(self.out):
                    os.remove(self.out)
                if kind == "file":
                    open(self.out, "w").write("keep")
                else:
                    os.makedirs(self.out)
                    if kind == "dir with file":
                        open(os.path.join(self.out, "keep.txt"), "w").write("keep")
                before = tree(self.out) if os.path.isdir(self.out) else open(self.out).read()
                r = self.cli("--allow-train-schema-reuse")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("already exists", r.stderr)
                self.assertEqual(tree(self.out) if os.path.isdir(self.out) else open(self.out).read(), before)
        for bad in (self.base, os.path.join(self.new, "sub")):
            self.assertNotEqual(self.cli(out=bad).returncode, 0)

    def run_main(self, *extra):
        with self.assertRaises(SystemExit) as cm:
            msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse", *extra])
        return cm.exception.code

    def test_failure_while_copying_leaves_nothing(self):
        calls = []
        real = msd._copy_exclusive

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 7:
                raise OSError("disk full")
            return real(src, dst)
        with mock.patch.object(msd, "_copy_exclusive", flaky):
            with self.assertRaises(OSError):
                msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse"])
        self.assertTrue(len(calls) >= 7)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_failure_writing_manifest_or_verifying_leaves_nothing(self):
        for target in ("_write_manifest", "_verify_output"):
            with self.subTest(target):
                with mock.patch.object(msd, target, side_effect=RuntimeError("boom")):
                    with self.assertRaises(RuntimeError):
                        msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse"])
                self.assertFalse(os.path.exists(self.out))
                self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_jsonl_failure_leaves_neither_directory(self):
        with mock.patch.object(msd, "_write_jsonl", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse",
                          "--jsonl-out", os.path.join(self.tmp, "jl")])
        self.assertFalse(os.path.exists(self.out) or os.path.exists(os.path.join(self.tmp, "jl")))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_copy_never_overwrites(self):
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        open(a, "w").write("x"); open(b, "w").write("keep")
        with self.assertRaises(FileExistsError):
            msd._copy_exclusive(a, b)
        self.assertEqual(open(b).read(), "keep")

    def test_corrupted_copy_is_detected_before_publishing(self):
        real = msd._copy_exclusive

        def corrupt(src, dst):
            real(src, dst)
            if "benchmark_fix/test" in dst or "benchmark_fix\\test" in dst:
                open(dst, "ab").write(b" ")
        with mock.patch.object(msd, "_copy_exclusive", corrupt):
            with self.assertRaises(SystemExit):
                msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse"])
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])


class LedgerCompatibleCapTest(Base):
    """--max-new-references-per-schema counts like chunked_run.validate_chunk: per schema id taken from <id>__a<attempt>.json."""

    def test_counts_per_schema_id_not_per_schema_content(self):
        # two different schema ids that carry identical schema content: each id has ONE reference, the cap is per id
        shutil.rmtree(self.new)
        make_new(self.new, tags=("n0", "n1"), schema_of={"n0": "n1"})
        old = os.path.join(self.new, "benchmark_create", "train")
        self.assertEqual(sorted(os.listdir(old)), ["s__x__n1__a1.json", "s__x__n1__a2.json"])   # one id, two attempts
        r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "1")
        self.assert_refused(r, "global cap is 1")
        shutil.rmtree(self.new)
        make_new(self.new, tags=("n0", "n1"))
        for task in TASKS:           # same schema content under two ids: not a violation of the per-id ledger rule
            d = os.path.join(self.new, f"benchmark_{task}", "train")
            os.rename(os.path.join(d, "s__x__n1__a2.json"), os.path.join(d, "s__x__n0__a2.json"))
            m = json.load(open(os.path.join(self.new, "merged_chunks_manifest.json")))
            m["files"][task]["s__x__n0__a2.json"] = m["files"][task].pop("s__x__n1__a2.json")
            write_manifest(self.new, m)
        r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "2")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_one_attempt_shared_by_three_tasks_is_one_reference(self):
        r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "1")     # n0 and n1 are different ids
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
        self.assertEqual(m["policy"]["observed_max_new_references_per_schema"], 1)

    def test_cap_exceeded_per_task_and_per_schema_and_boundary(self):
        shutil.rmtree(self.new)
        make_new(self.new, tags=("n0", "n1"), schema_of={"n0": "n1"})
        for cap, ok in ((1, False), (2, True), (3, True)):
            shutil.rmtree(self.out, ignore_errors=True)
            r = self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", str(cap))
            self.assertEqual(r.returncode == 0, ok, (cap, r.stderr))        # strictly more than the cap fails, equal passes

    def test_unparsable_name_cannot_be_counted(self):
        shutil.rmtree(self.new)
        make_new(self.new)
        d = os.path.join(self.new, "benchmark_create", "train")
        os.rename(os.path.join(d, "s__x__n0__a1.json"), os.path.join(d, "weird.json"))
        m = json.load(open(os.path.join(self.new, "merged_chunks_manifest.json")))
        m["files"]["create"]["weird.json"] = m["files"]["create"].pop("s__x__n0__a1.json")
        write_manifest(self.new, m)
        self.assert_refused(self.cli("--allow-train-schema-reuse", "--max-new-references-per-schema", "3"), "cannot count")


class DestinationOverlapTest(Base):
    def assert_untouched_refusal(self, out, *extra, text=None):
        before = (tree(self.base), tree(self.new))
        r = self.cli(*extra, out=out)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        if text:
            self.assertIn(text, r.stderr)
        self.assertEqual((tree(self.base), tree(self.new)), before)
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_out_dir_equal_inside_or_containing_a_source(self):
        for label, out in (("equal base", self.base), ("equal new", self.new),
                           ("inside base", os.path.join(self.base, "merged")), ("inside new", os.path.join(self.new, "benchmark_fix", "x")),
                           ("parent of both", self.tmp), ("letter case", os.path.join(self.tmp, "BASE"))):
            with self.subTest(label):
                self.assert_untouched_refusal(out)

    def test_symlinks_into_sources(self):
        link = os.path.join(self.tmp, "link_to_base")
        os.symlink(self.base, link)
        for label, out in (("through a symlink", os.path.join(link, "merged")), ("the symlink itself", link)):
            with self.subTest(label):
                self.assert_untouched_refusal(out)
        dangling = os.path.join(self.tmp, "dangling")
        os.symlink(os.path.join(self.tmp, "nowhere"), dangling)
        self.assert_untouched_refusal(dangling)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "nowhere")))
        nested = os.path.join(self.tmp, "chain1", "chain2")           # symlink chain pointing into the new run
        os.makedirs(os.path.dirname(nested))
        os.symlink(os.path.join(self.new, "benchmark_fix"), nested)
        self.assert_untouched_refusal(os.path.join(nested, "deeper", "out"))

    def test_chunk_directories_are_protected_too(self):
        chunks = os.path.join(self.tmp, "chunks"); os.makedirs(os.path.join(chunks, "c01"))
        for out in (os.path.join(chunks, "merged"), chunks):
            with self.subTest(out):
                self.assert_untouched_refusal(out, "--chunks-root", chunks, "--anchor-file", self.base, "--reserved-file", self.base)
        self.assertEqual(os.listdir(chunks), ["c01"])

    def test_jsonl_out_overlaps(self):
        jl = os.path.join(self.tmp, "jl")
        for label, out, jsonl in (("same", self.out, self.out), ("jsonl inside out", self.out, os.path.join(self.out, "jl")),
                                  ("out inside jsonl", os.path.join(jl, "out"), jl), ("jsonl inside base", self.out, os.path.join(self.base, "jl")),
                                  ("jsonl equals new", self.out, self.new), ("jsonl contains base", self.out, self.tmp)):
            with self.subTest(label):
                self.assert_untouched_refusal(out, "--jsonl-out", jsonl)
                self.assertFalse(os.path.exists(os.path.join(self.base, "jl")))
        self.assertFalse(os.path.exists(self.out) or os.path.exists(jl))
        self.assertTrue(os.path.isdir(self.base) and os.path.isdir(self.new))

    def test_separate_siblings_and_symlinked_parent_are_fine(self):
        elsewhere = os.path.join(self.tmp, "elsewhere"); os.makedirs(elsewhere)
        link = os.path.join(self.tmp, "ln"); os.symlink(elsewhere, link)
        r = self.cli("--allow-train-schema-reuse", "--jsonl-out", os.path.join(self.tmp, "jl"), out=os.path.join(link, "merged"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.isfile(os.path.join(elsewhere, "merged", "merge_manifest.json")))


class TwoStepPublishTest(Base):
    def argv(self, jl):
        return ["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse", "--jsonl-out", jl]

    def no_staging(self):
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])

    def test_second_publish_failure_never_leaves_a_valid_looking_result(self):
        jl = os.path.join(self.tmp, "jl")
        real, calls = msd._publish, []

        def second_fails(staging, final):
            calls.append(final)
            if len(calls) == 2:
                raise OSError("rename failed")
            return real(staging, final)
        before = (tree(self.base), tree(self.new))
        with mock.patch.object(msd, "_publish", second_fails):
            with self.assertRaises(OSError):
                msd.main(self.argv(jl))
        self.assertEqual(calls, [jl, self.out])             # the export goes first, the merge directory (the result) last
        self.assertFalse(os.path.exists(self.out))
        self.assertFalse(os.path.exists(jl))                # the orphan export was renamed away, not left under its real name
        self.assertTrue(os.path.isfile(os.path.join(jl + ".INCOMPLETE-0", "fix", "train.jsonl")))
        self.no_staging()
        self.assertEqual((tree(self.base), tree(self.new)), before)

    def test_first_publish_failure_publishes_nothing(self):
        jl = os.path.join(self.tmp, "jl")
        with mock.patch.object(msd, "_publish", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                msd.main(self.argv(jl))
        self.assertFalse(os.path.exists(self.out) or os.path.exists(jl) or os.path.exists(jl + ".INCOMPLETE-0"))
        self.no_staging()

    def test_destination_appearing_meanwhile_is_not_overwritten(self):
        jl = os.path.join(self.tmp, "jl")
        real, calls = msd._publish, []

        def racer(staging, final):
            calls.append(final)
            if final == self.out:
                os.makedirs(self.out)
                open(os.path.join(self.out, "theirs.txt"), "w").write("keep")
            return real(staging, final)
        with mock.patch.object(msd, "_publish", racer):
            with self.assertRaises(SystemExit):
                msd.main(self.argv(jl))
        self.assertEqual(os.listdir(self.out), ["theirs.txt"])
        self.assertTrue(os.path.isdir(jl + ".INCOMPLETE-0") and not os.path.exists(jl))
        self.no_staging()

    def test_failed_quarantine_is_loud_and_does_not_hide_the_error(self):
        jl = os.path.join(self.tmp, "jl")
        real_publish, real_rename = msd._publish, os.rename
        calls = []

        def second_fails(staging, final):
            calls.append(final)
            if len(calls) == 2:
                raise OSError("rename failed")
            return real_publish(staging, final)

        def no_rename(a, b):
            if str(b).endswith(".INCOMPLETE-0"):
                raise OSError("denied")
            return real_rename(a, b)
        with mock.patch.object(msd, "_publish", second_fails), mock.patch.object(os, "rename", no_rename):
            with self.assertRaises(OSError) as cm:
                msd.main(self.argv(jl))
        self.assertIn("rename failed", str(cm.exception))
        self.assertFalse(os.path.exists(self.out))
        self.no_staging()

    def test_success_publishes_both_and_manifest_describes_the_export(self):
        jl = os.path.join(self.tmp, "jl")
        msd.main(self.argv(jl))
        m = json.load(open(os.path.join(self.out, "merge_manifest.json")))
        self.assertEqual(m["jsonl_export"]["counts"]["create/train"], 4)
        self.assertTrue(os.path.isfile(os.path.join(jl, "create", "test.jsonl")))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge")], [])


class CleanupSafetyTest(Base):
    def test_only_own_plain_staging_directories_are_removed(self):
        own = os.path.join(self.tmp, ".merge_staging_own"); os.makedirs(own)
        msd._remove_own_staging(own, {own: os.path.realpath(own)}, [self.base, self.new])
        self.assertFalse(os.path.exists(own))

    def test_refuses_symlink_foreign_nested_and_swapped_paths(self):
        before = (tree(self.base), tree(self.new))
        link = os.path.join(self.tmp, ".merge_staging_link"); os.symlink(self.base, link)
        msd._remove_own_staging(link, {link: os.path.realpath(link)}, [self.new])            # a symlink to a source
        nested = os.path.join(self.base, ".merge_staging_nested"); os.makedirs(nested)       # a "staging" dir inside a source
        msd._remove_own_staging(nested, {nested: os.path.realpath(nested)}, [self.base])
        foreign = os.path.join(self.tmp, ".merge_staging_foreign"); os.makedirs(foreign)     # never created by this run
        msd._remove_own_staging(foreign, {}, [self.base])
        swapped = os.path.join(self.tmp, ".merge_staging_swapped"); os.makedirs(swapped)     # replaced by a symlink after creation
        created = {swapped: os.path.realpath(swapped)}
        os.rmdir(swapped); os.symlink(self.new, swapped)
        msd._remove_own_staging(swapped, created, [])
        notstaging = os.path.join(self.tmp, "important"); os.makedirs(notstaging)
        msd._remove_own_staging(notstaging, {notstaging: os.path.realpath(notstaging)}, [])
        self.assertTrue(os.path.isdir(nested) and os.path.isdir(foreign) and os.path.isdir(notstaging) and os.path.islink(link))
        os.rmdir(nested)                                                                    # remove our own test artefact
        self.assertEqual((tree(self.base), tree(self.new)), before)

    def test_every_failure_path_leaves_the_sources_untouched(self):
        before = (tree(self.base), tree(self.new))
        for target in ("_build", "_write_manifest", "_verify_output", "_write_jsonl", "_publish"):
            with self.subTest(target):
                with mock.patch.object(msd, target, side_effect=RuntimeError("boom")):
                    with self.assertRaises(RuntimeError):
                        msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.out, *OPTS, "--allow-train-schema-reuse",
                                  "--jsonl-out", os.path.join(self.tmp, "jl")])
                self.assertEqual((tree(self.base), tree(self.new)), before)
                self.assertEqual(sorted(os.listdir(self.tmp)), ["base", "new"])


if __name__ == "__main__":
    unittest.main()
