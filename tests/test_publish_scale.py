"""Regression tests for src/publish_scale_only.py (synthetic data; a fake Hugging Face hub; no network, no token)."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_scale_dataset as asc                                           # noqa: E402
import hf_baseline as hfb                                                   # noqa: E402
import merge_scale_dataset as msd                                           # noqa: E402
import publish_scale_only as pub                                            # noqa: E402
import test_audit_scale as T                                                # noqa: E402
from data_generation.published import TASKS                                 # noqa: E402

FAKE_TOKEN = "hf_" + "Z" * 34


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = pub.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue() + err.getvalue()


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeRepo:
    def __init__(self, main_files):
        self.private = True
        self.commits = {}
        self.branches = {}
        self.add_commit("main", main_files)

    def add_commit(self, branch, files):
        sha = hashlib.sha1(json.dumps(sorted((k, hashlib.sha256(v).hexdigest()) for k, v in files.items())).encode() + branch.encode() + str(len(self.commits)).encode()).hexdigest()
        self.commits[sha] = dict(files)
        self.branches[branch] = sha
        return sha

    def resolve(self, rev):
        return self.branches.get(rev, rev)


class FakeApi:
    repo = None

    def __init__(self, token=None):
        assert token == FAKE_TOKEN

    def dataset_info(self, repo, revision=None):
        return NS(private=self.repo.private, sha=self.repo.resolve(revision or "main"))

    def list_repo_refs(self, repo, repo_type=None):
        return NS(branches=[NS(name=n) for n in self.repo.branches])

    def list_repo_tree(self, repo, repo_type=None, revision=None, recursive=False):
        files = self.repo.commits[self.repo.resolve(revision)]
        return [NS(path=p, size=len(d), blob_id=blob_id(d), lfs=None) for p, d in files.items()]

    def create_branch(self, repo, branch, repo_type=None, revision=None, exist_ok=False):
        assert branch not in self.repo.branches
        self.repo.branches[branch] = self.repo.resolve(revision)

    def upload_folder(self, repo_id, repo_type, folder_path, revision, ignore_patterns=None, commit_message="", commit_description=""):
        files = dict(self.repo.commits[self.repo.resolve(revision)])
        for d, _, fs in os.walk(folder_path):
            for f in fs:
                files[os.path.relpath(os.path.join(d, f), folder_path).replace(os.sep, "/")] = open(os.path.join(d, f), "rb").read()
        return NS(oid=self.repo.add_commit(revision, files))


def fake_download(repo, name, repo_type=None, revision=None, cache_dir=None, token=None):
    data = FakeApi.repo.commits[FakeApi.repo.resolve(revision)][name]
    p = os.path.join(cache_dir, revision, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)
    return p


class PublishBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base, self.new, self.merged, self.jl, self.aud, self.rel = (os.path.join(self.tmp, n) for n in ("base", "new", "merged", "jl", "aud", "rel"))
        for patch in (mock.patch.object(pub, "ORIGINAL_ROWS", {"train": 2, "test": 1}), mock.patch.object(pub, "NEW_ROWS", {t: 2 for t in TASKS})):
            patch.start()
            self.addCleanup(patch.stop)
        T.sv.reset_caches()
        T.make_baseline(self.base)
        card = os.path.join(self.base, "original_card")
        os.makedirs(card)
        open(os.path.join(card, "README.md"), "w").write("original card\n")
        m = json.load(open(os.path.join(self.base, "published_manifest.json")))
        m["original_card"] = {"license": "mit", "sha256": T.sha(os.path.join(card, "README.md"))}
        json.dump(m, open(os.path.join(self.base, "published_manifest.json"), "w"))
        T.make_new(self.new)
        with contextlib.redirect_stdout(io.StringIO()):
            msd.main(["--baseline", self.base, "--new", self.new, "--out-dir", self.merged, "--jsonl-out", self.jl, "--allow-train-schema-reuse", *T.EXP])
            code = asc.main(["--baseline", self.base, "--new", self.new, "--merged", self.merged, "--jsonl", self.jl, "--out-dir", self.aud,
                             "--offline", "--bootstrap", "20", *T.EXP])
        self.assertEqual(code, 0)

    def build(self, out=None, **over):
        argv = ["build", "--export", self.jl, "--baseline", self.base, "--new", self.new, "--merged", self.merged, "--audit-dir", self.aud,
                "--out-dir", out or self.rel, "--expected-revision", T.REV]
        return run(argv)

    def lines(self, rel):
        return open(os.path.join(self.jl, rel), "rb").read().split(b"\n")[:-1]

    def write_lines(self, rel, lines):
        open(os.path.join(self.jl, rel), "wb").write(b"\n".join(lines) + b"\n")


class BuildTest(PublishBase):
    def test_build_succeeds_and_is_self_consistent(self):
        before = {p: T.tree(p) for p in (self.base, self.new, self.merged, self.jl, self.aud)}
        code, out = self.build()
        self.assertEqual(code, None, out)
        self.assertEqual(pub.check_release(self.rel), [])
        prov = json.load(open(os.path.join(self.rel, "PROVENANCE.json")))
        for t in TASKS:
            self.assertEqual(prov["counts"][t], {"train": 4, "test": 1})
            self.assertEqual((prov["files"][f"{t}/train.jsonl"]["original_rows"], prov["files"][f"{t}/train.jsonl"]["new_rows"]), (2, 2))
            self.assertEqual(open(os.path.join(self.rel, f"{t}/train.jsonl"), "rb").read(), open(os.path.join(self.jl, t, "train.jsonl"), "rb").read())
        tsv = open(os.path.join(self.rel, "row_origin.tsv")).read().splitlines()
        self.assertEqual(len(tsv), 1 + 3 * 5)
        self.assertEqual(sum(1 for r in tsv if "\tnew\t" in r), 6)
        readme = open(os.path.join(self.rel, "README.md"), encoding="utf-8").read()
        for needle in ("byte for byte", "Controlled schema reuse", "Known defects of the ORIGINAL", "Lower measured structural complexity", "Data integrity", T.REV):
            self.assertIn(needle, readme)
        self.assertNotIn(self.tmp, readme)
        self.assertEqual({p: T.tree(p) for p in (self.base, self.new, self.merged, self.jl, self.aud)}, before)     # inputs untouched
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])
        self.assertFalse(os.path.exists(os.path.join(self.rel, "audit", "validation_failures.csv")))              # bulky files stay local
        self.assertIn("validation_failures.csv", prov["audit"]["files"])

    def assertRefused(self, why=None):
        code, out = self.build()
        self.assertEqual(code, 1, out)
        self.assertFalse(os.path.exists(self.rel))
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".merge_staging_")], [])
        if why:
            self.assertIn(why, out)

    def test_altered_original_row_is_refused(self):
        ln = self.lines("fix/train.jsonl")
        ln[0] = ln[0].replace(b"alpha", b"alphX") if b"alpha" in ln[0] else ln[0].replace(b"bravo", b"bravX")
        self.write_lines("fix/train.jsonl", ln)
        self.assertRefused("fix/train.jsonl")

    def test_altered_new_row_missing_row_extra_row_and_wrong_count(self):
        ln = self.lines("create/train.jsonl")
        self.write_lines("create/train.jsonl", ln[:-1])
        self.assertRefused("create/train.jsonl: 3 rows, expected 4")
        self.write_lines("create/train.jsonl", ln + [ln[0]])
        self.assertRefused("duplicate line")
        self.write_lines("create/train.jsonl", ln[:-1] + [ln[-1].replace(b"charlie", b"charlix").replace(b"delta", b"deltx")])
        self.assertRefused("unexpected line")

    def test_test_split_row_changed_or_leaked_into_train(self):
        ln = self.lines("modify/test.jsonl")
        self.write_lines("modify/test.jsonl", [ln[0].replace(b"tango", b"tangx")])
        self.assertRefused("modify/test.jsonl")

    def test_missing_export_file(self):
        os.remove(os.path.join(self.jl, "fix", "test.jsonl"))
        self.assertRefused("fix/test.jsonl: missing")

    def test_export_not_matching_merged_dataset(self):
        p = os.path.join(self.merged, "benchmark_create", "train", "s__x__charlie__a1.json")
        r = json.load(open(p)); r["description"] = "changed after the export"; json.dump(r, open(p, "w"))
        self.assertRefused("differs from the merged dataset")

    def test_audit_problems_block_the_release(self):
        p = os.path.join(self.aud, "summary.json")
        s = json.load(open(p)); s["data_integrity"]["verdict"] = "NOT INTACT"; json.dump(s, open(p, "w"))
        self.assertRefused("not INTACT")
        s["data_integrity"]["verdict"] = "INTACT"
        for c in s["checks"]:
            if c["id"] == "C1":
                c["status"] = "FAIL"
        json.dump(s, open(p, "w"))
        self.assertRefused("C1")

    def test_machine_specific_paths_in_audit_files_block_the_release(self):
        with open(os.path.join(self.aud, "quality_report.md"), "a") as f:
            f.write("\nsource: /Users/someone/private/data\n")
        self.assertRefused("machine-specific path")

    def test_output_directory_rules(self):
        os.makedirs(self.rel)
        code, out = self.build()
        self.assertNotEqual(code, 0)
        self.assertEqual(os.listdir(self.rel), [])
        code, _ = self.build(out=os.path.join(self.jl, "inside"))
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists(os.path.join(self.jl, "inside")))

    def test_wrong_baseline_revision_is_refused(self):
        code, out = run(["build", "--export", self.jl, "--baseline", self.base, "--new", self.new, "--audit-dir", self.aud, "--out-dir", self.rel,
                         "--expected-revision", "b" * 40])
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists(self.rel))


class CheckReleaseTest(PublishBase):
    def test_tampering_is_detected(self):
        self.assertEqual(self.build()[0], None)
        p = os.path.join(self.rel, "fix", "train.jsonl")
        data = open(p, "rb").read()
        open(p, "wb").write(data.replace(b"charlie0", b"charlie1", 1))
        self.assertTrue(pub.check_release(self.rel))
        open(p, "wb").write(data)
        self.assertEqual(pub.check_release(self.rel), [])
        open(os.path.join(self.rel, "stray.txt"), "w").write("x")
        self.assertTrue(any("file set differs" in x for x in pub.check_release(self.rel)))
        self.assertEqual(run(["check-release", "--release-dir", self.rel])[0], 1)


class HubFlowTest(PublishBase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.build()[0], None)
        main_files = {}
        for t in TASKS:
            for s in ("train", "test"):
                main_files[f"{t}/{s}.jsonl"] = open(os.path.join(self.base, "original_files", t, f"{s}.jsonl"), "rb").read()
        main_files.update({"README.md": b"main card", "NOTICE.md": b"n", "ORIGINAL_DATASET_CARD.md": b"o", "PROVENANCE.json": b"{}", ".gitattributes": b"x"})
        self.repo = FakeRepo(main_files)
        FakeApi.repo = self.repo
        self.main = self.repo.branches["main"]
        hub = NS(HfApi=FakeApi, hf_hub_download=fake_download)
        known = {"files": {k: (blob_id(v), len(v)) for k, v in main_files.items() if k.endswith(".jsonl")}}
        for p in (mock.patch.object(hfb, "_hub", lambda: hub), mock.patch.object(hfb, "_auth_problems", lambda *a: []),
                  mock.patch.object(hfb, "_known", lambda: known), mock.patch.dict(os.environ, {"HF_TOKEN": FAKE_TOKEN}),
                  mock.patch.object(sys.stdin, "isatty", lambda: True)):
            p.start()
            self.addCleanup(p.stop)

    def upload(self, answer=None, **kw):
        answer = answer if answer is not None else f"{pub.DEFAULT_REPO}:{pub.BRANCH}"
        with mock.patch("builtins.input", lambda prompt="": answer):
            return run(["upload", "--release-dir", self.rel, "--expect-main", kw.get("main", self.main)])

    def test_upload_then_verify_remote_passes_and_main_is_unchanged(self):
        code, out = self.upload()
        self.assertEqual(code, None, out)
        self.assertNotIn(FAKE_TOKEN, out)
        self.assertEqual(self.repo.branches["main"], self.main)
        sha = self.repo.branches[pub.BRANCH]
        self.assertNotEqual(sha, self.main)
        self.assertIn(sha, out)
        code, out = run(["verify-remote", "--release-dir", self.rel, "--expect-main", self.main, "--commit", sha])
        self.assertEqual(code, None, out)
        self.assertNotIn(FAKE_TOKEN, out)
        rec = json.load(open(os.path.join(self.tmp, "scale-only-7k.verify_receipt.json")))
        self.assertEqual((rec["result"], rec["commit"], rec["private"]), ("PASS", sha, True))
        self.assertNotIn(FAKE_TOKEN, json.dumps(rec))

    def test_nothing_happens_without_the_exact_confirmation(self):
        code, out = self.upload(answer="yes")
        self.assertNotEqual(code, None)
        self.assertEqual(list(self.repo.branches), ["main"])

    def test_refusals(self):
        self.repo.private = False
        self.assertNotEqual(self.upload()[0], None)
        self.repo.private = True
        self.assertNotEqual(self.upload(main="0" * 40)[0], None)
        self.repo.branches[pub.BRANCH] = self.main
        code, out = self.upload()
        self.assertNotEqual(code, None)
        self.assertIn("already exists", out)
        del self.repo.branches[pub.BRANCH]
        self.assertEqual(list(self.repo.branches), ["main"])
        self.assertNotEqual(run(["upload", "--release-dir", self.rel, "--repo-id", "aviv1ron1/Benchson", "--expect-main", self.main])[0], None)

    def test_corrupted_release_folder_is_not_uploaded(self):
        p = os.path.join(self.rel, "create", "train.jsonl")
        open(p, "ab").write(b"{}\n")
        self.assertNotEqual(self.upload()[0], None)
        self.assertEqual(list(self.repo.branches), ["main"])

    def test_verify_remote_detects_a_changed_remote_file_a_moved_main_and_public_repo(self):
        self.upload()
        sha = self.repo.branches[pub.BRANCH]
        files = dict(self.repo.commits[sha])
        files["fix/train.jsonl"] = files["fix/train.jsonl"].replace(b"charlie0", b"charlie1", 1)
        bad = self.repo.add_commit(pub.BRANCH, files)
        code, out = run(["verify-remote", "--release-dir", self.rel, "--expect-main", self.main, "--commit", bad])
        self.assertEqual(code, 1)
        self.assertIn("fix/train.jsonl", out)
        self.repo.branches[pub.BRANCH] = sha
        code, out = run(["verify-remote", "--release-dir", self.rel, "--expect-main", "1" * 40, "--commit", sha])
        self.assertEqual(code, 1)
        self.assertIn("main changed", out)
        self.repo.private = False
        code, out = run(["verify-remote", "--release-dir", self.rel, "--expect-main", self.main, "--commit", sha])
        self.assertEqual(code, 1)
        self.assertIn("PUBLIC", out)

    def test_extra_remote_file_is_detected(self):
        self.upload()
        sha = self.repo.branches[pub.BRANCH]
        files = dict(self.repo.commits[sha]); files["stray.bin"] = b"x"
        bad = self.repo.add_commit(pub.BRANCH, files)
        code, out = run(["verify-remote", "--release-dir", self.rel, "--expect-main", self.main, "--commit", bad])
        self.assertEqual(code, 1)
        self.assertIn("unexpected file set", out)


if __name__ == "__main__":
    unittest.main()
