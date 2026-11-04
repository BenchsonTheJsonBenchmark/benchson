import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_import_published import _snapshot, _rec


def run(script, *a, cwd):
    return subprocess.run([sys.executable, os.path.join(ROOT, "src", script), *a], capture_output=True, text=True, cwd=cwd)


class ReleasePrepTest(unittest.TestCase):
    def test_variants_keep_test_and_original_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, base, new = (os.path.join(tmp, n) for n in ("src", "pub", "new"))
            _snapshot(src)
            open(os.path.join(src, "README.md"), "w").write("---\nlicense: mit\n---\n# Original card\n")
            r = run("import_published_benchmark.py", "--revision", "b" * 40, "--source-dir", src, "--out-dir", base,
                    "--allow-count-mismatch", cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            for t in ("create", "fix", "modify"):
                d = os.path.join(new, f"benchmark_{t}", "train"); os.makedirs(d)
                rec = _rec(t, 99, "new"); rec["name"] = "n99.json"
                json.dump(rec, open(os.path.join(d, "jsonschemabench__x__n99.json.json"), "w"))
            repo = ["--new-repo-id", "someone/benchson-derivatives", "--baseline", base]
            self.assertNotEqual(run("prepare_hf_release.py", "--variant", "main", "--new-repo-id", "aviv1ron1/Benchson",
                                    "--baseline", base, cwd=tmp).returncode, 0)  # original repo refused
            r = run("prepare_hf_release.py", "--variant", "main", *repo, cwd=tmp); self.assertEqual(r.returncode, 0, r.stderr)
            r = run("prepare_hf_release.py", "--variant", "scale-only-7k", "--new-dir", new, *repo, cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            for t in ("create", "fix", "modify"):
                o = open(os.path.join(src, t, "train.jsonl"), "rb").read()
                n = open(os.path.join(tmp, "release", "scale-only-7k", t, "train.jsonl"), "rb").read()
                self.assertTrue(n.startswith(o)); self.assertEqual(n.count(b"\n"), o.count(b"\n") + 1)
                for variant in ("main", "scale-only-7k"):
                    self.assertEqual(open(os.path.join(src, t, "test.jsonl"), "rb").read(),
                                     open(os.path.join(tmp, "release", variant, t, "test.jsonl"), "rb").read())
                self.assertEqual(open(os.path.join(tmp, "release", "main", t, "train.jsonl"), "rb").read(),
                                 open(os.path.join(src, t, "train.jsonl"), "rb").read())
            prov = json.load(open(os.path.join(tmp, "release", "scale-only-7k", "PROVENANCE.json")))
            self.assertEqual(prov["original_revision"], "b" * 40)
            self.assertTrue(os.path.exists(os.path.join(tmp, "release", "PUBLISH_COMMANDS.md")))
            self.assertNotEqual(run("prepare_hf_release.py", "--variant", "main", *repo, cwd=tmp).returncode, 0)  # no overwrite


if __name__ == "__main__":
    unittest.main()
