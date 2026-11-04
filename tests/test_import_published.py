import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_generation.published import encode_record

SHA = "a" * 40


def _rec(task, i, split):
    schema = {"type": "object", "properties": {f"f{split}{i}": {"type": "string"}}}
    base = {"name": f"s{split}{i}.json", "source": "jsb", "subset": "x", "schema": schema}
    if task == "create":
        base.update(description="d", reference_json={"a": 1})
    elif task == "fix":
        base.update(erroneous_json={"a": 1}, valid_json={"a": "1"})
    else:
        base.update(data={"a": 1}, instructions="i", ground_truth={"a": 2}, modification={"op": "x"})
    return base


def _snapshot(root, train=3, test=2):
    for task in ("create", "fix", "modify"):
        os.makedirs(os.path.join(root, task))
        for split, n in (("train", train), ("test", test)):
            with open(os.path.join(root, task, f"{split}.jsonl"), "w") as f:
                for i in range(n):
                    f.write(json.dumps(encode_record(_rec(task, i, split))) + "\n")


class ImportPublishedTest(unittest.TestCase):
    def run_cli(self, *extra, src, out):
        return subprocess.run([sys.executable, os.path.join(ROOT, "src", "import_published_benchmark.py"),
                               "--source-dir", src, "--out-dir", out, *extra],
                              capture_output=True, text=True)

    def test_import_and_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
            _snapshot(src)
            r = self.run_cli("--revision", "main", src=src, out=out)
            self.assertNotEqual(r.returncode, 0)  # branch name rejected
            r = self.run_cli("--revision", SHA, src=src, out=out)
            self.assertNotEqual(r.returncode, 0)  # counts differ from 1000/150
            self.assertIn("Count check failed", r.stderr + r.stdout)
            r = self.run_cli("--revision", SHA, "--allow-count-mismatch", src=src, out=out)
            self.assertEqual(r.returncode, 0, r.stderr)
            train = os.listdir(os.path.join(out, "benchmark_modify", "train"))
            self.assertEqual(len(train), 3)
            path = os.path.join(out, "benchmark_modify", "train", train[0])
            self.assertFalse(os.access(path, os.W_OK) and os.getuid() != 0)
            reserved = json.load(open(os.path.join(out, "reserved_schemas.json")))
            self.assertEqual(reserved["revision"], SHA)
            self.assertEqual(len(reserved["ids"]["train"]), 3)
            self.assertEqual(len(reserved["ids"]["test"]), 2)
            self.assertEqual(len(reserved["ids_by_task"]["fix"]["train"]), 3)
            orig = os.path.join(out, "original_files", "fix", "train.jsonl")
            self.assertEqual(open(orig, "rb").read(), open(os.path.join(src, "fix", "train.jsonl"), "rb").read())
            r = self.run_cli("--revision", SHA, "--allow-count-mismatch", src=src, out=out)
            self.assertNotEqual(r.returncode, 0)  # non-empty target refused


if __name__ == "__main__":
    unittest.main()
