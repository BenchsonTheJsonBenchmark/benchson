import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "src", "merge_scale_dataset.py")


def _rec(task, tag):
    base = {"name": f"{tag}.json", "source": "s", "subset": "x",
            "schema": {"type": "object", "properties": {tag: {"type": "string"}}}}
    if task == "create":
        base.update(description="d", reference_json={"a": 1})
    elif task == "fix":
        base.update(erroneous_json={"a": 1}, valid_json={"a": "1"})
    else:
        base.update(data={}, instructions="i", ground_truth={}, modification={})
    return base


def _put(root, task, split, tag):
    d = os.path.join(root, f"benchmark_{task}", split)
    os.makedirs(d, exist_ok=True)
    rec = _rec(task, tag)
    with open(os.path.join(d, f"s__x__{tag}.json.json"), "w") as f:
        json.dump(rec, f)


class MergeTest(unittest.TestCase):
    def run_cli(self, base, new, out, *extra):
        return subprocess.run([sys.executable, CLI, "--baseline", base, "--new", new, "--out-dir", out, *extra],
                              capture_output=True, text=True)

    def test_merge_ok_and_leak_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, new, out = (os.path.join(tmp, n) for n in ("base", "new", "out"))
            for t in ("create", "fix", "modify"):
                _put(base, t, "train", "a"); _put(base, t, "test", "t"); _put(new, t, "train", "n")
            r = self.run_cli(base, new, out, "--jsonl-out", os.path.join(tmp, "jl"))
            self.assertEqual(r.returncode, 0, r.stderr)
            m = json.load(open(os.path.join(out, "merge_manifest.json")))
            self.assertEqual(m["merged_counts"]["fix"], {"train": 2, "test": 1})
            self.assertEqual(len(open(os.path.join(tmp, "jl", "fix", "train.jsonl")).readlines()), 2)
            self.assertNotEqual(self.run_cli(base, new, out).returncode, 0)  # non-empty out refused
            # a new example whose schema equals a published test schema is rejected
            leak = os.path.join(tmp, "new2"); _put(leak, "create", "train", "t")
            r = self.run_cli(base, leak, os.path.join(tmp, "out2"))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("TEST set", r.stderr)


    def test_controlled_reuse_flag_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, new = os.path.join(tmp, "base"), os.path.join(tmp, "new")
            for t in ("create", "fix", "modify"):
                _put(base, t, "train", "a"); _put(base, t, "test", "t")
                d = os.path.join(new, f"benchmark_{t}", "train"); os.makedirs(d)
                rec = _rec(t, "a")                                  # same schema as published train example "a"
                key = {"create": "reference_json", "fix": "valid_json", "modify": "data"}[t]
                rec[key] = {"a": 999}                               # but a genuinely different reference
                if t == "fix": rec["erroneous_json"] = {"a": 998}
                if t == "modify": rec["ground_truth"] = {"a": 1000}
                json.dump(rec, open(os.path.join(d, "s__x__a.json.json__a1.json"), "w"))
            self.assertNotEqual(self.run_cli(base, new, os.path.join(tmp, "o1")).returncode, 0)   # reuse not allowed
            r = self.run_cli(base, new, os.path.join(tmp, "o2"), "--allow-train-schema-reuse")
            self.assertEqual(r.returncode, 0, r.stderr)
            m = json.load(open(os.path.join(tmp, "o2", "merge_manifest.json")))
            self.assertEqual(m["new_schemas_also_in_published_train"], 1)
            self.assertEqual(m["duplicate_statistics"]["exact_duplicates_vs_published"], 0)
            # now make the new create example identical to the published one
            same = os.path.join(tmp, "new_same"); d = os.path.join(same, "benchmark_create", "train"); os.makedirs(d)
            json.dump(_rec("create", "a"), open(os.path.join(d, "s__x__a.json.json__a1.json"), "w"))
            r = self.run_cli(base, same, os.path.join(tmp, "o3"), "--allow-train-schema-reuse")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("duplicate", r.stderr)


if __name__ == "__main__":
    unittest.main()
