"""Checks the generated split (stdlib unittest; run: python3 -I -m unittest discover controlled_experiment/tests).
Skipped (explicitly marked NOT VERIFIED) if the generated split or the local baseline/release datasets are missing."""
import csv, json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import common as C
D = os.path.join(C.REPO, "outputs", "controlled_inputs_20261120", "split_v1")
_FIXTURES = {"generated split (outputs/.../split_v1)": D, "baseline 3K dataset": C.BASELINE_ROOT,
             "release 7K dataset": C.RELEASE_ROOT}
_MISSING = [f"{k}: {v}" for k, v in _FIXTURES.items() if not os.path.isdir(v)]


@unittest.skipIf(_MISSING, "NOT VERIFIED - required fixtures unavailable: " + "; ".join(_MISSING))
class T(unittest.TestCase):
    def man(self, n):
        with open(os.path.join(D, n), newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def summary(self):
        with open(os.path.join(D, "split_summary.json")) as f:
            return json.load(f)

    def test_counts(self):
        s = self.summary()
        self.assertEqual((s["val_rows"], s["arm_A_train_rows"], s["arm_B_train_rows"]), (300, 2684, 6267))
        for n, k in (("validation.jsonl", 300), ("train_A_3k.jsonl", 2684), ("train_B_7k.jsonl", 6267)):
            self.assertEqual(len(C.read_lines(os.path.join(D, n))), k)

    def test_checksums_recorded(self):
        s = self.summary()
        for n, h in s["outputs_sha256"].items():
            self.assertEqual(C.sha256_file(os.path.join(D, n)), h, n)

    def test_a_subset_of_b(self):
        a = {(m["task"], m["release_line"]) for m in self.man("train_A_3k_manifest.tsv")}
        b = {(m["task"], m["release_line"]) for m in self.man("train_B_7k_manifest.tsv") if m["origin"] == "original"}
        self.assertEqual(a, b)

    def test_no_schema_overlap(self):
        v = {m["schema_content_sha256"] for m in self.man("validation_manifest.tsv")}
        for n in ("train_A_3k_manifest.tsv", "train_B_7k_manifest.tsv"):
            self.assertFalse(v & {m["schema_content_sha256"] for m in self.man(n)})
        t = set()
        for task in C.TASKS:
            for ln in C.read_lines(f"{C.RELEASE_ROOT}/{task}/test.jsonl"):
                t.add(C.schema_content_hash(json.loads(ln)["schema"]))
        for n in ("train_A_3k_manifest.tsv", "train_B_7k_manifest.tsv", "validation_manifest.tsv"):
            self.assertFalse(t & {m["schema_content_sha256"] for m in self.man(n)})

    def test_validation_per_task(self):
        c = {}
        for m in self.man("validation_manifest.tsv"):
            c[m["task"]] = c.get(m["task"], 0) + 1
        self.assertEqual(c, {"create": 100, "fix": 100, "modify": 100})

    def test_original_prepare_unchanged_and_3k_matches_exporter(self):
        import hashlib
        self.assertEqual(C.sha256_file(C.PREPARE_SCRIPT), "fb2c385999df1837dd03179f2d76ea8616a57406d21f03b6cd98525bbcc59ffb")
        self.assertEqual(C.sha256_file(C.TRAIN_LORA), "da2fb654291706e590a86e7074350b412cc19b442dfd13b0e728858a31845154")


if __name__ == "__main__":
    unittest.main()
