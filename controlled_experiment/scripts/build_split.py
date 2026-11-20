"""Deterministic A/B training + shared validation split for the controlled 3K vs 7K experiment (CPU-only, stdlib).

Arm A = original 3K train; Arm B = Scale-Only 7K train. Both use the SAME validation set (original rows only, grouped by
canonical schema content) and the SAME Benchson-Test schema-content leakage exclusion. Training rows are rendered by the
ORIGINAL, unmodified `prepare_benchson_data.py` (so the 100 Modify rows whose instruction it unquotes keep that behaviour;
this is reported, not normalised). Nothing in Test, the datasets, or `original_training/` is modified.

Usage: python3 build_split.py --out DIR [--seed-start 0] [--per-task 100]
"""
import argparse
import csv
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

C.add_repo_paths()


def load_rows(root, split):
    out = {}
    for t in C.TASKS:
        lines = C.read_lines(os.path.join(root, t, f"{split}.jsonl"))
        out[t] = [(ln, json.loads(ln)) for ln in lines]
    return out


def render(root):
    """Original prepare script output, as {task: [raw line bytes]} aligned with the source file order (no skips allowed)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "o.jsonl")
        C.run_original_prepare(root, p)
        rows = C.read_lines(p)
    res, k = {}, 0
    for t in C.TASKS:
        n = len(C.read_lines(os.path.join(root, t, "train.jsonl")))
        res[t] = rows[k:k + n]
        k += n
    assert k == len(rows), "prepare script skipped rows; alignment would be wrong"
    for t in C.TASKS:
        assert all(json.loads(r)["task_type"] == t for r in res[t])
    return res


def pick_validation(orig, banned, per_task, seed):
    """Greedy, seeded, group-wise. orig: {task: [(line_bytes,row)]}; groups never straddle tasks or arms."""
    groups = defaultdict(lambda: Counter())
    for t in C.TASKS:
        for _, r in orig[t]:
            groups[C.schema_content_hash(r["schema"])][t] += 1
    keys = sorted(k for k in groups if k not in banned)
    random.Random(seed).shuffle(keys)
    chosen, tot = [], Counter()
    for k in keys:
        if all(tot[t] + groups[k][t] <= per_task for t in C.TASKS):
            chosen.append(k)
            tot.update(groups[k])
        if all(tot[t] == per_task for t in C.TASKS):
            break
    return set(chosen), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-seeds", type=int, default=2000)
    ap.add_argument("--per-task", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base, rel = load_rows(C.BASELINE_ROOT, "train"), load_rows(C.RELEASE_ROOT, "train")
    # Test must be identical between the two sources; it is only READ.
    for t in C.TASKS:
        assert C.sha256_file(f"{C.BASELINE_ROOT}/{t}/test.jsonl") == C.sha256_file(f"{C.RELEASE_ROOT}/{t}/test.jsonl")
    test_hashes = set()
    for t in C.TASKS:
        for ln in C.read_lines(f"{C.RELEASE_ROOT}/{t}/test.jsonl"):
            test_hashes.add(C.schema_content_hash(json.loads(ln)["schema"]))

    origin = {}
    with open(f"{C.RELEASE_ROOT}/row_origin.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == "train":
                origin[(r["task"], int(r["line"]))] = r
    # original rows of the release must be exactly the baseline rows (same bytes, same relative order)
    for t in C.TASKS:
        rel_orig = [ln for i, (ln, _) in enumerate(rel[t], 1) if origin[(t, i)]["origin"] == "original"]
        assert rel_orig == [ln for ln, _ in base[t]], f"{t}: release original rows != baseline rows"
    rel_orig_idx = {t: [i for i in range(1, len(rel[t]) + 1) if origin[(t, i)]["origin"] == "original"] for t in C.TASKS}

    seed = None
    for s in range(a.seed_start, a.seed_start + a.max_seeds):
        val_hashes, tot = pick_validation(base, test_hashes, a.per_task, s)
        if all(tot[t] == a.per_task for t in C.TASKS):
            seed = s
            break
    if seed is None:
        raise SystemExit("no seed produced exact per-task validation counts")

    prep = {"A": render(C.BASELINE_ROOT), "B": render(C.RELEASE_ROOT)}
    sch = {"A": {t: [C.schema_content_hash(r["schema"]) for _, r in base[t]] for t in C.TASKS},
           "B": {t: [C.schema_content_hash(r["schema"]) for _, r in rel[t]] for t in C.TASKS}}
    src = {"A": base, "B": rel}

    # validation: original rows (baseline line numbers) whose schema content is in val_hashes; rendered from arm A's converter output
    val_rows, val_manifest = [], []
    for t in C.TASKS:
        for i, h in enumerate(sch["A"][t], 1):
            if h in val_hashes:
                val_rows.append(prep["A"][t][i - 1])
                val_manifest.append(("val", t, i, rel_orig_idx[t][i - 1], "original", h, C.sha256_bytes(base[t][i - 1][0])))
    assert Counter(m[1] for m in val_manifest) == {t: a.per_task for t in C.TASKS}

    summary = {"validation_seed": seed, "validation_schema_contents": len(val_hashes), "test_schema_contents": len(test_hashes)}
    outputs = {}
    for arm in ("A", "B"):
        rows, man, removed = [], [], Counter()
        for t in C.TASKS:
            for i, h in enumerate(sch[arm][t], 1):
                if h in val_hashes:
                    removed[(t, "val_group")] += 1
                    continue
                if h in test_hashes:
                    removed[(t, "test_leak")] += 1
                    continue
                if arm == "A":
                    bl, rl, org = i, rel_orig_idx[t][i - 1], "original"
                else:
                    org = origin[(t, i)]["origin"]
                    rl = i
                    bl = rel_orig_idx[t].index(i) + 1 if org == "original" else ""
                rows.append(prep[arm][t][i - 1])
                man.append(("train", t, i if arm == "B" else bl, rl, org, h, C.sha256_bytes(src[arm][t][i - 1][0])))
        outputs[arm] = (rows, man)
        summary[f"arm_{arm}_train_rows"] = len(rows)
        summary[f"arm_{arm}_train_by_task"] = dict(Counter(m[1] for m in man))
        summary[f"arm_{arm}_removed"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(removed.items())}
        summary[f"arm_{arm}_origin"] = dict(Counter(m[4] for m in man))

    # exact-subset proof: every retained A row (task, release line) is retained in B
    a_keys = {(m[1], m[3]) for m in outputs["A"][1]}
    b_keys = {(m[1], m[3]) for m in outputs["B"][1] if m[4] == "original"}
    assert a_keys == b_keys, "retained original rows of A are not exactly the retained original rows of B"
    a_bytes = Counter(r for r in outputs["A"][0])
    b_bytes = Counter(r for r in outputs["B"][0])
    summary["A_subset_of_B_by_row_identity"] = True
    summary["A_subset_of_B_by_rendered_bytes"] = all(b_bytes[k] >= v for k, v in a_bytes.items())
    # no validation schema content in either training arm; no test schema content either
    for arm in ("A", "B"):
        hs = {m[5] for m in outputs[arm][1]}
        assert not (hs & val_hashes) and not (hs & test_hashes)
    summary["no_val_or_test_schema_in_train"] = True
    # validation and train disjoint as rows
    summary["val_rows"] = len(val_rows)
    summary["val_by_task"] = {t: a.per_task for t in C.TASKS}

    files = {"validation.jsonl": (val_rows, val_manifest), "train_A_3k.jsonl": outputs["A"], "train_B_7k.jsonl": outputs["B"]}
    checks = {}
    for name, (rows, man) in files.items():
        C.write_lines(os.path.join(a.out, name), rows)
        with open(os.path.join(a.out, name.replace(".jsonl", "_manifest.tsv")), "x", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["split", "task", "src_line_in_arm_file", "release_line", "origin", "schema_content_sha256", "source_line_sha256"])
            w.writerows(man)
    with open(os.path.join(a.out, "validation_schema_hashes.txt"), "x") as f:
        f.write("\n".join(sorted(val_hashes)) + "\n")
    with open(os.path.join(a.out, "test_leak_removed_rows.tsv"), "x") as f:
        f.write("arm\ttask\tline\n")
        for arm in ("A", "B"):
            for t in C.TASKS:
                for i, h in enumerate(sch[arm][t], 1):
                    if h in test_hashes and h not in val_hashes:
                        f.write(f"{arm}\t{t}\t{i}\n")
    for fn in sorted(os.listdir(a.out)):
        checks[fn] = C.sha256_file(os.path.join(a.out, fn))
    summary["inputs_sha256"] = {f"{r}/{t}/{s}.jsonl": C.sha256_file(f"{root}/{t}/{s}.jsonl")
                                for r, root in (("baseline", C.BASELINE_ROOT), ("release", C.RELEASE_ROOT)) for t in C.TASKS for s in ("train", "test")}
    summary["inputs_sha256"]["release/row_origin.tsv"] = C.sha256_file(f"{C.RELEASE_ROOT}/row_origin.tsv")
    summary["original_training_sha256"] = {n: C.sha256_file(os.path.join(C.REPO, "original_training", "scripts", n))
                                           for n in ("train_lora.py", "prepare_benchson_data.py")}
    summary["outputs_sha256"] = checks
    with open(os.path.join(a.out, "split_summary.json"), "x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
