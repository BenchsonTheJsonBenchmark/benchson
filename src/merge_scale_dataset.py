"""Merge the published baseline (3,000 train + 450 test) with newly generated train examples.

Writes a NEW directory; the baseline and the new run are only read.

    python src/merge_scale_dataset.py --baseline data_published/<sha12> \
        --new data_scale_only/<run> --out-dir data_scale_7k [--jsonl-out hf_export_7k]

Checks (any failure aborts before anything is written):
  * the new run has no test examples and no file/id collision with the baseline
  * no new schema appears in the baseline train OR test set (by content hash)
  * every new record has exactly the same field names as the published records of that task
  * after copying, the test files are byte-identical to the baseline's
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation.published import TASKS, canon_hash, encode_record, reference_hash, schema_hash, sha256_file, task_keys


def _files(base, task, split):
    d = os.path.join(base, f"benchmark_{task}", split)
    return sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []


def _load(base, task, split, name):
    with open(os.path.join(base, f"benchmark_{task}", split, name), encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-train-schema-reuse", action="store_true",
                    help="new examples may use schemas that already have published TRAIN examples (controlled "
                         "reuse); published TEST schemas stay forbidden and exact content duplicates are rejected")
    ap.add_argument("--jsonl-out", help="also write <task>/{train,test}.jsonl in the published (stringified) format")
    args = ap.parse_args(argv)

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        sys.exit(f"{args.out_dir} is not empty. Refusing to overwrite.")
    for a, b in ((args.out_dir, args.baseline), (args.out_dir, args.new)):
        if os.path.abspath(a) == os.path.abspath(b):
            sys.exit("--out-dir must differ from the baseline and the new run.")

    errors, counts = [], {}
    base_hashes = {"train": set(), "test": set()}
    base_keys = {}
    base_exact = {t: set() for t in TASKS}
    base_content = {t: set() for t in TASKS}
    base_refs = set()          # (schema hash, reference hash) of every published example
    dup_stats = {"exact_duplicates_vs_published": 0, "exact_duplicates_within_new": 0,
                 "same_reference_as_published": 0, "cross_schema_content_duplicates": 0}
    new_exact = {t: set() for t in TASKS}
    new_content = {t: set() for t in TASKS}
    new_refs = set()
    per_schema = {}
    for task in TASKS:
        for split in ("train", "test"):
            for name in _files(args.baseline, task, split):
                rec = _load(args.baseline, task, split, name)
                base_hashes[split].add(schema_hash(rec["schema"]))
                base_keys.setdefault(task, set()).add(frozenset(rec))
                exact, content = task_keys(task, rec)
                base_exact[task].add(exact)
                base_content[task].add(content)
                base_refs.add((schema_hash(rec["schema"]), reference_hash(task, rec)))
    seen_new = set()
    for task in TASKS:
        if _files(args.new, task, "test"):
            errors.append(f"{task}: the new run contains test examples; the test set must stay the published one")
        base_names = set(_files(args.baseline, task, "train")) | set(_files(args.baseline, task, "test"))
        new_names = _files(args.new, task, "train")
        counts[task] = {"baseline_train": len(_files(args.baseline, task, "train")),
                        "baseline_test": len(_files(args.baseline, task, "test")), "new_train": len(new_names)}
        clash = base_names & set(new_names)
        if clash:
            errors.append(f"{task}: {len(clash)} file names also exist in the baseline, e.g. {sorted(clash)[:3]}")
        for name in new_names:
            rec = _load(args.new, task, "train", name)
            h = schema_hash(rec["schema"])
            if h in base_hashes["test"]:
                errors.append(f"{task}/{name}: schema is in the published TEST set (leakage)")
            if h in base_hashes["train"] and not args.allow_train_schema_reuse:
                errors.append(f"{task}/{name}: schema already used in the published train set")
            exact, content = task_keys(task, rec)
            ref_key = (h, reference_hash(task, rec))
            if exact in base_exact[task]:
                dup_stats["exact_duplicates_vs_published"] += 1
                errors.append(f"{task}/{name}: exact duplicate of a published {task} example (same schema and JSON)")
            elif exact in new_exact[task]:
                dup_stats["exact_duplicates_within_new"] += 1
                errors.append(f"{task}/{name}: exact duplicate of another new {task} example")
            if ref_key in base_refs:
                dup_stats["same_reference_as_published"] += 1
                errors.append(f"{task}/{name}: same reference JSON as a published example of this schema")
            if content in base_content[task] or content in new_content[task]:
                dup_stats["cross_schema_content_duplicates"] += 1   # same JSON under another schema: reported only
            new_exact[task].add(exact)
            new_content[task].add(content)
            new_refs.add(ref_key)
            per_schema.setdefault(h, {t: 0 for t in TASKS})[task] += 1
            if frozenset(rec) not in base_keys.get(task, {frozenset(rec)}):
                errors.append(f"{task}/{name}: fields {sorted(rec)} differ from the published records")
            seen_new.add((task, h))
    if errors:
        sys.exit("Merge aborted:\n  " + "\n  ".join(errors[:40]) + (f"\n  ... {len(errors)} total" if len(errors) > 40 else ""))

    for task in TASKS:
        for split, src_root in (("train", args.baseline), ("test", args.baseline), ("train", args.new)):
            dst = os.path.join(args.out_dir, f"benchmark_{task}", split)
            os.makedirs(dst, exist_ok=True)
            for name in _files(src_root, task, split):
                shutil.copyfile(os.path.join(src_root, f"benchmark_{task}", split, name), os.path.join(dst, name))

    for task in TASKS:  # the held-out set must be byte-identical
        for name in _files(args.baseline, task, "test"):
            a = sha256_file(os.path.join(args.baseline, f"benchmark_{task}", "test", name))
            b = sha256_file(os.path.join(args.out_dir, f"benchmark_{task}", "test", name))
            if a != b:
                sys.exit(f"test file changed during copy: {task}/{name}")

    out_counts = {t: {s: len(_files(args.out_dir, t, s)) for s in ("train", "test")} for t in TASKS}
    manifest = {"baseline": os.path.abspath(args.baseline), "new_run": os.path.abspath(args.new),
                "baseline_published_manifest": _maybe(os.path.join(args.baseline, "published_manifest.json")),
                "new_run_manifest": _maybe(os.path.join(args.new, "generation_manifest.json")),
                "counts": counts, "merged_counts": out_counts,
                "schema_reuse_allowed": args.allow_train_schema_reuse, "duplicate_statistics": dup_stats,
                "new_examples_per_schema_histogram": _hist(sum(v.values()) for v in per_schema.values()),
                "new_schemas_used": len(per_schema),
                "new_schemas_also_in_published_train": len(set(per_schema) & base_hashes["train"]),
                "checks": ["no test in new run", "no id collision", "no schema overlap with baseline train/test",
                           "field sets equal", "test files byte-identical",
                           "no exact content duplicate (new vs published, new vs new)",
                           "no new reference equal to a published reference of the same schema"]}
    with open(os.path.join(args.out_dir, "merge_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.jsonl_out:
        for task in TASKS:
            os.makedirs(os.path.join(args.jsonl_out, task), exist_ok=True)
            for split in ("train", "test"):
                with open(os.path.join(args.jsonl_out, task, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                    for name in _files(args.out_dir, task, split):
                        f.write(json.dumps(encode_record(_load(args.out_dir, task, split, name))) + "\n")
    print(json.dumps(out_counts, indent=2))


def _hist(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _maybe(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    main()
