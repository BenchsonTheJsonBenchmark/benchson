"""Import the published Benchson dataset (HF `aviv1ron1/Benchson`) as a read-only baseline.

Nothing is generated and no model is called. The result is a separate directory:

    <out-dir>/benchmark_<task>/{train,test}/<id>.json     (read-only files)
    <out-dir>/reserved_schemas.json                        (ids + schema hashes, for exclusions)
    <out-dir>/published_manifest.json                      (revision, counts, file sha256)

Usage:
    python src/import_published_benchmark.py --revision <40-hex sha> --out-dir data_published/<sha12>
    python src/import_published_benchmark.py --revision <sha> --source-dir /path/to/local/snapshot ...
"""
import argparse
import hashlib
import json
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_generation.published import (KNOWN_REVISIONS, reference_hash, task_keys, git_blob_oid, TASKS, JSON_FIELDS, REQUIRED_FIELDS, decode_row, encode_record,
                                       is_pinned_revision, parse_card_license, record_id, schema_hash, sha256_file)

EXPECTED = {"train": 1000, "test": 150}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _locate(root, task, split):
    for cand in (os.path.join(root, task, f"{split}.jsonl"), os.path.join(root, f"{task}_{split}.jsonl")):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"no {split}.jsonl for task '{task}' under {root}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="aviv1ron1/Benchson")
    ap.add_argument("--revision", required=True, help="full 40-hex commit SHA (branches/tags are rejected)")
    ap.add_argument("--source-dir", help="already-downloaded snapshot (offline); otherwise downloaded from the Hub")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", help="Hugging Face cache for the download (use a temporary folder so it can be deleted afterwards)")
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args(argv)

    if not is_pinned_revision(args.revision):
        sys.exit("--revision must be a full 40-character commit SHA, not a branch or tag name.")
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        sys.exit(f"{args.out_dir} is not empty. Refusing to overwrite; choose a new directory.")

    src = args.source_dir
    if not src:
        from huggingface_hub import snapshot_download
        # token=False: this is a public read, so no credential is ever sent to the original repo.
        # Only the card and the three task folders are fetched (the tree also holds a stray .DS_Store).
        src = snapshot_download(repo_id=args.repo, repo_type="dataset", revision=args.revision, token=False,
                                cache_dir=args.cache_dir, allow_patterns=["README.md", "create/*", "fix/*", "modify/*"])

    rows, per_file, originals, oid_checks = {}, {}, {}, {}
    known = KNOWN_REVISIONS.get(args.revision, {}).get("files", {})
    for task in TASKS:
        for split in ("train", "test"):
            path = _locate(src, task, split)
            rows[(task, split)] = _read_jsonl(path)
            rel = os.path.relpath(path, src)
            per_file[rel] = sha256_file(path)
            originals[rel] = path
            oid = git_blob_oid(path)
            if rel in known:
                ok = (oid == known[rel][0] and os.path.getsize(path) == known[rel][1])
                oid_checks[rel] = "match" if ok else "MISMATCH"
                if not ok:
                    sys.exit(f"{rel}: git blob id {oid} differs from the id Hugging Face reports for revision "
                             f"{args.revision[:12]} ({known[rel][0]}). This is not the pinned snapshot.")
            else:
                oid_checks[rel] = f"unverified (revision not in KNOWN_REVISIONS); blob id {oid}"

    card_info = None
    card_path = os.path.join(src, "README.md")
    if os.path.isfile(card_path):
        text = open(card_path, encoding="utf-8").read()
        known_card = KNOWN_REVISIONS.get(args.revision, {}).get("other_files", {}).get("README.md")
        card_oid = git_blob_oid(card_path)
        if known_card and (card_oid != known_card[0] or os.path.getsize(card_path) != known_card[1]):
            sys.exit(f"README.md: git blob id {card_oid} differs from the id Hugging Face reports for revision "
                     f"{args.revision[:12]} ({known_card[0]}). This is not the pinned snapshot.")
        card_info = {"sha256": sha256_file(card_path), "git_blob_oid": card_oid, "bytes": os.path.getsize(card_path),
                     "license": parse_card_license(text),
                     "git_blob_oid_check": "match" if known_card else "unverified (revision not in KNOWN_REVISIONS)"}

    problems = []
    for (task, split), items in rows.items():
        if len(items) != EXPECTED[split]:
            problems.append(f"{task}/{split}: {len(items)} rows, expected {EXPECTED[split]}")
    if problems and not args.allow_count_mismatch:
        sys.exit("Count check failed:\n  " + "\n  ".join(problems))

    records, ids_seen, bad = {}, {}, []
    for (task, split), items in rows.items():
        for i, row in enumerate(items):
            try:
                rec = decode_row(task, row)
            except Exception as e:
                bad.append(f"{task}/{split}[{i}]: decode failed: {e}")
                continue
            missing = [k for k in REQUIRED_FIELDS[task] if k not in rec]
            if missing:
                bad.append(f"{task}/{split}[{i}]: missing {missing}")
            if encode_record(rec) != row:
                bad.append(f"{task}/{split}[{i}]: round-trip differs (data would not be preserved)")
            rid = record_id(rec)
            key = (task, split, rid)
            if key in ids_seen:
                bad.append(f"{task}/{split}: duplicate id {rid}")
            ids_seen[key] = True
            records[key] = rec
    if bad:
        sys.exit("Validation failed:\n  " + "\n  ".join(bad[:30]) + (f"\n  ... {len(bad)} total" if len(bad) > 30 else ""))

    os.makedirs(args.out_dir, exist_ok=True)
    for (task, split, rid), rec in records.items():
        d = os.path.join(args.out_dir, f"benchmark_{task}", split)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{rid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    for rel, path in originals.items():  # byte-identical copies, used later to republish the baseline unchanged
        dst = os.path.join(args.out_dir, "original_files", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(path, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    if card_info:  # verbatim copy of the original dataset card, republished as ORIGINAL_DATASET_CARD.md
        dst = os.path.join(args.out_dir, "original_card", "README.md")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(card_path, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    reserved = {"dataset": args.repo, "revision": args.revision,
                "ids": {"train": [], "test": []}, "schema_hashes": {"train": [], "test": []},
                "ids_by_task": {t: {"train": [], "test": []} for t in TASKS}}
    for (task, split, rid), rec in records.items():
        reserved["ids_by_task"][task][split].append(rid)
        reserved["ids"][split].append(rid)
        reserved["schema_hashes"][split].append(schema_hash(rec["schema"]))
    for split in ("train", "test"):
        reserved["ids"][split] = sorted(set(reserved["ids"][split]))
        reserved["schema_hashes"][split] = sorted(set(reserved["schema_hashes"][split]))
    for t in TASKS:
        for sp in ("train", "test"):
            reserved["ids_by_task"][t][sp] = sorted(reserved["ids_by_task"][t][sp])
    # Content fingerprints, so a later run can reject a new example that repeats a published one.
    refs_by_schema, keys = {}, {t: {"exact": set(), "content": set()} for t in TASKS}
    for (task, split, rid), rec in records.items():
        refs_by_schema.setdefault(schema_hash(rec["schema"]), set()).add(reference_hash(task, rec))
        exact, content = task_keys(task, rec)
        keys[task]["exact"].add(exact)
        keys[task]["content"].add(content)
    reserved["reference_hashes_by_schema"] = {k: sorted(v) for k, v in sorted(refs_by_schema.items())}
    reserved["task_keys"] = {t: {k: sorted(v) for k, v in d.items()} for t, d in keys.items()}
    overlap = sorted(set(reserved["schema_hashes"]["train"]) & set(reserved["schema_hashes"]["test"]))

    with open(os.path.join(args.out_dir, "reserved_schemas.json"), "w") as f:
        json.dump(reserved, f, indent=1)
    manifest = {"dataset": args.repo, "revision": args.revision,
                "counts": {f"{t}/{s}": len(rows[(t, s)]) for t in TASKS for s in ("train", "test")},
                "source_file_sha256": per_file, "git_blob_oid_check": oid_checks, "original_card": card_info,
                "known_revision": KNOWN_REVISIONS.get(args.revision, {}).get("branch"),
                "unique_schemas": {s: len(reserved["schema_hashes"][s]) for s in ("train", "test")},
                "train_test_schema_overlap": len(overlap),
                "count_warnings": problems}
    with open(os.path.join(args.out_dir, "published_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if overlap:
        print(f"WARNING: {len(overlap)} schemas appear in both published train and test.")


if __name__ == "__main__":
    main()
