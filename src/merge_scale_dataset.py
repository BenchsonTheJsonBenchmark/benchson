"""Merge the published baseline (3,000 train + 450 test) with the newly generated train examples (4,000).

Writes a NEW directory; the baseline and the new run are only read.

    python src/merge_scale_dataset.py --baseline data_published/<sha12> \
        --new data_scale_only/<merged-chunks-dir> --out-dir data_scale_7k --allow-train-schema-reuse \
        [--jsonl-out hf_export_7k] [--chunks-root data_scale_only/full/chunks --anchor-file ... --reserved-file ...]

Everything is checked BEFORE anything is written (any failure aborts); the result is built in a private staging directory
next to --out-dir, verified file by file, and only then renamed to --out-dir. A failed run leaves no --out-dir.

Baseline (never trusted by name):
  * published_manifest.json names the pinned revision, the expected counts (1000 train / 150 test per task)
  * original_files/<task>/<split>.jsonl match the manifest SHA256 and the Hugging Face git blob ids
  * every benchmark_<task>/<split>/*.json equals one published JSONL row (nothing missing, nothing extra)
New dataset:
  * merged_chunks_manifest.json is well formed, lists the expected chain (c01..c08) and totals
  * the files on disk are exactly the listed files, with the recorded SHA256, in the expected quotas, no test split
  * with --chunks-root the whole chain is re-validated by chunked_run (otherwise only the manifest-level chain is checked)
Examples:
  * no new schema in the published test set; published train schemas only with --allow-train-schema-reuse
  * exact duplicates, same-reference duplicates, field sets, file-name collisions (also case-insensitive)
  * optional --max-new-references-per-schema (the cap the chunks were generated with)
Result: 7,000 train + 450 test; every baseline file and every new file is byte-identical to its source (SHA256).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generation.published import (KNOWN_REVISIONS, REQUIRED_FIELDS, TASKS, decode_row, encode_record, git_blob_oid,
                                       is_pinned_revision, record_id, reference_hash, schema_hash, sha256_file, task_keys)

PINNED_REVISION = "7962b0947f7d96eb3a0e6b9b928d939c7f3061a6"
EXPECTED_NEW = {"create": 1334, "fix": 1333, "modify": 1333}
EXPECTED_BASELINE = {"train": 1000, "test": 150}
EXPECTED_CHUNKS = 8
MERGED_FORMAT = "benchson-merged-chunks/1"
MANIFEST_FORMAT = "benchson-scale-merge/2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_DIR = re.compile(r"^c(\d{2})_.+-r\d+$")


class MergeAbort(Exception):
    pass


# ------------------------------------------------------------------ small helpers

def _listing(base, task, split):
    """All entries of a split directory (not only *.json: a stray file must be noticed, never ignored)."""
    d = os.path.join(base, f"benchmark_{task}", split)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _files(base, task, split):
    return [f for f in _listing(base, task, split) if f.endswith(".json")]


def _path(base, task, split, name):
    return os.path.join(base, f"benchmark_{task}", split, name)


def _load(base, task, split, name):
    with open(_path(base, task, split, name), encoding="utf-8") as f:
        return json.load(f)


def _read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise MergeAbort(f"{what} not found: {path}")
    except (OSError, ValueError) as e:
        raise MergeAbort(f"{what} is unreadable or not valid JSON ({path}): {e}")


def _parse_counts(text):
    out = {}
    for part in text.split(","):
        task, _, n = part.partition("=")
        if task.strip() not in TASKS or not n.strip().isdigit():
            raise argparse.ArgumentTypeError(f"expected create=N,fix=N,modify=N, got {text!r}")
        out[task.strip()] = int(n)
    if set(out) != set(TASKS):
        raise argparse.ArgumentTypeError(f"expected all of {TASKS}, got {sorted(out)}")
    return out


def _ancestors(path):
    path = os.path.realpath(path)          # resolves symlinks in every existing component
    while True:
        yield path
        up = os.path.dirname(path)
        if up == path:
            return
        path = up


def _overlap(a, b):
    """True when a and b are the same place or one contains the other. Symlinks are resolved, and existing parts are compared
    with samefile, so a different spelling (letter case on a case-insensitive disk, bind/hard-linked directory) cannot slip by."""
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    fa, fb = ra.casefold(), rb.casefold()      # conservative: also refuse paths that only differ in letter case
    if fa == fb or fa.startswith(fb.rstrip(os.sep) + os.sep) or fb.startswith(fa.rstrip(os.sep) + os.sep):
        return True
    for x, y in ((a, b), (b, a)):
        if os.path.exists(y):
            for anc in _ancestors(x):
                if os.path.exists(anc) and os.path.samefile(anc, y):
                    return True
    return False


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


# ------------------------------------------------------------------ baseline integrity

def verify_baseline(base, expected_revision, expected):
    """Checks that `base` really is the pinned published dataset. Returns (provenance, {task: {split: {name: record}}})."""
    errors = []
    manifest_path = os.path.join(base, "published_manifest.json")
    manifest = _read_json(manifest_path, "baseline published_manifest.json")
    rev = manifest.get("revision")
    if not is_pinned_revision(rev):
        errors.append(f"published_manifest.json revision {rev!r} is not a full commit hash")
    elif rev != expected_revision:
        errors.append(f"baseline revision {rev} is not the pinned revision {expected_revision}")
    mcounts = manifest.get("counts") or {}
    for task in TASKS:
        for split in ("train", "test"):
            if mcounts.get(f"{task}/{split}") != expected[split]:
                errors.append(f"published_manifest.json counts[{task}/{split}] = {mcounts.get(f'{task}/{split}')}, expected {expected[split]}")
    known = KNOWN_REVISIONS.get(rev, {}).get("files", {}) if isinstance(rev, str) else {}
    sha_map = manifest.get("source_file_sha256") or {}
    records, rows_sha, oid_results = {}, {}, {}
    for task in TASKS:
        records[task] = {"train": {}, "test": {}}
        for split in ("train", "test"):
            rel = f"{task}/{split}.jsonl"
            path = os.path.join(base, "original_files", task, f"{split}.jsonl")
            if not os.path.isfile(path):
                errors.append(f"original_files/{rel} is missing: the baseline cannot be verified against the published rows")
                continue
            digest = sha256_file(path)
            rows_sha[rel] = digest
            if sha_map.get(rel) != digest:
                errors.append(f"original_files/{rel}: SHA256 {digest[:12]} differs from published_manifest.json ({str(sha_map.get(rel))[:12]})")
            if rel in known:
                ok = git_blob_oid(path) == known[rel][0] and os.path.getsize(path) == known[rel][1]
                oid_results[rel] = "match" if ok else "MISMATCH"
                if not ok:
                    errors.append(f"original_files/{rel}: git blob id/size differs from Hugging Face revision {rev[:12]}")
            else:
                oid_results[rel] = "unverified (revision not in KNOWN_REVISIONS)"
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if len(rows) != expected[split]:
                errors.append(f"original_files/{rel}: {len(rows)} rows, expected {expected[split]}")
            want = {}
            for row in rows:
                rec = decode_row(task, row)
                want[record_id(rec) + ".json"] = (rec, row)
            have = _listing(base, task, split)
            if len(have) != expected[split]:
                errors.append(f"benchmark_{task}/{split}: {len(have)} entries, expected {expected[split]}")
            if set(have) != set(want):
                errors.append(f"benchmark_{task}/{split}: files differ from the published rows "
                              f"(missing {sorted(set(want) - set(have))[:3]}, unexpected {sorted(set(have) - set(want))[:3]})")
            for name in sorted(set(have) & set(want)):
                try:
                    rec = _load(base, task, split, name)
                except (OSError, ValueError) as e:
                    errors.append(f"benchmark_{task}/{split}/{name}: unreadable: {e}")
                    continue
                if rec != want[name][0] or encode_record(rec) != want[name][1]:
                    errors.append(f"benchmark_{task}/{split}/{name}: content differs from the published JSONL row")
                records[task][split][name] = rec
    if errors:
        raise MergeAbort("Baseline integrity check failed:\n  " + "\n  ".join(errors[:30]) + (f"\n  ... {len(errors)} total" if len(errors) > 30 else ""))
    prov = {"revision": rev, "directory_name": os.path.basename(os.path.abspath(base)),
            "published_manifest_sha256": sha256_file(manifest_path), "source_file_sha256": rows_sha,
            "git_blob_oid_check": oid_results,
            "verified": "published_manifest.json revision and counts, original_files SHA256 and git blob ids, "
                        "every benchmark_*/{train,test} file equals its published JSONL row"}
    return prov, records


# ------------------------------------------------------------------ new dataset integrity

def verify_new(new, expected_new, expected_chunks, errors):
    """Checks the merged-chunks directory against its manifest. Returns (provenance, manifest, {task: [names]})."""
    path = os.path.join(new, "merged_chunks_manifest.json")
    manifest = _read_json(path, "merged_chunks_manifest.json")
    if not isinstance(manifest, dict):
        raise MergeAbort("merged_chunks_manifest.json is not a JSON object")
    if manifest.get("format") != MERGED_FORMAT:
        errors.append(f"merged_chunks_manifest.json format {manifest.get('format')!r}, expected {MERGED_FORMAT!r}")
    chain = manifest.get("chain")
    if not isinstance(chain, list) or len(chain) != expected_chunks:
        errors.append(f"merged_chunks_manifest.json lists {len(chain) if isinstance(chain, list) else 'no'} chunks, expected {expected_chunks}")
        chain = chain if isinstance(chain, list) else []
    for i, link in enumerate(chain, 1):
        m = _CHUNK_DIR.match(link.get("dir", "")) if isinstance(link, dict) else None
        if not m or int(m.group(1)) != i:
            errors.append(f"chain[{i - 1}] is not chunk {i:02d} ({link.get('dir') if isinstance(link, dict) else link!r}): wrong order or name")
        for key in ("state_sha256", "success_marker_sha256"):
            if not isinstance(link, dict) or not _HEX64.match(str(link.get(key, ""))):
                errors.append(f"chain[{i - 1}].{key} is missing or not a SHA256")
    if manifest.get("totals") != expected_new:
        errors.append(f"merged_chunks_manifest.json totals {manifest.get('totals')} differ from the expected {expected_new}")
    listed = manifest.get("files")
    if not isinstance(listed, dict) or set(listed) != set(TASKS):
        errors.append("merged_chunks_manifest.json has no per-task file list")
        listed = {t: {} for t in TASKS}
    names = {}
    allowed_top = {"merged_chunks_manifest.json"} | {f"benchmark_{t}" for t in TASKS}
    for extra in sorted(set(os.listdir(new)) - allowed_top):
        errors.append(f"unexpected entry in the new dataset directory: {extra}")
    for task in TASKS:
        if _listing(new, task, "test"):
            errors.append(f"{task}: the new run contains test examples; the test set must stay the published one")
        files = listed[task] if isinstance(listed[task], dict) else {}
        on_disk = _listing(new, task, "train")
        names[task] = [n for n in on_disk if n.endswith(".json")]
        if len(on_disk) != expected_new[task] or len(files) != expected_new[task]:
            errors.append(f"{task}: {len(on_disk)} files on disk and {len(files)} listed in the manifest, expected {expected_new[task]}")
        if set(on_disk) != set(files):
            errors.append(f"{task}: files on disk differ from the manifest (missing {sorted(set(files) - set(on_disk))[:3]}, "
                          f"unexpected {sorted(set(on_disk) - set(files))[:3]})")
        for name in sorted(set(on_disk) & set(files)):
            if not _HEX64.match(str(files[name])):
                errors.append(f"{task}/{name}: manifest hash is not a SHA256")
            elif sha256_file(_path(new, task, "train", name)) != files[name]:
                errors.append(f"{task}/{name}: SHA256 differs from the hash recorded in merged_chunks_manifest.json")
    prov = {"merged_chunks_manifest_sha256": sha256_file(path), "directory_name": os.path.basename(os.path.abspath(new)),
            "chain": chain, "totals": manifest.get("totals"),
            "file_sha256_verified": "every new file equals the SHA256 recorded when the chain was merged",
            "chain_verification": "manifest-level only: chain shape, order, recorded hashes and file hashes. The chunk state and "
                                  "success-marker files were NOT re-read (no --chunks-root)"}
    return prov, manifest, names


def verify_chain_dirs(manifest, root, anchor_file, reserved_file, errors):
    """Optional: re-validates the real chunk chain with the existing chunked_run machinery and compares it with the manifest."""
    import chunked_run
    from data_generation import chunking
    chain = manifest.get("chain") or []
    if not chain:
        return None
    dirs = [os.path.join(os.path.abspath(root), link["dir"]) for link in chain]
    res = chunked_run.validate_chunk(dirs[-1], anchor_file, reserved_file)
    if res["code"] != 0:
        errors.append(f"chain does not validate in {root} ({res['verdict']}): " + "; ".join((res["technical"] + res["invalid"] + res["shortfall"])[:5]))
        return None
    real = chunking.load_chain(dirs[-1])
    if [os.path.basename(d) for d, _ in real] != [link["dir"] for link in chain]:
        errors.append("the chunk directories do not form the chain named in merged_chunks_manifest.json")
        return None
    for (cdir, state), link in zip(real, chain):
        if state["self_sha256"] != link["state_sha256"]:
            errors.append(f"{link['dir']}: chunk_state.json hash differs from the merged manifest")
        if chunking.load_seal(cdir, state)["self_sha256"] != link["success_marker_sha256"]:
            errors.append(f"{link['dir']}: chunk_success.json hash differs from the merged manifest")
    for task in TASKS:
        mine = {n: h for _, state in real for n, h in state["files"][task].items()}
        if mine != (manifest.get("files") or {}).get(task):
            errors.append(f"{task}: files recorded in the chunk states differ from merged_chunks_manifest.json")
    return {"validated": True, "chunks": [link["dir"] for link in chain], "verdict": res["verdict"], "cumulative": res.get("cumulative")}


# ------------------------------------------------------------------ copy / verify / publish

def _copy_exclusive(src, dst):
    """Copies bytes; refuses to touch an existing file (a name collision must never overwrite anything)."""
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        shutil.copyfileobj(fin, fout)
        fout.flush()
        os.fsync(fout.fileno())


def _remove_own_staging(path, created, protected):
    """Deletes ONLY a staging directory this run created (same real path as when it was made, a plain directory, never a
    symlink) and never anything that is, contains or lies inside a source directory."""
    if not path or created.get(path) != os.path.realpath(path) or os.path.islink(path) or not os.path.isdir(path):
        return
    if not os.path.basename(path).startswith(".merge_staging_") or any(_overlap(path, p) for p in protected):
        return
    shutil.rmtree(path, ignore_errors=True)


def _quarantine(path):
    """Moves an export that was published but whose merge directory could not be, out of the way (rename only)."""
    for n in range(1000):
        target = f"{path}.INCOMPLETE-{n}"
        if not os.path.lexists(target):
            try:
                os.rename(path, target)
                print(f"WARNING: the merge failed after the JSONL export was published; the export was renamed to {target} "
                      "and is NOT a valid result.", file=sys.stderr)
            except OSError as e:
                print(f"ERROR: could not rename the orphan export {path} ({e}). It is NOT a valid result: the merge failed.", file=sys.stderr)
            return


def _publish(staging, final):
    if os.path.lexists(final):
        raise MergeAbort(f"{final} appeared while merging. Refusing to overwrite it.")
    os.rename(staging, final)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-train-schema-reuse", action="store_true",
                    help="new examples may use schemas that already have published TRAIN examples (controlled "
                         "reuse); published TEST schemas stay forbidden and exact content duplicates are rejected")
    ap.add_argument("--jsonl-out", help="also write <task>/{train,test}.jsonl in the published (stringified) format")
    ap.add_argument("--expected-revision", default=PINNED_REVISION, help="pinned published revision the baseline must be")
    ap.add_argument("--expected-new", type=_parse_counts, default=EXPECTED_NEW, metavar="create=N,fix=N,modify=N")
    ap.add_argument("--expected-baseline-train", type=int, default=EXPECTED_BASELINE["train"])
    ap.add_argument("--expected-baseline-test", type=int, default=EXPECTED_BASELINE["test"])
    ap.add_argument("--expected-chunks", type=int, default=EXPECTED_CHUNKS)
    ap.add_argument("--max-new-references-per-schema", type=int, default=None,
                    help="when given, fail if a schema has more new references (attempts) than this, or more new examples "
                         "per task (the cap the chunks were generated with; it is not recorded in the merged dataset)")
    ap.add_argument("--chunks-root", help="directory with the sealed chunk directories: re-validates the whole chain")
    ap.add_argument("--anchor-file")
    ap.add_argument("--reserved-file")
    args = ap.parse_args(argv)
    try:
        run(args)
    except MergeAbort as e:
        sys.exit(f"Merge aborted:\n  {e}")


def run(args):
    out_dir = os.path.abspath(args.out_dir)
    jsonl_out = os.path.abspath(args.jsonl_out) if args.jsonl_out else None
    protected = [p for p in (args.baseline, args.new, args.chunks_root, args.anchor_file, args.reserved_file) if p]
    for label, p in (("--out-dir", out_dir), ("--jsonl-out", jsonl_out)):
        if p is None:
            continue
        if os.path.lexists(p):
            raise MergeAbort(f"{p} already exists. Refusing to overwrite or reuse it; choose a new {label}.")
        for src in protected:
            if _overlap(p, src):
                raise MergeAbort(f"{label} must be outside the baseline, the new run and the chunk directories (and must not contain them or be inside them).")
    if jsonl_out and _overlap(jsonl_out, out_dir):
        raise MergeAbort("--jsonl-out and --out-dir must be separate directories (neither may contain or equal the other).")
    if args.chunks_root and not (args.anchor_file and args.reserved_file):
        raise MergeAbort("--chunks-root needs --anchor-file and --reserved-file")
    expected_base = {"train": args.expected_baseline_train, "test": args.expected_baseline_test}

    errors, counts = [], {}
    # ---- A. baseline
    base_prov, base_records = verify_baseline(args.baseline, args.expected_revision, expected_base)
    # ---- B. new dataset
    new_prov, merged_manifest, new_names = verify_new(args.new, args.expected_new, args.expected_chunks, errors)
    if args.chunks_root:
        new_prov["chain_verification"] = verify_chain_dirs(merged_manifest, args.chunks_root, args.anchor_file, args.reserved_file, errors) \
            or "FAILED (see errors)"
    # ---- C/D/E. examples
    base_hashes = {"train": set(), "test": set()}
    base_keys = {}
    base_exact = {t: set() for t in TASKS}
    base_content = {t: set() for t in TASKS}
    base_refs = set()          # (schema hash, reference hash) of every published example
    dup_stats = {"exact_duplicates_vs_published": 0, "exact_duplicates_within_new": 0,
                 "same_reference_as_published": 0, "cross_schema_content_duplicates": 0}
    new_exact = {t: set() for t in TASKS}
    new_content = {t: set() for t in TASKS}
    per_schema, attempts, per_sid_task = {}, {}, {}
    for task in TASKS:
        for split in ("train", "test"):
            for rec in base_records[task][split].values():
                base_hashes[split].add(schema_hash(rec["schema"]))
                base_keys.setdefault(task, set()).add(frozenset(rec))
                exact, content = task_keys(task, rec)
                base_exact[task].add(exact)
                base_content[task].add(content)
                base_refs.add((schema_hash(rec["schema"]), reference_hash(task, rec)))
    for task in TASKS:
        base_names = set(base_records[task]["train"]) | set(base_records[task]["test"])
        names = new_names[task]
        counts[task] = {"baseline_train": len(base_records[task]["train"]), "baseline_test": len(base_records[task]["test"]),
                        "new_train": len(names)}
        clash = base_names & set(names)
        if clash:
            errors.append(f"{task}: {len(clash)} file names also exist in the baseline, e.g. {sorted(clash)[:3]}")
        lowered = {}
        for name in sorted(base_names | set(names)):
            lowered.setdefault(name.lower(), []).append(name)
        folded = [v for v in lowered.values() if len(v) > 1 and len(set(v)) > 1]
        if folded:
            errors.append(f"{task}: file names that differ only in letter case (would collide on a case-insensitive file system): {folded[:3]}")
        for name in names:
            try:
                rec = _load(args.new, task, "train", name)
                h = schema_hash(rec["schema"])
                exact, content = task_keys(task, rec)
                ref_key = (h, reference_hash(task, rec))
            except (OSError, ValueError, KeyError, TypeError) as e:
                errors.append(f"{task}/{name}: unreadable or malformed record: {e!r}")
                continue
            if h in base_hashes["test"]:
                errors.append(f"{task}/{name}: schema is in the published TEST set (leakage)")
            if h in base_hashes["train"] and not args.allow_train_schema_reuse:
                errors.append(f"{task}/{name}: schema already used in the published train set")
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
            row = per_schema.setdefault(h, {t: 0 for t in TASKS})
            row[task] += 1
            sid, _, att = name[:-5].rpartition("__a")             # the same split the chunk-chain validation uses
            if not sid or not att:
                errors.append(f"{task}/{name}: file name is not <schema id>__a<attempt>.json (cannot count it against the schema cap)")
            attempts.setdefault(sid, set()).add(att)             # distinct attempts per schema id = new references (ledger: accepted_references)
            per_sid_task[(task, sid)] = per_sid_task.get((task, sid), 0) + 1
            if not isinstance(rec, dict) or not REQUIRED_FIELDS[task] <= set(rec):
                errors.append(f"{task}/{name}: missing required fields {sorted(REQUIRED_FIELDS[task] - set(rec)) if isinstance(rec, dict) else '(not an object)'}")
            elif frozenset(rec) not in base_keys.get(task, {frozenset(rec)}):
                errors.append(f"{task}/{name}: fields {sorted(rec)} differ from the published records")
    cap = args.max_new_references_per_schema
    if cap is not None:
        for sid, atts in sorted(attempts.items()):      # same rules as chunked_run.validate_chunk: per schema id, strictly more than cap fails
            if len(atts) > cap:
                errors.append(f"schema {sid}: {len(atts)} new references across chunks, global cap is {cap}")
        for (task, sid), n in sorted(per_sid_task.items()):
            if n > cap:
                errors.append(f"{task}/{sid}: {n} new examples across chunks, global cap is {cap}")
    if errors:
        raise MergeAbort("\n  ".join(errors[:40]) + (f"\n  ... {len(errors)} total" if len(errors) > 40 else ""))

    # ---- build in a private staging directory next to the destination, verify, publish
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)
    created = {}                      # staging directories made by THIS run (the only ones that may ever be removed)
    staging = tempfile.mkdtemp(prefix=".merge_staging_", dir=parent)
    created[staging] = os.path.realpath(staging)
    jsonl_staging = None
    if jsonl_out:
        os.makedirs(os.path.dirname(jsonl_out), exist_ok=True)
        jsonl_staging = tempfile.mkdtemp(prefix=".merge_staging_", dir=os.path.dirname(jsonl_out))
        created[jsonl_staging] = os.path.realpath(jsonl_staging)
    published_jsonl = published_out = False
    try:
        out_hashes, out_counts = _build(args, staging, base_records, new_names)
        _write_manifest(args, staging, base_prov, new_prov, merged_manifest, counts, out_counts, out_hashes, dup_stats,
                        per_schema, attempts, base_hashes, cap)
        _verify_output(args, staging, base_records, new_names, out_hashes)
        jsonl_info = None
        if jsonl_out:
            jsonl_info = _write_jsonl(args, staging, jsonl_staging, base_records, out_counts)
        if jsonl_info is not None:
            with open(os.path.join(staging, "merge_manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["jsonl_export"] = jsonl_info
            with open(os.path.join(staging, "merge_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        # Order matters: the export is derived data, the merge directory (with merge_manifest.json) is the result. The export is
        # published first, so a failure of the second step can never leave a "finished" result next to a missing export.
        if jsonl_out:
            _publish(jsonl_staging, jsonl_out)
            published_jsonl = True
        _publish(staging, out_dir)
        published_out = True
    except BaseException:
        if published_jsonl and not published_out:
            _quarantine(jsonl_out)         # renamed, never deleted: an orphan export must not look like a finished one
        _remove_own_staging(staging, created, protected)
        _remove_own_staging(jsonl_staging, created, protected)
        raise
    print(json.dumps(out_counts, indent=2))


def _build(args, staging, base_records, new_names):
    out_hashes = {t: {"train": {}, "test": {}} for t in TASKS}
    for task in TASKS:
        for split, root, names in (("train", args.baseline, sorted(base_records[task]["train"])),
                                   ("test", args.baseline, sorted(base_records[task]["test"])),
                                   ("train", args.new, sorted(new_names[task]))):
            dst = os.path.join(staging, f"benchmark_{task}", split)
            os.makedirs(dst, exist_ok=True)
            for name in names:
                src = _path(root, task, split, name)
                _copy_exclusive(src, os.path.join(dst, name))
                digest = sha256_file(src)
                if sha256_file(os.path.join(dst, name)) != digest:
                    raise MergeAbort(f"copy of {task}/{split}/{name} differs from its source")
                out_hashes[task][split][name] = digest
    out_counts = {t: {s: len(out_hashes[t][s]) for s in ("train", "test")} for t in TASKS}
    return out_hashes, out_counts


def _write_manifest(args, staging, base_prov, new_prov, merged_manifest, counts, out_counts, out_hashes, dup_stats,
                    per_schema, attempts, base_hashes, cap):
    manifest = {
        "format": MANIFEST_FORMAT,
        # existing keys, kept for readers of the first format
        "baseline": os.path.abspath(args.baseline), "new_run": os.path.abspath(args.new),
        "baseline_published_manifest": _maybe(os.path.join(args.baseline, "published_manifest.json")),
        "new_run_manifest": _maybe(os.path.join(args.new, "generation_manifest.json")),
        "counts": counts, "merged_counts": out_counts,
        "schema_reuse_allowed": args.allow_train_schema_reuse, "duplicate_statistics": dup_stats,
        "new_examples_per_schema_histogram": _hist(sum(v.values()) for v in per_schema.values()),
        "new_schemas_used": len(per_schema),
        "new_schemas_also_in_published_train": len(set(per_schema) & base_hashes["train"]),
        "checks": ["baseline verified against published_manifest.json, original JSONL SHA256 and git blob ids, row by row",
                   "new dataset verified against merged_chunks_manifest.json (files, SHA256, quotas, chain shape)",
                   "no test in new run", "no id collision (also case-insensitive)",
                   "no schema overlap with baseline test; overlap with baseline train only with the reuse flag",
                   "field sets equal", "test files byte-identical", "train files byte-identical",
                   "no exact content duplicate (new vs published, new vs new)",
                   "no new reference equal to a published reference of the same schema",
                   "output verified in staging before it was published"],
        # provenance (reproducible identifiers; the paths above are only informative)
        "baseline_provenance": base_prov,
        "new_provenance": new_prov,
        "input_counts": {"baseline_train": {t: counts[t]["baseline_train"] for t in TASKS},
                         "baseline_test": {t: counts[t]["baseline_test"] for t in TASKS},
                         "new_train": {t: counts[t]["new_train"] for t in TASKS}},
        "output_counts": {"per_task": out_counts,
                          "train_total": sum(c["train"] for c in out_counts.values()),
                          "test_total": sum(c["test"] for c in out_counts.values())},
        "integrity": {"baseline_train_and_test_files_byte_identical": True, "new_files_byte_identical": True,
                      "output_file_count_matches_inputs": True, "method": "SHA256 of every source and output file"},
        "output_file_sha256": out_hashes,
        "policy": {"schema_reuse": "controlled train-schema reuse" if args.allow_train_schema_reuse else "no reuse of published train schemas",
                   "published_test_schemas": "forbidden", "exact_duplicates": "forbidden (vs published and within new)",
                   "same_reference_on_same_schema": "forbidden vs published",
                   "cross_schema_content_duplicates": "reported only",
                   "max_new_references_per_schema_checked": cap,
                   "observed_max_new_references_per_schema": max((len(a) for a in attempts.values()), default=0),
                   "cap_counting": "per schema id (file name <id>__a<attempt>): distinct attempts = references, and examples per task, as in chunked_run.validate_chunk"},
        "limitations": [
            "the per-schema reference cap and cross-chunk file-name uniqueness are only re-verified from the merged dataset when "
            "--max-new-references-per-schema is given; the generation ledger and chunk states need --chunks-root"
            if not args.chunks_root else
            "chunk chain re-validated with chunked_run (--chunks-root); the cap is checked only when --max-new-references-per-schema is given",
            "remote $ref documents used by schema validation are not part of the hashes (see chunk validation reports)",
            "_metadata sidecars of the chunks are not copied into the merged dataset"],
    }
    with open(os.path.join(staging, "merge_manifest.json"), "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _verify_output(args, staging, base_records, new_names, out_hashes):
    problems = []
    for task in TASKS:
        for split in ("train", "test"):
            expected = set(base_records[task][split]) | (set(new_names[task]) if split == "train" else set())
            have = _listing(staging, task, split)
            if set(have) != expected or len(have) != len(expected):
                problems.append(f"{task}/{split}: output has {len(have)} entries, expected {len(expected)}")
            for name in have:
                if sha256_file(_path(staging, task, split, name)) != out_hashes[task][split].get(name):
                    problems.append(f"{task}/{split}/{name}: output hash differs")
        for name in base_records[task]["test"]:
            if sha256_file(_path(staging, task, "test", name)) != sha256_file(_path(args.baseline, task, "test", name)):
                problems.append(f"test file changed during copy: {task}/{name}")
        for name in base_records[task]["train"]:
            if sha256_file(_path(staging, task, "train", name)) != sha256_file(_path(args.baseline, task, "train", name)):
                problems.append(f"baseline train file changed during copy: {task}/{name}")
        for name in new_names[task]:
            if sha256_file(_path(staging, task, "train", name)) != sha256_file(_path(args.new, task, "train", name)):
                problems.append(f"new file changed during copy: {task}/{name}")
    if sorted(os.listdir(staging)) != sorted([f"benchmark_{t}" for t in TASKS] + ["merge_manifest.json"]):
        problems.append(f"unexpected entries in the output: {sorted(os.listdir(staging))}")
    with open(os.path.join(staging, "merge_manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    totals = m["output_counts"]
    want_train = sum(args.expected_new.values()) + len(TASKS) * args.expected_baseline_train
    want_test = len(TASKS) * args.expected_baseline_test
    if (totals["train_total"], totals["test_total"]) != (want_train, want_test):
        problems.append(f"output totals {totals['train_total']}/{totals['test_total']}, expected {want_train}/{want_test}")
    if problems:
        raise MergeAbort("Output verification failed:\n  " + "\n  ".join(problems[:20]))


def _write_jsonl(args, staging, jsonl_staging, base_records, out_counts):
    """Writes the published-format export from the verified output and re-reads it to check counts and content."""
    info = {"counts": {}, "baseline_lines_byte_identical_to_published": 0}
    for task in TASKS:
        os.makedirs(os.path.join(jsonl_staging, task), exist_ok=True)
        for split in ("train", "test"):
            path = os.path.join(jsonl_staging, task, f"{split}.jsonl")
            names = _files(staging, task, split)
            with open(path, "x", encoding="utf-8") as f:
                for name in names:
                    f.write(json.dumps(encode_record(_load(staging, task, split, name))) + "\n")
            with open(path, encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            if len(lines) != out_counts[task][split]:
                raise MergeAbort(f"JSONL {task}/{split}: {len(lines)} lines, expected {out_counts[task][split]}")
            for name, line in zip(names, lines):
                if json.loads(line) != encode_record(_load(staging, task, split, name)):
                    raise MergeAbort(f"JSONL {task}/{split}: line for {name} does not match its file")
            with open(os.path.join(args.baseline, "original_files", task, f"{split}.jsonl"), encoding="utf-8") as f:
                published = {line.rstrip("\n") for line in f if line.strip()}
            ours = {line.rstrip("\n") for name, line in zip(names, lines) if name in base_records[task][split]}
            if ours != published:
                raise MergeAbort(f"JSONL {task}/{split}: published rows are not reproduced byte for byte")
            info["baseline_lines_byte_identical_to_published"] += len(ours)
            info["counts"][f"{task}/{split}"] = len(lines)
    return info


if __name__ == "__main__":
    main()
