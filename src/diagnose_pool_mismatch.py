"""Read-only diagnosis of "Only X/Y published train schemas are in the reconstructed training pool".

No model, nothing written, nothing deleted. It answers, for every published train id that is missing
from the rebuilt training pool, WHY it is missing:

    in_train_pool         fine
    in_test_partition     the schema exists but the seeded split put it on the TEST side   -> partitioning
    over_size_limit       the schema file exists but is larger than max_schema_kb         -> filtering
    unreadable_file       the file exists but is not valid JSON / has no schema           -> source data
    different_id_same_content   no file has this id, but a loaded schema has the same content hash
                                                                                         -> identity differs
    not_found             nothing on disk has this id or content                         -> missing source data

It also fingerprints the corpus (file counts and git tree ids of data/schemas and data/jsb_schemas against
the values of the original v2 build, commit e7f8b22) and the rebuilt partition, so two checkouts (Mac and
cluster) can be compared by value.

    python src/diagnose_pool_mismatch.py --reserved data_published/<sha12>/reserved_schemas.json
        [--config configs/scale_only_vllm_granite.json] [--git-baseline e7f8b22]

--git-baseline derives the ORIGINAL v2 train/test ids from that commit's data/benchmark_*/{train,test} file
names and checks them against the rebuilt pool (needs the commit in the local clone).
Exit code: 0 = every published train id is in the rebuilt train pool, 1 = not, 2 = could not run.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_generation.benchmark_builder import BenchmarkBuilder          # noqa: E402
from data_generation.published import schema_hash                       # noqa: E402
import prepare_jsb_schemas as pjs                                       # noqa: E402

ORIGINAL_COMMIT = "e7f8b22"
# git tree ids of the corpus folders at the original v2 build (git rev-parse e7f8b22:data/<name>)
PINNED_CORPUS = {
    "schemas": {"tree": "ba23389d3fa99c945945fdfa9fa2ac7c5849486a", "files": 700},
    "jsb_schemas": {"tree": pjs.PINNED_TREE_OID, "files": pjs.PINNED_FILE_COUNT},
}
TASKS = ("create", "fix", "modify")


def _group(rid):
    parts = rid.split("__")
    return "/".join(parts[:2]) if parts[0] == "jsonschemabench" and len(parts) > 2 else parts[0]


def corpus_report(root=ROOT, names=("schemas", "jsb_schemas")):
    out = {}
    for name in names:
        folder = os.path.join(root, "data", name)
        entry = {"path": folder, "exists": os.path.isdir(folder)}
        if entry["exists"]:
            entry["files"] = pjs.count_files(folder)
            try:
                entry["tree_oid"] = pjs.tree_oid(folder)
            except pjs.PrepareError as e:
                entry["tree_oid"] = f"error: {e}"
            pin = PINNED_CORPUS.get(name)
            if pin:
                entry["matches_original_build"] = (entry["files"] == pin["files"] and entry["tree_oid"] == pin["tree"])
                entry["pinned"] = pin
        out[name] = entry
    return out


def _scan_all(builder, sources):
    """Every json under data/<source>/test regardless of size: id -> {size, readable, hash}."""
    found = {}
    for src in sources:
        folder = os.path.join(os.getcwd(), "data", src["name"], "test")
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(folder, fn)
            info = {"size": os.path.getsize(path), "readable": False, "hash": None, "limit_kb": src.get("max_schema_kb")}
            try:
                with open(path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and "schema" in loaded:
                    schema, source, subset = loaded["schema"], loaded.get("source", src["name"]), loaded.get("subset")
                    origin = loaded.get("name") or os.path.basename(path)
                else:
                    schema, source, subset = loaded, src["name"], None
                    origin = os.path.splitext(fn)[0]
                if isinstance(schema, dict):
                    info.update(readable=True, hash=schema_hash(schema))
                    found[builder._safe_id(source, subset, origin)] = info
                    continue
            except (OSError, ValueError):
                pass
            found[builder._safe_id(src["name"], None, os.path.splitext(fn)[0])] = info
    return found


def git_baseline_ids(commit, repo=ROOT):
    """Original train/test ids = file names under data/benchmark_<task>/{train,test} at `commit`."""
    ids = {"train": set(), "test": set()}
    for task in TASKS:
        for split in ids:
            r = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit, f"data/benchmark_{task}/{split}"],
                               cwd=repo, capture_output=True, text=True)
            if r.returncode != 0:
                return None
            ids[split] |= {os.path.basename(p)[:-5] for p in r.stdout.splitlines() if p.endswith(".json")}
    return ids


def diagnose(reserved_path, sources, seed=0, test_ratio=0.15, git_commit=None, root=ROOT):
    cwd = os.getcwd()
    os.chdir(root)
    try:
        cfg = {"sources": sources, "seed": seed, "split": {"test_ratio": test_ratio},
               "counts": {"train": {t: 1 for t in TASKS}}}
        b = BenchmarkBuilder(cfg)
        records = b._load_schema_records(apply_exclusions=False)
        part = b._partition(records)
        sid = lambda r: b._safe_id(r[1], r[2], r[3])
        train_pool = {sid(r) for r in part["train"]}
        test_pool = {sid(r) for r in part["test"]}
        content = {schema_hash(r[0]) for r in records}
        scan = _scan_all(b, sources)
        with open(reserved_path, encoding="utf-8") as f:
            res = json.load(f)
        train_ids = set(res["ids"]["train"]); test_ids = set(res["ids"]["test"])

        reasons = collections.Counter()
        by_group = collections.defaultdict(collections.Counter)
        examples = collections.defaultdict(list)
        for rid in sorted(train_ids):
            if rid in train_pool:
                why = "in_train_pool"
            elif rid in test_pool:
                why = "in_test_partition"
            elif rid in scan and not scan[rid]["readable"]:
                why = "unreadable_file"
            elif rid in scan and scan[rid]["limit_kb"] is not None and scan[rid]["size"] > scan[rid]["limit_kb"] * 1024:
                why = "over_size_limit"
            else:
                why = "not_found"
            reasons[why] += 1
            by_group[_group(rid)][why] += 1
            if why != "in_train_pool" and len(examples[why]) < 5:
                examples[why].append(rid)
        hashes_in_corpus = len(set(res["schema_hashes"]["train"]) & content)
        out = {
            "corpus": corpus_report(root, [s["name"] for s in sources]),
            "rebuilt": {"eligible": len(records), "train_pool": len(train_pool), "test_pool": len(test_pool),
                        "by_group_train_pool": dict(sorted(collections.Counter(
                            "/".join(p for p in (r[1], r[2]) if p) for r in part["train"]).items())),
                        "train_pool_id_fingerprint": hashlib.sha256("\n".join(sorted(train_pool)).encode()).hexdigest()[:16],
                        "test_pool_id_fingerprint": hashlib.sha256("\n".join(sorted(test_pool)).encode()).hexdigest()[:16]},
            "published": {"revision": res.get("revision"), "train_ids": len(train_ids), "test_ids": len(test_ids),
                          "train_ids_by_group": dict(sorted(collections.Counter(_group(i) for i in train_ids).items())),
                          "train_schema_hashes": len(res["schema_hashes"]["train"]),
                          "train_hashes_present_in_rebuilt_corpus": hashes_in_corpus},
            "train_id_reasons": dict(reasons),
            "train_id_reasons_by_group": {g: dict(c) for g, c in sorted(by_group.items())},
            "examples": dict(examples),
            "published_test_ids_in_test_pool": len(test_ids & test_pool),
            "published_test_ids_in_train_pool": len(test_ids & train_pool),
        }
        # content present although the id is not: identity differs rather than data missing
        missing = [i for i in train_ids if i not in train_pool and i not in test_pool and i not in scan]
        out["not_found_ids_total"] = len(missing)
        out["verdict"] = _verdict(out)
        if git_commit:
            base = git_baseline_ids(git_commit, root)
            if base is None:
                out["git_baseline"] = {"commit": git_commit, "error": "commit or its data/benchmark_* folders not in this clone"}
            else:
                out["git_baseline"] = {
                    "commit": git_commit, "train_ids": len(base["train"]), "test_ids": len(base["test"]),
                    "train_ids_in_rebuilt_train_pool": len(base["train"] & train_pool),
                    "test_ids_in_rebuilt_test_pool": len(base["test"] & test_pool),
                    "git_train_ids_not_in_reserved_train": len(base["train"] - train_ids),
                    "reserved_train_ids_not_in_git_train": len(train_ids - base["train"])}
        return out
    finally:
        os.chdir(cwd)


def _verdict(r):
    reasons, notes = r["train_id_reasons"], []
    missing = sum(v for k, v in reasons.items() if k != "in_train_pool")
    if missing == 0:
        return ["every published train id is in the rebuilt training pool"]
    for name, e in r["corpus"].items():
        if e.get("matches_original_build") is False:
            notes.append(f"corpus data/{name} differs from the original build ({e.get('files')} files, tree {e.get('tree_oid')}) -> source data")
        if not e["exists"]:
            notes.append(f"data/{name} is missing")
    if reasons.get("not_found"):
        notes.append(f"{reasons['not_found']} ids have no file with that id: missing source data or different identity "
                     f"(content hashes present: {r['published']['train_hashes_present_in_rebuilt_corpus']}/{r['published']['train_schema_hashes']})")
    if reasons.get("in_test_partition"):
        notes.append(f"{reasons['in_test_partition']} ids sit on the test side of the seeded split -> partitioning (seed/ratio/pool composition)")
    if reasons.get("over_size_limit"):
        notes.append(f"{reasons['over_size_limit']} ids exceed max_schema_kb -> filtering")
    if reasons.get("unreadable_file"):
        notes.append(f"{reasons['unreadable_file']} ids point to unreadable files -> source data")
    return notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reserved", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "scale_only_vllm_granite.json"))
    ap.add_argument("--git-baseline", default=None, help="e.g. e7f8b22 (original v2 build)")
    ap.add_argument("--root", default=ROOT, help="checkout whose data/ folders are analysed (default: this repo)")
    args = ap.parse_args(argv)
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
        sources = [{"name": s["name"], "max_schema_kb": s.get("max_schema_kb", 10)} for s in cfg["sources"]]
        report = diagnose(args.reserved, sources, cfg.get("seed", 0), cfg.get("split", {}).get("test_ratio", 0.15),
                          args.git_baseline, root=os.path.abspath(args.root))
    except (OSError, ValueError, KeyError) as e:
        print(f"CANNOT RUN: {type(e).__name__}: {e}")
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["train_id_reasons"].get("in_train_pool", 0) == report["published"]["train_ids"] else 1


if __name__ == "__main__":
    sys.exit(main())
