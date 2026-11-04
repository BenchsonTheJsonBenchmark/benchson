"""How many schemas exist, how many were used by the published dataset, and what is left?

Read-only analysis (no model, no writes). It rebuilds the ORIGINAL schema pool and train/test
partition (seed 0, test_ratio 0.15, max_schema_kb 10 - the values recorded in the published
manifests) and compares it with the ids of the published examples.

    python src/analyze_schema_pool.py --reserved data_published/<sha12>/reserved_schemas.json [--targets 1334,1333,1333]
Without --reserved only the pool/partition part is printed.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.published import TASKS, schema_hash

# Recorded in the published v2 manifests (git e7f8b22): pool 1669 = test 250 + train 1419.
EXPECTED_V2 = {"eligible": 1669, "test_pool": 250, "train_pool": 1419}


def analyze(reserved_path=None, seed=0, test_ratio=0.15, max_kb=10, sources=("schemas", "jsb_schemas"),
            targets=(1334, 1333, 1333), assumed_acceptance=0.7):
    cfg = {"llm_provider": {}, "sources": [{"name": n, "max_schema_kb": max_kb} for n in sources],
           "seed": seed, "split": {"test_ratio": test_ratio}, "output_root": "/nonexistent-analysis-only",
           "counts": {"test": {t: 1 for t in TASKS}, "train": {t: 1 for t in TASKS}}}
    b = BenchmarkBuilder(cfg)
    recs = b._load_schema_records()
    pools = b._partition(recs)
    sid = lambda r: b._safe_id(r[1], r[2], r[3])
    train_ids, test_ids = {sid(r) for r in pools["train"]}, {sid(r) for r in pools["test"]}
    hashes = collections.Counter(schema_hash(r[0]) for r in recs)
    out = {"eligible_schemas": len(recs), "original_partition": {"test_pool": len(pools["test"]),
           "train_pool": len(pools["train"])},
           "matches_published_v2_manifest": {"eligible": len(recs) == EXPECTED_V2["eligible"],
                                              "test_pool": len(pools["test"]) == EXPECTED_V2["test_pool"],
                                              "train_pool": len(pools["train"]) == EXPECTED_V2["train_pool"]},
           "by_source_train_pool": {f"{k[0]}/{k[1]}": v for k, v in sorted(collections.Counter(
               (r[1], r[2]) for r in pools["train"]).items(), key=str)},
           "distinct_contents": len(hashes), "contents_present_more_than_once": sum(v > 1 for v in hashes.values()),
           "content_overlap_between_pools": len({schema_hash(r[0]) for r in pools["train"]} &
                                                 {schema_hash(r[0]) for r in pools["test"]})}
    if not reserved_path:
        return out
    res = json.load(open(reserved_path))
    by_task = res["ids_by_task"]
    used_train = set(res["ids"]["train"])
    used_test = set(res["ids"]["test"])
    out["published"] = {"revision": res.get("revision"), "unique_train_ids": len(used_train),
                        "unique_test_ids": len(used_test),
                        "train_ids_not_in_reconstructed_pool": len(used_train - train_ids - test_ids),
                        "test_ids_not_in_reconstructed_pool": len(used_test - train_ids - test_ids),
                        "train_ids_found_in_original_test_pool": len(used_train & test_ids),
                        "test_ids_found_in_original_train_pool": len(used_test & train_ids)}
    unused_strict = train_ids - used_train - used_test
    leftover_test_pool = test_ids - used_test - used_train
    per_task = {t: len(train_ids - set(by_task[t]["train"]) - used_test) for t in TASKS}
    out["unused_after_excluding_published"] = {
        "A_strict_schema_never_used_by_any_published_task__train_pool": len(unused_strict),
        "extra_never_used_schemas_in_original_test_pool": len(leftover_test_pool),
        "B_per_task_train_pool_schemas_without_that_task_example": per_task}
    sc = {}
    for label, avail in (("A_strict", {t: len(unused_strict) for t in TASKS}),
                         ("A_strict+leftover_test_pool", {t: len(unused_strict) + len(leftover_test_pool) for t in TASKS}),
                         ("B_per_task", per_task)):
        sc[label] = {"upper_bound_per_task": avail,
                     "upper_bound_total": sum(avail.values()),
                     "realistic_total_at_acceptance_%.0f%%" % (assumed_acceptance * 100):
                         int(sum(avail.values()) * assumed_acceptance),
                     "reaches_target": all(avail[t] >= n for t, n in zip(TASKS, targets))}
    out["capacity_without_reusing_schemas"] = sc
    out["targets_per_task"] = dict(zip(TASKS, targets))
    out["note"] = ("Upper bounds assume every remaining schema yields an accepted example. For B, schemas the original "
                   "run already TRIED and rejected for that task are included; their real yield is unknown.")
    return out


def analyze_expansion(expanded_source, reserved_path=None, max_kb=10):
    """Compares an EXPANDED copy of the JSONSchemaBench import with the original pool.

    `expanded_source` is a data/<name> folder produced by import_jsonschemabench.py with larger
    per-subset counts (same seed and subset order), written next to - never over - data/jsb_schemas.
    """
    def ids_of(sources):
        b = BenchmarkBuilder({"llm_provider": {}, "sources": [{"name": n, "max_schema_kb": max_kb} for n in sources],
                              "seed": 0, "output_root": "/nonexistent-analysis-only"})
        recs = b._load_schema_records()
        return {b._safe_id(r[1], r[2], r[3]): schema_hash(r[0]) for r in recs}
    original = ids_of(("schemas", "jsb_schemas"))
    expanded = ids_of(("schemas", expanded_source))
    new = {i: h for i, h in expanded.items() if i not in original}
    published_hashes = set()
    if reserved_path:
        published_hashes = set(json.load(open(reserved_path)).get("schema_hashes", {}).get("train", [])) | \
                           set(json.load(open(reserved_path)).get("schema_hashes", {}).get("test", []))
    original_hashes = set(original.values())
    return {"original_pool": len(original), "expanded_pool": len(expanded),
            "original_ids_missing_from_expanded": sum(1 for i in original if i not in expanded),
            "prefix_property_holds": all(i in expanded for i in original),
            "new_schemas": len(new),
            "new_schemas_with_content_identical_to_an_original_schema": sum(h in original_hashes for h in new.values()),
            "new_schemas_with_content_identical_to_a_published_schema": sum(h in published_hashes for h in new.values()),
            "new_by_prefix": dict(collections.Counter(i.split("__")[1] if "__" in i else "?" for i in new))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reserved")
    ap.add_argument("--targets", default="1334,1333,1333")
    ap.add_argument("--assumed-acceptance", type=float, default=0.7)
    ap.add_argument("--expanded-source", help="name of an expanded jsb import under data/ to compare with the original pool")
    a = ap.parse_args()
    if a.expanded_source:
        print(json.dumps(analyze_expansion(a.expanded_source, a.reserved), indent=2))
        sys.exit(0)
    print(json.dumps(analyze(a.reserved, targets=tuple(int(x) for x in a.targets.split(",")),
                             assumed_acceptance=a.assumed_acceptance), indent=2))
