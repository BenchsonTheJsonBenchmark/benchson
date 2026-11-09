"""The published-baseline-anchored schema pool (protocol "Approach A") for the scale-only experiment.

The published Hugging Face snapshot was NOT built from the schema corpus/partition that git commit e7f8b22
documents (its train schemas do not line up with that reconstruction), so the pool is taken from the published
training rows themselves: every distinct schema that occurs in a published TRAIN example, with the id, content,
provenance (source, subset, name) and content hash exactly as published. Nothing is re-derived from data/.

The anchor file `published_train_schemas.json` is built once, offline, from the read-only baseline folder written by
`import_published_benchmark.py` (`benchmark_<task>/{train,test}/<id>.json`), and verified against
`reserved_schemas.json` every time it is loaded. It never modifies the baseline.
"""
import hashlib
import json
import os

from data_generation.published import (TASKS, decode_row, record_id, safe_id, schema_hash)

FORMAT = "benchson-anchor-pool/1"
ANCHOR_FILENAME = "published_train_schemas.json"

# Descriptive keywords ignored by the STRUCTURAL signature (a report only, never a filter).
_ANNOTATIONS = {"title", "description", "examples", "example", "$comment", "$id", "id", "$schema", "default",
                "deprecated", "readOnly", "writeOnly"}
# Keywords whose value maps property/definition NAMES to schemas: the names are content, not annotations.
_NAME_MAPS = {"properties", "patternProperties", "definitions", "$defs", "dependentSchemas"}


class AnchorError(ValueError):
    pass


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def schema_size_bytes(schema):
    """The size used by the 10 KB restriction: UTF-8 bytes of the canonical compact JSON of the schema.

    Defined on the schema alone (independent of file layout or indentation), so it can be recomputed from the
    published rows. It differs from the on-disk file size the original `max_schema_kb` filter used.
    """
    return len(_canonical(schema).encode("utf-8"))


def _strip(node, is_name_map=False):
    if isinstance(node, dict):
        if is_name_map:
            return {k: _strip(v) for k, v in node.items()}
        return {k: _strip(v, k in _NAME_MAPS) for k, v in node.items() if k not in _ANNOTATIONS}
    if isinstance(node, list):
        return [_strip(x) for x in node]
    return node


def structure_hash(schema):
    """Hash of the schema without descriptive annotations (titles, descriptions, examples, defaults, ids)."""
    return hashlib.sha256(_canonical(_strip(schema)).encode("utf-8")).hexdigest()


def _read_split(baseline_dir, task, split):
    folder = os.path.join(baseline_dir, f"benchmark_{task}", split)
    if os.path.isdir(folder):
        records = []
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".json"):
                with open(os.path.join(folder, fn), encoding="utf-8") as f:
                    records.append(json.load(f))
        return records
    jsonl = os.path.join(baseline_dir, "original_files", task, f"{split}.jsonl")
    if os.path.isfile(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            return [decode_row(task, json.loads(line)) for line in f if line.strip()]
    raise AnchorError(f"no published {task}/{split} records under {baseline_dir}")


def build_anchor(baseline_dir):
    """Builds the anchor structure (a dict) from a baseline folder. Deterministic and read-only."""
    train, test = {}, {}
    for split, store in (("train", train), ("test", test)):
        for task in TASKS:
            for rec in _read_split(baseline_dir, task, split):
                rid = record_id(rec)
                schema = rec["schema"]
                if not isinstance(schema, dict):
                    raise AnchorError(f"{task}/{split}/{rid}: schema is not an object")
                h = schema_hash(schema)
                if rid in store:
                    if store[rid]["schema_hash"] != h:
                        raise AnchorError(f"{split}: id {rid} carries two different schemas across tasks")
                    store[rid]["tasks"].append(task)
                    continue
                store[rid] = {"id": rid, "name": rec["name"], "source": rec.get("source"), "subset": rec.get("subset"),
                              "schema": schema, "schema_hash": h, "structure_hash": structure_hash(schema),
                              "size_bytes": schema_size_bytes(schema), "tasks": [task]}
    revision = dataset = manifest_sha = None
    reserved = os.path.join(baseline_dir, "reserved_schemas.json")
    if os.path.isfile(reserved):
        with open(reserved, encoding="utf-8") as f:
            r = json.load(f)
        revision, dataset = r.get("revision"), r.get("dataset")
    manifest = os.path.join(baseline_dir, "published_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "rb") as f:
            manifest_sha = hashlib.sha256(f.read()).hexdigest()
    body = {"train": [dict(v, tasks=sorted(set(v["tasks"]))) for _, v in sorted(train.items())],
            "test": [{"id": v["id"], "schema_hash": v["schema_hash"], "structure_hash": v["structure_hash"]}
                     for _, v in sorted(test.items())]}
    return {"format": FORMAT, "dataset": dataset, "revision": revision, "published_manifest_sha256": manifest_sha,
            "counts": {"train_schemas": len(body["train"]), "test_schemas": len(body["test"])},
            "anchor_sha256": hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(), **body}


def write_anchor(baseline_dir, out_path=None):
    """Writes the anchor file next to the baseline's reserved_schemas.json. Refuses to overwrite."""
    out_path = out_path or os.path.join(baseline_dir, ANCHOR_FILENAME)
    if os.path.exists(out_path):
        raise AnchorError(f"{out_path} already exists; refusing to overwrite")
    anchor = build_anchor(baseline_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(anchor, f, ensure_ascii=False, indent=1)
    return anchor


def load_anchor(path, reserved_path=None):
    """Loads and VERIFIES an anchor file. Any inconsistency raises AnchorError (there is no lenient mode)."""
    try:
        with open(path, encoding="utf-8") as f:
            anchor = json.load(f)
    except (OSError, ValueError) as e:
        raise AnchorError(f"cannot read anchor file {path}: {e} (create it with: python src/export_anchor_pool.py "
                          "--baseline-dir data_published/<sha12>)") from None
    if not isinstance(anchor, dict) or anchor.get("format") != FORMAT:
        raise AnchorError(f"{path}: unsupported format {anchor.get('format') if isinstance(anchor, dict) else None!r}")
    train, test = anchor.get("train"), anchor.get("test")
    if not isinstance(train, list) or not isinstance(test, list) or not train:
        raise AnchorError(f"{path}: missing or empty train/test lists")
    body = {"train": train, "test": test}
    if hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest() != anchor.get("anchor_sha256"):
        raise AnchorError(f"{path}: anchor_sha256 does not match the content (file modified or corrupt)")
    seen = set()
    for e in train:
        for k in ("id", "name", "schema", "schema_hash", "structure_hash", "size_bytes", "tasks"):
            if k not in e:
                raise AnchorError(f"{path}: train entry without '{k}': {str(e)[:80]}")
        if not isinstance(e["schema"], dict):
            raise AnchorError(f"{path}: schema of {e['id']} is not an object")
        if e["id"] in seen:
            raise AnchorError(f"{path}: duplicate train id {e['id']}")
        seen.add(e["id"])
        if safe_id(e["source"], e["subset"], e["name"]) != e["id"]:
            raise AnchorError(f"{path}: id {e['id']} does not match source/subset/name")
        if schema_hash(e["schema"]) != e["schema_hash"]:
            raise AnchorError(f"{path}: schema content of {e['id']} does not match its schema_hash")
        if structure_hash(e["schema"]) != e["structure_hash"] or schema_size_bytes(e["schema"]) != e["size_bytes"]:
            raise AnchorError(f"{path}: derived fields of {e['id']} do not match its schema")
    if reserved_path:
        with open(reserved_path, encoding="utf-8") as f:
            res = json.load(f)
        if anchor.get("revision") != res.get("revision"):
            raise AnchorError(f"anchor revision {anchor.get('revision')} differs from reserved revision {res.get('revision')}")
        checks = {
            "train ids": (seen, set(res["ids"]["train"])),
            "train schema hashes": ({e["schema_hash"] for e in train}, set(res["schema_hashes"]["train"])),
            "test ids": ({e["id"] for e in test}, set(res["ids"]["test"])),
            "test schema hashes": ({e["schema_hash"] for e in test}, set(res["schema_hashes"]["test"])),
        }
        for what, (a, b) in checks.items():
            if a != b:
                raise AnchorError(f"anchor and reserved_schemas.json disagree on {what}: "
                                  f"{len(a - b)} only in anchor, {len(b - a)} only in reserved")
    return anchor


def select_pool(anchor, max_schema_kb, dedupe_by_content=True):
    """Applies the exclusions to the published train schemas. Returns (kept_entries, statistics).

    Order of exclusions (each schema is counted once, under the first rule that removes it):
    published TEST id, published TEST content hash, size restriction, content duplicate inside the pool.
    Structural overlap with the published test set is reported but never used to filter.
    """
    test_ids = {e["id"] for e in anchor["test"]}
    test_hashes = {e["schema_hash"] for e in anchor["test"]}
    test_struct = {}
    for e in anchor["test"]:
        test_struct.setdefault(e["structure_hash"], []).append(e["id"])
    limit = int(float(max_schema_kb) * 1024)
    kept, seen = [], set()
    stats = {"published_train_schemas": len(anchor["train"]), "published_test_schemas": len(anchor["test"]),
             "excluded_published_test_by_id": 0, "excluded_published_test_by_content_hash": 0,
             "excluded_over_size_limit": 0, "removed_content_duplicates_in_pool": 0}
    for e in sorted(anchor["train"], key=lambda x: x["id"]):
        if e["id"] in test_ids:
            stats["excluded_published_test_by_id"] += 1
        elif e["schema_hash"] in test_hashes:
            stats["excluded_published_test_by_content_hash"] += 1
        elif e["size_bytes"] > limit:
            stats["excluded_over_size_limit"] += 1
        elif dedupe_by_content and e["schema_hash"] in seen:
            stats["removed_content_duplicates_in_pool"] += 1
        else:
            seen.add(e["schema_hash"])
            kept.append(e)
    by_source = {}
    for e in kept:
        key = "/".join(p for p in (e["source"], e["subset"]) if p) or "unknown"
        by_source[key] = by_source.get(key, 0) + 1
    # Reports only (nothing is removed because of them).
    struct_overlap = sorted(e["id"] for e in kept if e["structure_hash"] in test_struct)
    pool_struct = {}
    for e in kept:
        pool_struct.setdefault(e["structure_hash"], []).append(e["id"])
    stats.update(
        size_restriction={"max_schema_kb": max_schema_kb, "measure": "UTF-8 bytes of canonical compact JSON of the schema",
                          "experimental_restriction_not_a_claim_about_the_hf_build": True},
        usable_schemas=len(kept), usable_by_source=dict(sorted(by_source.items())),
        structural_overlap_with_published_test={"pool_schemas": len(struct_overlap), "ids_first_20": struct_overlap[:20],
                                                "filtered": False},
        structural_duplicates_inside_pool={"groups": sum(1 for v in pool_struct.values() if len(v) > 1),
                                           "schemas_in_groups": sum(len(v) for v in pool_struct.values() if len(v) > 1),
                                           "filtered": False},
        anchor_sha256=anchor["anchor_sha256"], revision=anchor.get("revision"))
    return kept, stats
