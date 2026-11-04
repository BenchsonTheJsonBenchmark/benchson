"""Helpers for the published Benchson dataset (Hugging Face) and for keeping it untouched.

The published dataset is stored as JSONL with every dict/list value turned into a JSON
string (see export_hf._stringify). The builder, on the other hand, works with one
JSON file per example, named after the schema. These helpers convert between the two
losslessly and compute the identifiers used to keep train and test schemas disjoint.
"""

import hashlib
import json
import os
import re

TASKS = ("create", "fix", "modify")

# Fields that hold JSON objects/arrays (stringified in the published JSONL).
JSON_FIELDS = {
    "create": ("schema", "reference_json"),
    "fix": ("schema", "erroneous_json", "valid_json"),
    "modify": ("schema", "data", "ground_truth", "modification"),
}

# Fields every record must have; anything else a published row carries is kept as it is.
REQUIRED_FIELDS = {
    "create": {"name", "source", "subset", "schema", "description", "reference_json"},
    "fix": {"name", "source", "subset", "schema", "erroneous_json", "valid_json"},
    "modify": {"name", "source", "subset", "schema", "data", "instructions", "ground_truth",
               "modification"},
}

_SHA = re.compile(r"^[0-9a-f]{40}$")


def is_pinned_revision(revision):
    """A full 40-character commit hash (branch names and tags can move, so they do not count)."""
    return bool(revision) and bool(_SHA.match(revision))


def safe_id(source, subset, origin):
    """Identical to BenchmarkBuilder._safe_id: the file name (without .json) of an example."""
    raw = "__".join(p for p in (source, subset, origin) if p)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)[:150]


def schema_hash(schema):
    """Content hash of a schema, independent of its name, file and key order."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_id(record):
    return safe_id(record.get("source"), record.get("subset"), record.get("name"))


def decode_row(task, row):
    """Published JSONL row -> builder record (stringified JSON fields parsed back)."""
    record = dict(row)
    for field in JSON_FIELDS[task]:
        if isinstance(record.get(field), str):
            record[field] = json.loads(record[field])
    return record


def encode_record(record):
    """Builder record -> published JSONL row (exactly export_hf._stringify)."""
    return {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in record.items()}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reserved(path):
    """Reads a reserved-schemas file (written by import_published_benchmark.py).

    Returns (ids, hashes, info): every schema id / schema hash that already appears in the
    published train or test split and therefore must not be used for new training data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set(data["ids"].get("train", [])) | set(data["ids"].get("test", []))
    hashes = set(data["schema_hashes"].get("train", [])) | set(data["schema_hashes"].get("test", []))
    info = {"file": str(path), "dataset": data.get("dataset"), "revision": data.get("revision"),
            "reserved_ids": len(ids), "reserved_schema_hashes": len(hashes)}
    return ids, hashes, info


def git_blob_oid(path):
    """The git blob id of a file (what the Hugging Face tree API reports as `oid`)."""
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


# Read from the Hugging Face API (https://huggingface.co/api/datasets/aviv1ron1/Benchson/tree/<sha>?recursive=true)
# on 2026-11-04. The six data files are identical in `main` and in the `v2` tag (58b8525c...); only README.md differs.
KNOWN_REVISIONS = {
    "7962b0947f7d96eb3a0e6b9b928d939c7f3061a6": {
        "repo": "aviv1ron1/Benchson", "branch": "main", "last_modified": "2026-06-29T14:39:53Z",
        "files": {
            "create/test.jsonl": ("87c9566c6bca885a04b00f047e1e45537be05ab9", 966579),
            "create/train.jsonl": ("75cd32160768596abb5ccbc00abd15d2bec7bbc2", 6509886),
            "fix/test.jsonl": ("3592f6aa9d8a4d66753f2d13de6521230ec13b2f", 1092347),
            "fix/train.jsonl": ("e9389511a980582bccf17ea7273867bd126f7c50", 8057429),
            "modify/test.jsonl": ("57b9d99697626c2515d0469f04cd5d5b1d647b34", 907630),
            "modify/train.jsonl": ("9bd97e2a9b8cb3660cb6fc6d082f80e6cd7d33b7", 6552209),
        },
        # The dataset card of the same revision (git blob id, size), read from the same API response.
        # Its front matter says `license: mit`. The tree also holds a stray .DS_Store, which is never copied.
        "other_files": {"README.md": ("ea2f7f6ac568e630819a41a47cb9b851ea85507b", 5305)},
        "license": "mit",
    },
}


def parse_card_license(card_text):
    """`license:` value from the YAML front matter of a Hugging Face dataset card, or None."""
    lines = card_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        m = re.match(r"^license:\s*(\S+)\s*$", line)
        if m:
            return m.group(1).strip("'\"").lower()
    return None


# ---------------------------------------------------------------- content fingerprints (duplicate detection)

def canon_hash(*parts):
    """sha1 of the canonical JSON of `parts` (key order and whitespace do not matter)."""
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# The task-relevant JSON fields. The first one is the "reference object" the example was built from.
CONTENT_FIELDS = {
    "create": ("reference_json",),
    "fix": ("valid_json", "erroneous_json"),
    "modify": ("data", "ground_truth"),
}


def reference_of(task, record):
    """The ground-truth object an example is built around (shared by the three tasks in the original design)."""
    return record[CONTENT_FIELDS[task][0]]


def task_keys(task, record):
    """(exact, content): `exact` includes the schema, `content` ignores it (same JSON under another schema)."""
    values = [record.get(f) for f in CONTENT_FIELDS[task]]
    return canon_hash(schema_hash(record["schema"]), *values), canon_hash(*values)


def reference_hash(task, record):
    return canon_hash(reference_of(task, record))


def load_reference_index(path):
    """From reserved_schemas.json: ({schema_hash: {reference_hash}}, {task: {"exact": set, "content": set}}).
    Raises ValueError when the file was written by an older importer."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "reference_hashes_by_schema" not in data or "task_keys" not in data:
        raise ValueError(f"{path} has no content fingerprints; re-run import_published_benchmark.py")
    refs = {k: set(v) for k, v in data["reference_hashes_by_schema"].items()}
    keys = {t: {"exact": set(v.get("exact", [])), "content": set(v.get("content", []))}
            for t, v in (data.get("task_keys") or {}).items()}
    return refs, keys


def overlap_report(orig_dir):
    """Train/test overlap inside a published snapshot (`<orig_dir>/<task>/{train,test}.jsonl`).

    A schema counts as shared when its canonical content hash (key order and file name ignored) occurs in both splits.
    Example overlap is checked four ways: same task fields incl. schema, same task fields ignoring the schema, same
    reference object on the same schema, and byte-identical decoded rows. Returns ids and counts only.
    """
    per, union = {}, {}
    for task in TASKS:
        rows = {}
        for split in ("train", "test"):
            with open(os.path.join(orig_dir, task, f"{split}.jsonl"), encoding="utf-8") as fh:
                rows[split] = [decode_row(task, json.loads(line)) for line in fh if line.strip()]
        by_hash = {sp: {} for sp in rows}
        for sp, rs in rows.items():
            for r in rs:
                by_hash[sp].setdefault(schema_hash(r["schema"]), []).append(r)
        common = set(by_hash["train"]) & set(by_hash["test"])
        exact = {sp: {task_keys(task, r)[0] for r in rows[sp]} for sp in rows}
        content = {sp: {task_keys(task, r)[1] for r in rows[sp]} for sp in rows}
        refs = {sp: {(schema_hash(r["schema"]), reference_hash(task, r)) for r in rows[sp]} for sp in rows}
        full = {sp: {json.dumps(r, sort_keys=True) for r in rows[sp]} for sp in rows}
        per[task] = {"shared_schemas": len(common),
                     "identical_examples_incl_schema": len(exact["train"] & exact["test"]),
                     "identical_examples_ignoring_schema": len(content["train"] & content["test"]),
                     "same_schema_and_same_reference": len(refs["train"] & refs["test"]),
                     "identical_rows": len(full["train"] & full["test"])}
        for h in common:
            u = union.setdefault(h, {"tasks": [], "train_ids": set(), "test_ids": set(), "sources": set()})
            u["tasks"].append(task)
            u["train_ids"] |= {record_id(r) for r in by_hash["train"][h]}
            u["test_ids"] |= {record_id(r) for r in by_hash["test"][h]}
            u["sources"] |= {f"{r.get('source')}/{r.get('subset')}" for r in by_hash["train"][h] + by_hash["test"][h]}
    schemas = [{"tasks": sorted(u["tasks"]), "train_ids": sorted(u["train_ids"]), "test_ids": sorted(u["test_ids"]),
                "sources": sorted(u["sources"]), "same_id_on_both_sides": bool(u["train_ids"] & u["test_ids"])}
               for u in union.values()]
    schemas.sort(key=lambda x: x["train_ids"])
    return {"shared_schema_contents": len(union), "per_task": per, "schemas": schemas}
