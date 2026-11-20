"""Shared helpers for the controlled 3K vs 7K preparation (CPU-only; stdlib + unmodified repo code).

Nothing here modifies `original_training/`, the datasets, or any Test record.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import types

TASKS = ("create", "fix", "modify")
GOLD = {"create": "reference_json", "fix": "valid_json", "modify": "ground_truth"}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE_ROOT = os.path.join(REPO, "data_published", "7962b0947f7d", "original_files")
RELEASE_ROOT = os.path.join(REPO, "release", "scale-only-7k")
PREPARE_SCRIPT = os.path.join(REPO, "original_training", "scripts", "prepare_benchson_data.py")
TRAIN_LORA = os.path.join(REPO, "original_training", "scripts", "train_lora.py")
TRAIN_ALORA = os.path.join(REPO, "original_training", "scripts", "train_alora.py")


def add_repo_paths():
    for p in (os.path.join(REPO, "src"), REPO):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path):
    """Raw JSONL lines as bytes, without the trailing newline."""
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def write_lines(path, lines):
    """Writes raw lines (bytes), each followed by one newline. Refuses to overwrite."""
    with open(path, "xb") as f:
        for line in lines:
            f.write(line + b"\n")


def schema_content_hash(schema_obj):
    """Canonical schema-content hash: the repo's own `published.schema_hash` (sorted keys, compact, sha256)."""
    add_repo_paths()
    from data_generation.published import schema_hash
    return schema_hash(schema_obj)


def example_id(row):
    add_repo_paths()
    from data_generation.published import record_id
    return record_id(row)


def tier_of(row):
    return row.get("subset") or row.get("source")


def run_original_prepare(dataset_root, out_path, revision="LOCAL"):
    """Runs the ORIGINAL, unmodified `prepare_benchson_data.py` main() against local JSONL files.

    The only substitution is the `datasets` package (not installed here): a stub `load_dataset(path, name, split, revision)`
    returns the rows of `<dataset_root>/<name>/<split>.jsonl` in file order, which is how the Hugging Face loader yields
    them for these JSONL-backed configs. Returns the script's stdout.
    """
    add_repo_paths()
    stub = types.ModuleType("datasets")

    def load_dataset(path, name=None, split=None, revision=None, **kwargs):
        return [json.loads(line) for line in read_lines(os.path.join(dataset_root, name, f"{split}.jsonl"))]

    stub.load_dataset = load_dataset
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = stub
    argv = sys.argv
    buf = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location("original_prepare_benchson_data", PREPARE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.argv = ["prepare_benchson_data.py", "--dataset", "LOCAL", "--revision", revision, "--output", out_path]
        with contextlib.redirect_stdout(buf):
            mod.main()
    finally:
        sys.argv = argv
        if saved is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = saved
    return buf.getvalue()
