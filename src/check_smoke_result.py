"""Decides whether a smoke-test run succeeded. CPU only, no model, read-only.

A smoke run is a SUCCESS only when the output directory holds exactly N valid examples (default 2)
for each of Create, Fix and Modify, and nothing went wrong technically. Otherwise it says which kind
of problem it was, because they need different reactions:

    exit 0  SUCCESS            exactly N valid examples per task, run completed, no technical errors
    exit 2  DATA_SHORTFALL     the run completed cleanly but a task has fewer than N examples
                               (schemas/model did not yield enough; not an infrastructure problem)
    exit 3  TECHNICAL_FAILURE  the run did not complete or the model/server errored: non-zero exit code,
                               manifest not "completed", provider errors, worker errors, call-budget stop,
                               missing report files
    exit 4  INVALID_OUTPUT     examples exist but are malformed/invalid, or there are MORE than N for a
                               task (e.g. a stale output directory was reused)

Precedence when several apply: 3, then 4, then 2.

    python src/check_smoke_result.py --out-dir data_scale_only/smoke_123 \
        --calls-log data_scale_only/smoke_123_calls.jsonl --build-exit-code 0
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

TASKS = ("create", "fix", "modify")
DATASET_DIRS = {"create": "benchmark_create", "fix": "benchmark_fix", "modify": "benchmark_modify"}
REQUIRED = {
    "create": ("name", "source", "schema", "description", "reference_json"),
    "fix": ("name", "source", "schema", "erroneous_json", "valid_json"),
    "modify": ("name", "source", "schema", "data", "instructions", "ground_truth", "modification"),
}
# Failure stages that are infrastructure problems, not "the model did not manage this schema".
TECHNICAL_STAGES = {"worker_error"}
TECHNICAL_EVENT_KINDS = {"provider_error"}

SUCCESS, SHORTFALL, TECHNICAL, INVALID = 0, 2, 3, 4
NAMES = {SUCCESS: "SUCCESS", SHORTFALL: "DATA_SHORTFALL", TECHNICAL: "TECHNICAL_FAILURE", INVALID: "INVALID_OUTPUT"}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report_object(path, technical):
    try:
        result = _load_json(path)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        return result
    except (OSError, ValueError) as exc:
        technical.append(f"{path}: {exc}")
        return {}


def _jsonl(path, technical):
    rows = []
    line_no = 0
    try:
        with open(path, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    rows.append(row)
    except (OSError, ValueError) as exc:
        location = f"{path}:{line_no}" if line_no else path
        technical.append(f"{location}: {exc}")
    return rows


def example_problems(task, record, grounding_threshold=0.9):
    """Reasons an example is not a valid one (empty list = valid) and non-blocking warnings."""
    from evaluations import metrics
    problems, warnings = [], []
    if not isinstance(record, dict):
        return ["not a JSON object"], warnings
    may_be_empty = ("reference_json", "data", "ground_truth", "valid_json", "erroneous_json")   # {} / [] are legal JSON values
    for field in REQUIRED[task]:
        if field not in record or (field not in may_be_empty and record[field] in (None, "", [], {})):
            problems.append(f"missing or empty field '{field}'")
    if problems:
        return problems, warnings
    schema = record["schema"]
    if not isinstance(schema, dict):
        return ["schema is not an object"], warnings

    def valid(obj):
        return metrics.schema_compliance(obj, schema)[0] == 1

    if task == "create":
        if not isinstance(record["description"], str) or not record["description"].strip():
            problems.append("empty description")
        if not valid(record["reference_json"]):
            problems.append("reference_json does not satisfy the schema")
        else:
            from data_generation.benchmark_builder import _grounding_ratio
            ratio = _grounding_ratio(record["reference_json"], record["description"])
            if ratio < grounding_threshold:
                problems.append(f"grounding {ratio:.2f} below threshold {grounding_threshold}")
    elif task == "fix":
        if not valid(record["valid_json"]):
            problems.append("valid_json does not satisfy the schema")
        if valid(record["erroneous_json"]):
            problems.append("erroneous_json is actually valid (nothing to fix)")
        if record["erroneous_json"] == record["valid_json"]:
            problems.append("erroneous_json equals valid_json")
    else:
        if not isinstance(record["instructions"], str) or not record["instructions"].strip():
            problems.append("empty instructions")
        if not valid(record["data"]):
            problems.append("data does not satisfy the schema")
        if record["ground_truth"] == record["data"]:
            problems.append("ground_truth equals data (the edit changed nothing)")
        if not valid(record["ground_truth"]):
            warnings.append("ground_truth does not satisfy the schema (allowed by the original Modify task)")
    return problems, warnings


def latest_report_dir(out_dir):
    candidates = sorted(glob.glob(os.path.join(out_dir, "_generation_reports", "*", "generation_summary.json")))
    return os.path.dirname(candidates[-1]) if candidates else None


def check(out_dir, per_task=2, calls_log=None, build_exit_code=None, grounding_threshold=None, anchor_file=None,
          require_smoke_total=None):
    """Returns a dict: {"code", "verdict", "counts", "technical", "invalid", "shortfall", "warnings"}."""
    technical, invalid, shortfall, warnings = [], [], [], []
    counts = {t: 0 for t in TASKS}
    expected = dict(per_task) if isinstance(per_task, dict) else {t: per_task for t in TASKS}   # int, or per-task quotas
    train_hashes = test_hashes = test_ids = None
    if anchor_file:   # leakage guard: every new schema must be a published TRAIN schema and none a published TEST one
        from data_generation.anchor import AnchorError, load_anchor
        try:
            anchor = load_anchor(anchor_file)
            train_hashes = {e["schema_hash"] for e in anchor["train"]}
            test_hashes = {e["schema_hash"] for e in anchor["test"]}
            test_ids = {e["id"] for e in anchor["test"]}
        except AnchorError as e:
            technical.append(f"anchor file unusable: {e}")

    if build_exit_code not in (None, 0):
        technical.append(f"build_benchmark.py exited with code {build_exit_code}")
    if not os.path.isdir(out_dir):
        technical.append(f"output directory not found: {out_dir}")
        return _result(counts, technical, invalid, shortfall, warnings)

    manifest_path = os.path.join(out_dir, "generation_manifest.json")
    manifest = _report_object(manifest_path, technical)
    if require_smoke_total is not None and manifest.get("smoke_total") != require_smoke_total:
        technical.append(f"manifest smoke_total is {manifest.get('smoke_total')!r}, expected {require_smoke_total}")
    if manifest.get("status") != "completed":
        technical.append(f"run status is {manifest.get('status')!r}, not 'completed'")
    if grounding_threshold is None:
        config = manifest.get("config", {})
        validation = config.get("validation", {}) if isinstance(config, dict) else None
        train = validation.get("train", {}) if isinstance(validation, dict) else None
        threshold = train.get("grounding_threshold", 0.9) if isinstance(train, dict) else None
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            technical.append("invalid grounding threshold in generation_manifest.json")
            threshold = 0.9
        grounding_threshold = threshold

    report_dir = latest_report_dir(out_dir)
    if report_dir is None:
        technical.append("no generation_summary.json under _generation_reports/")
    else:
        summary = _report_object(os.path.join(report_dir, "generation_summary.json"), technical)
        if summary.get("completed") is not True:
            technical.append("generation_summary.json says the run did not complete")
        for fail in _jsonl(os.path.join(report_dir, "generation_failures.jsonl"), technical):
            if fail.get("stage") in TECHNICAL_STAGES:
                technical.append(f"worker error on {fail.get('schema_id')}: {str(fail.get('detail'))[:200]}")
        kinds = {}
        for ev in _jsonl(os.path.join(report_dir, "generation_events.jsonl"), technical):
            kind = ev.get("kind")
            if not isinstance(kind, str):
                technical.append("invalid event kind in generation_events.jsonl")
                continue
            kinds.setdefault(kind, []).append(ev)
        for kind in TECHNICAL_EVENT_KINDS:
            for ev in kinds.get(kind, [])[:5]:
                technical.append(f"{kind} on {ev.get('schema_id')}: {ev.get('detail')}")
        for kind in ("context_overflow",):   # expected capacity exclusion: never sent, so never a failed model call
            if kinds.get(kind):
                warnings.append(f"{len(kinds[kind])} x {kind} (request would not fit the model context window; "
                                "not sent, not an error)")
        for kind in ("empty_response", "suspected_truncation"):
            if kinds.get(kind):
                warnings.append(f"{len(kinds[kind])} x {kind} (model output quality; may explain a shortfall)")

    if calls_log:
        calls = _jsonl(calls_log, technical)
        if any(type(call.get("call")) is not int or call["call"] < 1 or "error" not in call for call in calls):
            technical.append("invalid entry in call log")
        errors = [c for c in calls if c.get("error")]
        if errors:
            technical.append(f"{len(errors)} of {len(calls)} model calls failed; first: {str(errors[0]['error'])[:200]}")
        if not calls:
            technical.append(f"call log is empty or missing: {calls_log} (no model call was made)")

    for task in TASKS:
        folder = os.path.join(out_dir, DATASET_DIRS[task], "train")
        files = sorted(f for f in os.listdir(folder) if f.endswith(".json")) if os.path.isdir(folder) else []
        counts[task] = len(files)
        if len(files) > expected[task]:
            invalid.append(f"{task}: {len(files)} examples, more than the expected {expected[task]} "
                           "(stale or reused output directory?)")
        elif len(files) < expected[task]:
            shortfall.append(f"{task}: {len(files)} of {expected[task]} examples")
        for name in files:
            try:
                record = _load_json(os.path.join(folder, name))
            except (OSError, ValueError) as e:
                invalid.append(f"{task}/{name}: unreadable ({e})")
                continue
            problems, warns = example_problems(task, record, grounding_threshold)
            if train_hashes is not None and isinstance(record, dict) and isinstance(record.get("schema"), dict):
                from data_generation.published import schema_hash
                h = schema_hash(record["schema"])
                if h not in train_hashes:
                    problems.append("schema is not a published train schema")
                if h in test_hashes or name[:-5].rpartition("__a")[0] in test_ids:
                    problems.append("schema overlaps the published test set (leakage)")
            invalid.extend(f"{task}/{name}: {p}" for p in problems)
            warnings.extend(f"{task}/{name}: {w}" for w in warns)
    return _result(counts, technical, invalid, shortfall, warnings)


def _result(counts, technical, invalid, shortfall, warnings):
    code = TECHNICAL if technical else INVALID if invalid else SHORTFALL if shortfall else SUCCESS
    return {"code": code, "verdict": NAMES[code], "counts": counts, "technical": technical,
            "invalid": invalid, "shortfall": shortfall, "warnings": warnings}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--calls-log", default=None)
    ap.add_argument("--build-exit-code", type=int, default=None)
    ap.add_argument("--anchor-file", default=None, help="published_train_schemas.json: also check that no example leaks the test set")
    ap.add_argument("--label", default="SMOKE", help="name shown in the result line (e.g. PILOT)")
    ap.add_argument("--require-smoke-total", type=int, default=None, help="manifest must record exactly this many requested examples")
    args = ap.parse_args(argv)
    res = check(args.out_dir, args.per_task, args.calls_log, args.build_exit_code, anchor_file=args.anchor_file,
                require_smoke_total=args.require_smoke_total)
    print(f"{args.label} RESULT: {res['verdict']}  (examples per task: {res['counts']}, expected {args.per_task} each)")
    for label in ("technical", "invalid", "shortfall", "warnings"):
        for item in res[label]:
            print(f"  [{label}] {item}")
    return res["code"]


if __name__ == "__main__":
    sys.exit(main())
