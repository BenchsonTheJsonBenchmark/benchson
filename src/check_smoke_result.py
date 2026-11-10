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
TECHNICAL_STAGES = {"worker_error", "token_count_error"}
TECHNICAL_EVENT_KINDS = {"provider_error", "token_count_error"}
# A /tokenize request that timed out even after its bounded retries (no model call was made, the attempt is a counted
# failure and wrote nothing). It is tolerated only in small, fully accounted numbers; anything beyond is technical.
TOKEN_COUNT_TIMEOUT = "token_count_timeout"
MAX_TOLERATED_TOKEN_COUNT_TIMEOUTS = 10          # per chunk
MAX_TOLERATED_TOKEN_COUNT_TIMEOUT_RATE = 0.01    # of all token counts (successful + timed out)

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


def _historical_split(failures, events, spec, technical):
    """One-time historical exception (see chunked_run.HISTORICAL_POLICY). Returns the failures/events that remain.

    Removes ONLY: provider_error events whose schema_id is one of the exact ids and whose detail is exactly
    spec["event_detail"], and worker_error failures for those ids whose detail is exactly spec["failure_detail"].
    Then requires exactly one event per id and exactly one failure per id and task (so 3 events, 9 failures).
    If that does not hold, NOTHING is excused and a technical problem is added (fail closed).
    """
    ids = set(spec["ids"])
    hist_events = [e for e in events if e.get("kind") == "provider_error" and e.get("schema_id") in ids
                   and e.get("detail") == spec["event_detail"]]
    hist_fails = [f for f in failures if f.get("stage") == "worker_error" and f.get("schema_id") in ids
                  and f.get("detail") == spec["failure_detail"]]
    ev_ids = sorted(e["schema_id"] for e in hist_events)
    ok = ev_ids == sorted(ids) and len(hist_fails) == len(ids) * len(TASKS)
    ok = ok and all(sorted(f.get("task") for f in hist_fails if f["schema_id"] == i) == sorted(TASKS) for i in ids)
    if not ok:
        technical.append("historical tolerance refused: expected exactly one provider_error event and one worker_error "
                         f"failure per task for each of {sorted(ids)}; found events {ev_ids}, {len(hist_fails)} failures")
        return failures, events, None
    drop_e, drop_f = {id(e) for e in hist_events}, {id(f) for f in hist_fails}
    return ([f for f in failures if id(f) not in drop_f], [e for e in events if id(e) not in drop_e],
            {"events": hist_events, "failures": hist_fails})


def _token_count_timeouts(out_dir, failures, events, manifest, technical, warnings):
    """Tolerates a few fully accounted token-count timeouts; returns the tolerated summary (or None)."""
    ev = [e for e in events if e.get("kind") == TOKEN_COUNT_TIMEOUT]
    fl = [f for f in failures if f.get("stage") == TOKEN_COUNT_TIMEOUT]
    if not ev and not fl:
        return None
    problems = []
    ev_ids = sorted(e.get("schema_id") for e in ev)
    fl_ids = sorted({f.get("schema_id") for f in fl})
    if sorted(set(ev_ids)) != fl_ids or len(set(ev_ids)) != len(ev_ids):
        problems.append(f"token-count timeout events {ev_ids} do not match failures {fl_ids} (one event per attempt expected)")
    guard = manifest.get("context_guard") if isinstance(manifest, dict) else None
    guard = guard if isinstance(guard, dict) else {}
    checked, counted = guard.get("checked"), guard.get("count_timeouts")
    if not (isinstance(checked, int) and isinstance(counted, int)):
        problems.append("manifest has no context_guard counters to account for the token-count timeouts")
    else:
        if counted != len(ev):
            problems.append(f"manifest counts {counted} token-count timeouts but the event log has {len(ev)}")
        if len(ev) > MAX_TOLERATED_TOKEN_COUNT_TIMEOUTS:
            problems.append(f"{len(ev)} token-count timeouts exceed the limit of {MAX_TOLERATED_TOKEN_COUNT_TIMEOUTS}")
        if len(ev) / max(1, checked + len(ev)) > MAX_TOLERATED_TOKEN_COUNT_TIMEOUT_RATE:
            problems.append(f"token-count timeout rate {len(ev)}/{checked + len(ev)} exceeds "
                            f"{MAX_TOLERATED_TOKEN_COUNT_TIMEOUT_RATE:.0%}")
    for sid in fl_ids:   # a failed attempt must not have left an example or a metadata sidecar
        for task in TASKS:
            for rel in (os.path.join(DATASET_DIRS[task], "train", f"{sid}.json"),
                        os.path.join(DATASET_DIRS[task], "_metadata", "train", f"{sid}.json")):
                if os.path.exists(os.path.join(out_dir, rel)):
                    problems.append(f"{rel} exists although attempt {sid} failed")
    if problems:
        technical.extend(f"token-count timeout not tolerated: {p}" for p in problems)
        return None
    warnings.append(f"{len(ev)} x token_count_timeout (the /tokenize request timed out after bounded retries; no model "
                    "call was made; counted as a failed attempt; tolerated within the limit)")
    return {"count": len(ev), "attempt_ids": fl_ids, "limit": MAX_TOLERATED_TOKEN_COUNT_TIMEOUTS,
            "rate_limit": MAX_TOLERATED_TOKEN_COUNT_TIMEOUT_RATE}


def check(out_dir, per_task=2, calls_log=None, build_exit_code=None, grounding_threshold=None, anchor_file=None,
          require_smoke_total=None, tolerate_historical=None):
    """Returns a dict: {"code", "verdict", "counts", "technical", "invalid", "shortfall", "warnings", "tolerated"}.

    `tolerate_historical` is None (strict, the default) or the one-time spec of chunked_run.HISTORICAL_POLICY.
    """
    technical, invalid, shortfall, warnings = [], [], [], []
    tolerated = {}
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
        failures_all = _jsonl(os.path.join(report_dir, "generation_failures.jsonl"), technical)
        events_all = []
        for ev in _jsonl(os.path.join(report_dir, "generation_events.jsonl"), technical):
            if not isinstance(ev.get("kind"), str):
                technical.append("invalid event kind in generation_events.jsonl")
                continue
            events_all.append(ev)
        if tolerate_historical:
            failures_all, events_all, hist = _historical_split(failures_all, events_all, tolerate_historical, technical)
            if hist:
                tolerated["historical"] = {"policy": tolerate_historical.get("policy"), "ids": sorted(tolerate_historical["ids"]),
                                           "events": len(hist["events"]), "failures": len(hist["failures"])}
        tc = _token_count_timeouts(out_dir, failures_all, events_all, manifest, technical, warnings)
        if tc:
            tolerated["token_count_timeouts"] = tc
        for fail in failures_all:
            if fail.get("stage") in TECHNICAL_STAGES:
                technical.append(f"worker error on {fail.get('schema_id')}: {str(fail.get('detail'))[:200]}")
            elif fail.get("stage") == TOKEN_COUNT_TIMEOUT and not tc:
                technical.append(f"token-count timeout on {fail.get('schema_id')} not tolerated")
        kinds = {}
        for ev in events_all:
            kinds.setdefault(ev["kind"], []).append(ev)
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
    return _result(counts, technical, invalid, shortfall, warnings, tolerated)


def _result(counts, technical, invalid, shortfall, warnings, tolerated=None):
    code = TECHNICAL if technical else INVALID if invalid else SHORTFALL if shortfall else SUCCESS
    return {"code": code, "verdict": NAMES[code], "counts": counts, "technical": technical,
            "invalid": invalid, "shortfall": shortfall, "warnings": warnings, "tolerated": tolerated or {}}


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
