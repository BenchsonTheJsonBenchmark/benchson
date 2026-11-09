"""Pilot metrics report: reads only what the run actually recorded.

Every number is tagged "measured" (read from a recorded file), "derived" (simple arithmetic on measured
numbers) or {"status": "unavailable", "reason": ...}. Nothing is estimated and nothing here changes how data is
generated or accepted.

    python src/pilot_report.py --out-dir OUT --calls-log CALLS.jsonl --per-task 20 \
        --build-exit-code 0 --gen-start-epoch T0 --gen-end-epoch T1 --max-llm-calls 500 --write REPORT.json
"""
import argparse
import datetime
import glob
import json
import os
import sys

TASKS = ("create", "fix", "modify")


def unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jsonl(path):
    rows, bad = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    bad += 1
    except OSError:
        return None, 0
    return rows, bad


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _parse_time(text):
    try:
        return datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def build_report(out_dir, calls_log=None, per_task=20, build_exit_code=None, gen_start=None, gen_end=None,
                 max_llm_calls=None, check_exit_code=None):
    notes = []
    targets = dict(per_task) if isinstance(per_task, dict) else {t: per_task for t in TASKS}   # int or per-task quotas
    manifest = _json(os.path.join(out_dir, "generation_manifest.json")) or {}
    summaries = sorted(glob.glob(os.path.join(out_dir, "_generation_reports", "*", "generation_summary.json")))
    report_dir = os.path.dirname(summaries[-1]) if summaries else None
    summary = _json(summaries[-1]) if summaries else None
    if summary is None:
        notes.append("generation_summary.json missing or unreadable: attempts, failures and reasons are unavailable")
        summary = {}

    # ---- accepted examples: counted from the files on disk (what the checker also counts)
    accepted = {}
    for task in TASKS:
        folder = os.path.join(out_dir, f"benchmark_{task}", "train")
        accepted[task] = len([f for f in os.listdir(folder) if f.endswith(".json")]) if os.path.isdir(folder) else 0

    failed_table = (summary.get("failed_attempts") or {}).get("train")
    per_task_rows = {}
    for task in TASKS:
        row = {"target": targets[task], "accepted": {"value": accepted[task], "source": "measured (files on disk)"},
               "reached_target": accepted[task] == targets[task]}
        if failed_table is not None and task in failed_table:
            failed = sum(failed_table[task].values())
            attempts = accepted[task] + failed
            row["failed_attempts"] = {"value": failed, "source": "measured (generation_summary.failed_attempts)"}
            row["attempts"] = {"value": attempts, "source": "derived: accepted + failed"}
            row["acceptance_rate"] = ({"value": round(accepted[task] / attempts, 4), "source": "derived: accepted / attempts"}
                                      if attempts else unavailable("no attempts recorded"))
        else:
            for key in ("failed_attempts", "attempts", "acceptance_rate"):
                row[key] = unavailable("generation_summary.json has no failed_attempts for this task")
        row["llm_calls"] = unavailable("calls are not tagged by task: one reference and description call set is shared "
                                       "by Create, Fix and Modify of the same schema attempt")
        per_task_rows[task] = row

    # ---- model calls (logical calls: SDK retries are disabled in the pilot, so 1 recorded call = 1 request)
    calls = None
    if calls_log:
        rows, bad = _jsonl(calls_log)
        if rows is None:
            notes.append(f"call log not found: {calls_log}")
        else:
            calls = rows
            if bad:
                notes.append(f"{bad} unreadable line(s) in the call log were ignored")
    if calls is None:
        llm = {"total_calls": unavailable("no call log")}
    else:
        secs = [c["seconds"] for c in calls if isinstance(c.get("seconds"), (int, float))]
        errors = [c for c in calls if c.get("error")]
        by_label = {}
        for c in calls:
            by_label[c.get("label")] = by_label.get(c.get("label"), 0) + 1
        llm = {"total_calls": {"value": len(calls), "source": "measured (call log)"},
               "by_label": by_label,
               "failed_calls": {"value": len(errors), "source": "measured (call log)"},
               "call_seconds": ({"sum": round(sum(secs), 1), "mean": round(sum(secs) / len(secs), 2),
                                 "p50": _percentile(secs, .5), "p95": _percentile(secs, .95), "max": max(secs),
                                 "source": "measured (call log)"} if secs else unavailable("no call durations")),
               "budget": {"max_llm_calls": max_llm_calls, "used": len(calls),
                          "exhausted": bool(max_llm_calls is not None and len(calls) >= max_llm_calls)}}
        total_accepted = sum(accepted.values())
        llm["calls_per_accepted_example"] = ({"value": round(len(calls) / total_accepted, 2), "source": "derived"}
                                             if total_accepted else unavailable("no accepted examples"))
        if errors:
            llm["first_error"] = str(errors[0].get("error"))[:300]
        stamps = [_parse_time(c.get("time")) for c in calls]
        stamps = [t for t in stamps if t]
        llm["first_to_last_call_seconds"] = ({"value": round((max(stamps) - min(stamps)).total_seconds(), 1),
                                              "source": "measured (call log timestamps)"} if len(stamps) > 1
                                             else unavailable("fewer than two timestamps"))
    builder_calls = summary.get("builder_llm_calls")
    llm["builder_calls_in_generation_summary"] = ({"value": builder_calls, "source": "measured (generation_summary)"}
                                                  if builder_calls is not None else unavailable("no summary"))

    # ---- duration: generation only (model download and vLLM startup are outside this window)
    if gen_start is not None and gen_end is not None and gen_end >= gen_start:
        duration = {"generation_seconds": {"value": gen_end - gen_start,
                                           "source": "measured (wall clock around build_benchmark.py only)"},
                    "excludes": "model download, vLLM startup, preflight"}
    else:
        duration = {"generation_seconds": unavailable("start/end timestamps were not supplied")}
    started, finished = _parse_time(manifest.get("started")), _parse_time(manifest.get("finished"))
    if started and finished:
        duration["manifest_seconds"] = {"value": round((finished - started).total_seconds(), 1),
                                        "source": "measured (generation_manifest started/finished)"}

    failure_reasons = summary.get("failure_reasons")
    events = summary.get("llm_events")
    reuse = summary.get("schema_reuse") or {}
    report = {
        "run": {"out_dir": out_dir, "manifest_status": manifest.get("status"), "slurm_job_id": manifest.get("slurm_job_id"),
                "model": manifest.get("model"), "smoke_total": manifest.get("smoke_total"),
                "build_exit_code": build_exit_code, "check_exit_code": check_exit_code,
                "pool_source": (reuse.get("settings") or {}).get("pool_source")},
        "tasks": per_task_rows,
        "all_tasks_reached_target": all(r["reached_target"] for r in per_task_rows.values()),
        "llm": llm,
        "duration": duration,
        "failure_reasons": ({"value": failure_reasons, "source": "measured (generation_summary, task/mode/stage)"}
                            if failure_reasons is not None else unavailable("no summary")),
        "llm_events": ({"value": events, "source": "measured (generation_summary)"} if events is not None
                       else unavailable("no summary")),
        "schema_attempts": ({"attempts_total": reuse.get("attempts_total"), "failed_attempts_total": reuse.get("failed_attempts_total"),
                             "schemas_used": reuse.get("schemas_used"), "schema_status": reuse.get("schema_status"),
                             "source": "measured (generation_summary.schema_reuse); one schema attempt can feed all three tasks"}
                            if reuse else unavailable("no schema_reuse section")),
        "notes": notes,
    }
    return report


def render(report):
    def val(x):
        return x["value"] if isinstance(x, dict) and "value" in x else ("n/a" if isinstance(x, dict) and x.get("status") else x)

    lines = ["=== PILOT REPORT ==="]
    for task, row in report["tasks"].items():
        lines.append(f"{task:7s} accepted {val(row['accepted'])}/{row['target']}  attempts {val(row['attempts'])}  "
                     f"failed {val(row['failed_attempts'])}  acceptance {val(row['acceptance_rate'])}  "
                     f"target reached: {row['reached_target']}")
    llm = report["llm"]
    lines.append(f"LLM calls: {val(llm['total_calls'])}  failed calls: {val(llm.get('failed_calls', 'n/a'))}  "
                 f"per accepted example: {val(llm.get('calls_per_accepted_example', 'n/a'))}")
    lines.append(f"generation seconds (excl. download/startup): {val(report['duration']['generation_seconds'])}")
    fr = report["failure_reasons"]
    lines.append("failure reasons: " + (json.dumps(fr["value"], sort_keys=True) if "value" in fr else "n/a"))
    for note in report["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calls-log", default=None)
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--targets", default=None, help="per-task quotas, e.g. create=166,fix=166,modify=166 (chunk runs)")
    ap.add_argument("--build-exit-code", type=int, default=None)
    ap.add_argument("--check-exit-code", type=int, default=None)
    ap.add_argument("--gen-start-epoch", type=int, default=None)
    ap.add_argument("--gen-end-epoch", type=int, default=None)
    ap.add_argument("--max-llm-calls", type=int, default=None)
    ap.add_argument("--write", default=None, help="write the JSON report here (never overwrites an existing file)")
    args = ap.parse_args(argv)
    per_task = ({k: int(v) for k, v in (i.split("=") for i in args.targets.split(","))} if args.targets else args.per_task)
    report = build_report(args.out_dir, args.calls_log, per_task, args.build_exit_code, args.gen_start_epoch,
                          args.gen_end_epoch, args.max_llm_calls, args.check_exit_code)
    if args.write:
        try:
            with open(args.write, "x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except OSError as e:
            print(f"cannot write report: {e}", file=sys.stderr)
            return 1
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
