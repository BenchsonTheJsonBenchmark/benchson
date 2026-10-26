"""Create-pipeline complexity pilot (experimental, preview-only).

Question: as schema/reference complexity grows, does Create generation succeed less
often - and is the description/grounding stage the bottleneck?

This runs the CURRENT Create pipeline, unchanged, on an explicit list of schemas and
records what happened to EACH schema on its own. It reuses the builder's own
`_mint_reference` and `_make_description` (same prompts, same retries, same gates,
same grounding threshold); it only adds a thin probe that notes which phase each model
call belongs to. Nothing is written to data/, and no dataset record format changes.

Why not just use `preview_generation.py`? The normal builder keeps trying schemas until
it reaches a target count, so a schema that fails is silently replaced by the next one
and its outcome is lost. Here every selected schema gets exactly one result row (per
repeat), whether it succeeded or failed.

The tier labels (LOW/MEDIUM/HIGH) come from the config. They are sampling labels for
this experiment only, not a Benchson definition of complexity.

Outputs (in --out-dir):
  results.csv        one row per schema (per repeat)
  results.json       the same rows plus per-attempt details (ungrounded strings, failure counts)
  results.md         readable summary table + per-tier totals
  llm_calls.jsonl    every prompt/raw response (call numbers are listed in results.json)
  config_used.json   the config as written (placeholders, no secrets)

Usage:
    uv run python src/preview_create_pilot.py --config configs/preview_create_pilot_watsonx.json
    uv run python src/preview_create_pilot.py --config configs/preview_create_pilot_mock.json
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import csv
import datetime
import json
import time

from config_utils import expand_env, load_env_file
from data_generation.benchmark_builder import BenchmarkBuilder, _grounding_ratio
from evaluations import metrics
from llm.llm_provider import LLMProvider
from llm.recording_provider import CallBudget, CallBudgetExceeded, ProviderFailing, RecordingProvider

_CSV_FIELDS = [
    "tier", "schema", "repeat", "schema_chars",
    "ref_success", "ref_build_attempts", "ref_calls", "ref_leaves", "ref_depth", "ref_json_chars",
    "ref_checked_strings",
    "desc_attempts", "desc_groundings", "best_grounding", "final_grounding", "grounding_threshold",
    "create_accepted", "llm_calls", "call_numbers", "failure_stage", "failure_detail",
]


# ------------------------------------------------------------------ probe

class PhaseProbe(LLMProvider):
    """Sits between the builder and the recording provider.

    Remembers every call of the schema being processed and which phase made it
    ("reference" or "description"), so the outcome can be reconstructed afterwards.
    The builder is unaware of it. Assumes calls are sequential (concurrency 1).
    """

    def __init__(self, inner, budget):
        super().__init__()
        self.inner = inner
        self.budget = budget
        self.model = getattr(inner, "model", None)
        self.phase = None
        self.calls = []

    def reset(self, phase):
        self.phase = phase
        self.calls = []

    def _generate(self, messages, parameters=None):
        entry = {"phase": self.phase, "messages": len(messages), "response": None,
                 "error": None, "call_no": None}
        try:
            text, _ = self.inner.generate(messages, parameters)
        except CallBudgetExceeded:
            raise  # the budget stopped the call before it was made: it is not a call
        except BaseException as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["call_no"] = self.budget.used
            self.calls.append(entry)
            raise
        entry["response"] = text if isinstance(text, str) else ""
        entry["call_no"] = self.budget.used
        self.calls.append(entry)
        return text


# ------------------------------------------------------------------ measurements

def checkable_strings(reference):
    """The strings the grounding gate looks at (mirrors `_grounding_ratio`'s filter)."""
    out = []
    for _, value in metrics._iter_leaves(reference):
        if isinstance(value, str):
            token = value.strip()
            if len(token) >= 4 and any(c.isalpha() for c in token):
                out.append(token)
    return out


def ungrounded_strings(reference, description):
    desc = description.lower()
    return [t for t in checkable_strings(reference) if t.lower() not in desc]


def classify_reference_calls(task, schema, ref_calls):
    """Counts what each reference response looked like. Used only for failed references."""
    counts = {"non_trivial": 0, "provider_error": 0, "truncated": 0, "invalid_json": 0,
              "schema_violation": 0}
    for call in ref_calls:
        if call["error"]:
            counts["provider_error"] += 1
            continue
        cleaned = task._clean_output(call["response"])
        error = task._validate_json(cleaned, schema)
        if error is None:
            counts["non_trivial"] += 1  # valid, so only the non-triviality gate can have rejected it
        elif error.startswith("JSON parse error"):
            counts["truncated" if not cleaned.endswith(("}", "]")) else "invalid_json"] += 1
        else:
            counts["schema_violation"] += 1
    return counts


_REFERENCE_FAILURE_PRIORITY = ("non_trivial", "provider_error", "truncated", "invalid_json",
                               "schema_violation")


def run_one(builder, probe, gates, item, repeat):
    """Runs the Create pipeline for one schema. Returns (row, details)."""
    schema, name, tier = item["schema"], item["name"], item["tier"]
    row = {
        "tier": tier, "schema": name, "repeat": repeat, "schema_chars": len(json.dumps(schema)),
        "ref_success": False, "ref_build_attempts": 0, "ref_calls": 0, "ref_leaves": None,
        "ref_depth": None, "ref_json_chars": None, "ref_checked_strings": None,
        "desc_attempts": 0, "desc_groundings": [], "best_grounding": None, "final_grounding": None,
        "grounding_threshold": gates["grounding_threshold"],
        "create_accepted": False, "llm_calls": 0, "call_numbers": None,
        "failure_stage": None, "failure_detail": None,
    }
    details = {"reference_failure_counts": None, "ungrounded_strings": [], "interrupted": False}

    probe.reset("reference")
    reference = None
    try:
        reference = builder._mint_reference(schema, probe, gates)
        if reference is not None:
            probe.phase = "description"
            row["create_accepted"] = builder._make_description(schema, reference, probe, gates) is not None
    except (CallBudgetExceeded, ProviderFailing) as e:
        details["interrupted"] = True
        row["failure_stage"] = ("call_budget_exhausted" if isinstance(e, CallBudgetExceeded)
                                else "provider_failing")
        row["failure_detail"] = str(e)
    except Exception as e:  # the production builder also skips a schema on a worker error
        row["failure_stage"] = "other_error"
        row["failure_detail"] = f"{type(e).__name__}: {e}"

    ref_calls = [c for c in probe.calls if c["phase"] == "reference"]
    desc_calls = [c for c in probe.calls if c["phase"] == "description"]
    row["llm_calls"] = len(probe.calls)
    row["ref_calls"] = len(ref_calls)
    if ref_calls:
        # A build attempt starts from the initial prompt; self-fix calls have longer histories.
        base = min(c["messages"] for c in ref_calls)
        row["ref_build_attempts"] = sum(1 for c in ref_calls if c["messages"] == base)
    if probe.calls:
        numbers = [c["call_no"] for c in probe.calls]
        row["call_numbers"] = f"{min(numbers)}-{max(numbers)}"

    if reference is not None:
        leaves = list(metrics._iter_leaves(reference))
        row.update(ref_success=True, ref_leaves=len(leaves),
                   ref_depth=max((len(p) for p, _ in leaves), default=0),
                   ref_json_chars=len(json.dumps(reference, ensure_ascii=False)),
                   ref_checked_strings=len(checkable_strings(reference)))
        row["desc_attempts"] = len(desc_calls)
        for call in desc_calls:
            text = (call["response"] or "").strip()
            if call["error"] or not text:
                row["desc_groundings"].append(None)
                details["ungrounded_strings"].append(None)
            else:
                row["desc_groundings"].append(round(_grounding_ratio(reference, text), 4))
                details["ungrounded_strings"].append(ungrounded_strings(reference, text))
        scored = [g for g in row["desc_groundings"] if g is not None]
        row["best_grounding"] = max(scored) if scored else None
        row["final_grounding"] = row["desc_groundings"][-1] if row["desc_groundings"] else None

    if not row["create_accepted"] and row["failure_stage"] is None:
        if reference is None:
            counts = classify_reference_calls(builder.task, schema, ref_calls)
            details["reference_failure_counts"] = counts
            kind = next((k for k in _REFERENCE_FAILURE_PRIORITY if counts[k]), "invalid_json")
            row["failure_stage"] = f"reference_{kind}"
        elif any(c["error"] for c in desc_calls):
            row["failure_stage"] = "description_provider_error"
        elif desc_calls and all(g is None for g in row["desc_groundings"]):
            row["failure_stage"] = "description_empty"
        else:
            row["failure_stage"] = "grounding_failed"
    return row, details


# ------------------------------------------------------------------ reports

def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _groundings_text(row):
    return ", ".join("empty" if g is None else f"{g:.2f}" for g in row["desc_groundings"])


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["desc_groundings"] = ";".join("" if g is None else str(g) for g in row["desc_groundings"])
            writer.writerow(flat)


def build_markdown(meta, rows):
    lines = ["# Create complexity pilot", "",
             f"- Model: `{meta['model']}` | repeats: {meta['repeats']} | "
             f"grounding threshold: {meta['grounding_threshold']}",
             f"- LLM calls used: {meta['calls_used']} (cap {meta['max_llm_calls']}, "
             f"worst case for this run {meta['worst_case_calls']})"]
    if meta["stopped"]:
        lines.append(f"- STOPPED EARLY: {meta['stopped']} (rows marked not_run were never tried; "
                     "they are NOT failures)")
    lines += ["", "| Tier | Schema | Rep | Ref ok | Leaves | Depth | Ref builds/calls | Desc attempts | "
              "Grounding per attempt | Best | Accepted | LLM calls | Failure |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r["tier"], r["schema"], str(r["repeat"]), _fmt(r["ref_success"]), _fmt(r["ref_leaves"]),
            _fmt(r["ref_depth"]), f"{r['ref_build_attempts']}/{r['ref_calls']}", str(r["desc_attempts"]),
            _groundings_text(r), _fmt(r["best_grounding"]), _fmt(r["create_accepted"]),
            str(r["llm_calls"]), r["failure_stage"] or ""]) + " |")

    lines += ["", "## Per tier (rows actually run)", "",
              "| Tier | Runs | Ref ok | Accepted | Mean leaves | Mean best grounding | LLM calls |",
              "|---|---|---|---|---|---|---|"]
    for tier in dict.fromkeys(r["tier"] for r in rows):
        ran = [r for r in rows if r["tier"] == tier and r["failure_stage"] != "not_run"]
        if not ran:
            continue
        leaves = [r["ref_leaves"] for r in ran if r["ref_leaves"] is not None]
        best = [r["best_grounding"] for r in ran if r["best_grounding"] is not None]
        lines.append(f"| {tier} | {len(ran)} | {sum(r['ref_success'] for r in ran)} | "
                     f"{sum(bool(r['create_accepted']) for r in ran)} | "
                     f"{round(sum(leaves) / len(leaves), 1) if leaves else ''} | "
                     f"{round(sum(best) / len(best), 2) if best else ''} | "
                     f"{sum(r['llm_calls'] for r in ran)} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main

def worst_case_calls(gates, max_retries):
    """Most model calls one schema can use in Create-only mode (no round-trip)."""
    attempts = gates["max_build_attempts"]
    return attempts * (max_retries + 1) + attempts


def load_plan(config, builder):
    """The ordered list of schemas to run, loaded through the builder's own loader."""
    tiers = config["experiment"]["tiers"]
    wanted = [(tier, path) for tier, paths in tiers.items() for path in paths]
    records = {rec[3]: rec for rec in builder._load_schema_records()}
    plan = []
    for tier, path in wanted:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in records:
            raise ValueError(f"Schema could not be loaded (missing, unreadable, or not an object): {path}")
        plan.append({"tier": tier, "name": stem, "path": path, "schema": records[stem][0]})
    return plan


def run(args):
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    try:
        config = expand_env(raw_config)
    except EnvironmentError as e:
        print(f"Error: {e}\nSet them in your shell first, for example:\n  export NAME=value")
        return 2
    if "experiment" not in config or not config["experiment"].get("tiers"):
        print("Error: the config needs an 'experiment' block with 'tiers' (tier -> list of schema files).")
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir or os.path.join("outputs", "experiments", "create_complexity", stamp))
    if out_dir.startswith(os.path.join(os.getcwd(), "data") + os.sep):
        print("Error: refusing to write experiment output inside data/.")
        return 2
    os.makedirs(out_dir, exist_ok=True)

    wanted = [p for paths in config["experiment"]["tiers"].values() for p in paths]
    if len(set(wanted)) != len(wanted):
        print("Error: a schema file is listed more than once in 'tiers'.")
        return 2

    cfg = copy.deepcopy(config)
    # The builder's own `files` support selects exactly these schemas (no random sampling).
    cfg["sources"] = [{"name": config["experiment"].get("source_name", "schemas"), "files": wanted}]
    cfg["output_root"] = os.path.join(out_dir, "dataset")  # never data/; nothing is written there anyway
    cfg["concurrency"] = 1
    builder = BenchmarkBuilder(cfg)
    gates = builder.gates_by_split["test"]
    if gates["round_trip"]:
        print("Error: this pilot measures the current preview settings with round_trip=false only "
              "(round-trip calls would be mixed into the description counts).")
        return 2

    try:
        plan = load_plan(cfg, builder)
    except ValueError as e:
        print(f"Error: {e}")
        return 2

    per_schema = worst_case_calls(gates, builder.max_retries)
    worst_case = per_schema * len(plan) * args.repeats
    max_calls = args.max_llm_calls if args.max_llm_calls is not None else worst_case
    log_path = os.path.join(out_dir, "llm_calls.jsonl")
    budget = CallBudget(max_calls)
    probe = PhaseProbe(RecordingProvider(LLMProvider.from_config(cfg["llm_provider"]),
                                         log_path, "builder", budget), budget)
    builder._validator = probe  # unused with round_trip=false; keeps the builder's contract

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(raw_config, f, indent=2)

    print(f"Create pilot -> {out_dir}")
    print(f"  model: {probe.model} | schemas: {len(plan)} x repeats {args.repeats} | "
          f"worst case {worst_case} calls ({per_schema}/schema) | cap {max_calls}")
    if max_calls < worst_case:
        print("  WARNING: the cap is below the worst case; schemas may end as call_budget_exhausted/not_run "
              "(those are not genuine failures).")

    rows, details, stopped = [], [], None
    started = time.time()
    total = len(plan) * args.repeats
    queue = [(r, item) for r in range(1, args.repeats + 1) for item in plan]
    for index, (repeat, item) in enumerate(queue, start=1):
        row, detail = run_one(builder, probe, gates, item, repeat)
        rows.append(row)
        details.append(detail)
        verdict = "ACCEPTED" if row["create_accepted"] else f"failed ({row['failure_stage']})"
        print(f"[{index}/{total}] {item['tier']:<6} {item['name']}: ref "
              f"{'ok' if row['ref_success'] else 'FAILED'} | desc attempts {row['desc_attempts']} "
              f"[{_groundings_text(row)}] | calls {row['llm_calls']} -> {verdict}")
        if detail["interrupted"]:
            stopped = row["failure_detail"]
            print(f"\nStopped early: {stopped}")
            for later_repeat, later in queue[index:]:
                blank = {"tier": later["tier"], "schema": later["name"], "repeat": later_repeat,
                         "schema_chars": len(json.dumps(later["schema"])), "ref_success": False,
                         "ref_build_attempts": 0, "ref_calls": 0, "ref_leaves": None, "ref_depth": None,
                         "ref_json_chars": None, "ref_checked_strings": None, "desc_attempts": 0,
                         "desc_groundings": [], "best_grounding": None, "final_grounding": None,
                         "grounding_threshold": gates["grounding_threshold"], "create_accepted": None,
                         "llm_calls": 0, "call_numbers": None, "failure_stage": "not_run",
                         "failure_detail": "run stopped before this schema was tried"}
                rows.append(blank)
                details.append({"reference_failure_counts": None, "ungrounded_strings": [], "interrupted": False})
            break

    meta = {"model": probe.model, "repeats": args.repeats, "grounding_threshold": gates["grounding_threshold"],
            "max_build_attempts": gates["max_build_attempts"], "max_retries": builder.max_retries,
            "calls_used": budget.used, "max_llm_calls": max_calls, "worst_case_calls": worst_case,
            "stopped": stopped, "seconds": round(time.time() - started, 1)}
    write_csv(os.path.join(out_dir, "results.csv"), rows)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": [{**r, **d} for r, d in zip(rows, details)]},
                  f, indent=2, ensure_ascii=False)
    markdown = build_markdown(meta, rows)
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(markdown)

    print("\n" + markdown)
    print(f"Results in: {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Create-pipeline complexity pilot (preview-only).")
    parser.add_argument("--config", required=True, help="Pilot config (see configs/preview_create_pilot_*.json).")
    parser.add_argument("--out-dir", help="Output folder (default outputs/experiments/create_complexity/<timestamp>).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run every schema this many times (default 1). Generation is stochastic, "
                             "so one run per schema is a noisy sample.")
    parser.add_argument("--max-llm-calls", type=int,
                        help="Hard stop after this many model calls. Default: the computed worst case "
                             "for this run, so the cap can never cut the experiment short.")
    parser.add_argument("--env-file", default=".env",
                        help="File with KEY=VALUE secrets (default .env; ignored by git).")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    loaded = load_env_file(args.env_file)
    if loaded:
        print(f"Loaded from {args.env_file}: {', '.join(loaded)}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
