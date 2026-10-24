"""Small-scale preview of the benchmark build.

Runs the REAL BenchmarkBuilder for a handful of examples, writes into a separate
folder (never into data/), and produces files meant for a human to read:

  review.md       the generated examples, their quality hints, and the raw model outputs
  llm_calls.jsonl every prompt sent and the raw response (before any cleaning/parsing)
  summary.json    the same statistics that are printed to the terminal
  dataset/        the generated instances, in the same layout the full build uses

Usage:
    uv run python src/preview_generation.py --config configs/preview_watsonx_granite.json --count 2
    uv run python src/preview_generation.py --config configs/preview_mock.json --count 3
    uv run python src/preview_generation.py --list-models granite
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import datetime
import json
import time
import urllib.request

from config_utils import expand_env, load_env_file
from data_generation.benchmark_builder import BenchmarkBuilder
from evaluations import metrics
from llm.llm_provider import LLMProvider
from llm.recording_provider import CallBudget, CallBudgetExceeded, ProviderFailing, RecordingProvider

_TASKS = ("create", "fix", "modify")
_DEFAULT_WATSONX_URL = "https://us-south.ml.cloud.ibm.com/"


# ------------------------------------------------------------------ list models

def list_models(needle):
    """Prints the model ids in the public WatsonX catalogue that contain `needle`."""
    base = os.environ.get("WATSONX_URL") or _DEFAULT_WATSONX_URL
    url = base.rstrip("/") + "/ml/v1/foundation_model_specs?version=2025-02-01&limit=200"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                    timeout=30) as response:
            data = json.load(response)
    except OSError as e:
        print(f"Could not reach {url}\n  {e}\nCheck your network / WATSONX_URL, or look up the model id "
              "in the WatsonX project instead.")
        return
    ids = sorted(m.get("model_id", "") for m in data.get("resources", []))
    matches = [i for i in ids if needle.lower() in i.lower()]
    print(f"Endpoint: {base}")
    print(f"{len(ids)} models in the public catalogue; {len(matches)} contain '{needle}':")
    for model_id in matches:
        print(f"  {model_id}")
    print("\nNote: this is the public catalogue. A model deployed on demand inside your own "
          "project may not be listed here - check the project's Deployments page for its id.")


# ------------------------------------------------------------------ run

def build_run_config(config, args, out_dir):
    """Turns a preview config into a builder config: tiny counts, separate output folder."""
    cfg = copy.deepcopy(config)
    # The flat `counts` shape means "test split only, no train/test partition".
    cfg["counts"] = {task: args.count for task in _TASKS}
    cfg["output_root"] = os.path.join(out_dir, "dataset")
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("concurrency", 1)
    validation = cfg.setdefault("validation", {})
    if args.round_trip:
        validation["round_trip"] = True
    else:
        validation.setdefault("round_trip", False)
    return cfg


def load_produced(cfg):
    """Reads back the instances the builder wrote: {task: [(filename, record), ...]}."""
    produced = {}
    for task in _TASKS:
        folder = os.path.join(cfg["output_root"], cfg.get(f"{task}_dataset", f"benchmark_{task}"), "test")
        items = []
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.endswith(".json"):
                    with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                        items.append((name, json.load(f)))
        produced[task] = items
    return produced


def summarize_calls(log_path):
    """Statistics over the raw call log (what the model actually returned)."""
    calls = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            calls = [json.loads(line) for line in f if line.strip()]

    responses = [c["response"] for c in calls if isinstance(c.get("response"), str)]
    seconds = [c["seconds"] for c in calls]

    def expects_json(call):
        return any(m["role"] == "system" and "output only" in m["content"].lower()
                   for m in call["messages"])

    clean = fenced = unparseable = truncated = 0
    expected = [c for c in calls if expects_json(c) and isinstance(c.get("response"), str)]
    for c in expected:
        text = c["response"]
        try:
            json.loads(text.strip())
            clean += 1
            continue
        except json.JSONDecodeError:
            pass
        if metrics.parse_json(text)[0] is not None:
            fenced += 1
        else:
            unparseable += 1
            if not text.rstrip().endswith(("}", "]")):
                truncated += 1

    return {
        "calls": len(calls),
        "errors": sum(1 for c in calls if c.get("error")),
        "avg_seconds": round(sum(seconds) / len(seconds), 2) if seconds else 0,
        "max_seconds": max(seconds) if seconds else 0,
        "avg_response_chars": round(sum(len(r) for r in responses) / len(responses)) if responses else 0,
        "max_response_chars": max((len(r) for r in responses), default=0),
        "with_think_tag": sum(1 for r in responses if "<think" in r.lower()),
        "empty_responses": sum(1 for r in responses if not r.strip()),
        "json_expected_calls": len(expected),
        "json_clean": clean,
        "json_in_code_fence": fenced,
        "json_unparseable": unparseable,
        "json_likely_truncated": truncated,
    }


# ------------------------------------------------------------------ review file

def _fmt_path(path):
    out = ""
    for step in path:
        out += f"[{step}]" if isinstance(step, int) else ("." if out else "") + str(step)
    return out or "(root)"


def _short(value, limit=110):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def _json_block(obj, max_lines=45):
    lines = json.dumps(obj, indent=2, ensure_ascii=False).splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "~~~json\n" + "\n".join(lines) + "\n~~~"


def _count_arrays(obj):
    """Arrays with 2+ items (a sign the object is not a bare minimum)."""
    if isinstance(obj, dict):
        return sum(_count_arrays(v) for v in obj.values())
    if isinstance(obj, list):
        return (1 if len(obj) >= 2 else 0) + sum(_count_arrays(v) for v in obj)
    return 0


def _shape_line(obj):
    leaves = list(metrics._iter_leaves(obj))
    depth = max((len(p) for p, _ in leaves), default=0)
    return f"values: {len(leaves)} | max depth: {depth} | arrays with 2+ items: {_count_arrays(obj)}"


def _change_lines(before, after):
    lines = []
    for path in sorted(metrics.changed_paths(before, after), key=lambda p: [str(s) for s in p]):
        had_b, val_b = metrics._navigate(before, path)
        had_a, val_a = metrics._navigate(after, path)
        lines.append(f"- `{_fmt_path(path)}`: {_short(val_b) if had_b else '(missing)'}"
                     f" -> {_short(val_a) if had_a else '(missing)'}")
    return lines or ["- (no differences)"]


def _example_section(task, index, record):
    tag = "/".join(p for p in (record.get("source"), record.get("subset")) if p)
    out = [f"### {task} #{index} - {record.get('name')} [{tag}]", ""]
    if task == "create":
        out += [_shape_line(record["reference_json"]), "", "**Description given to the model:**", "",
                "> " + record["description"].replace("\n", "\n> "), "",
                "**Expected JSON:**", "", _json_block(record["reference_json"])]
    elif task == "fix":
        out += [_shape_line(record["valid_json"]), "",
                "**Injected error (before -> after the fix):**", ""]
        out += _change_lines(record["erroneous_json"], record["valid_json"])
        out += ["", "**Expected (valid) JSON:**", "", _json_block(record["valid_json"])]
    else:
        spec = record.get("modification", {})
        out += [_shape_line(record["ground_truth"]), "",
                f"**Instruction:** {record['instructions']}", "",
                f"Programmatic edit: `{spec.get('op')}` at `{_fmt_path(spec.get('path', []))}`", "",
                "**Change (before -> after):**", ""]
        out += _change_lines(record["data"], record["ground_truth"])
        out += ["", "**Expected JSON after the change:**", "", _json_block(record["ground_truth"])]
    return out + [""]


def _call_section(call):
    last_user = next((m["content"] for m in reversed(call["messages"]) if m["role"] == "user"), "")
    out = [f"#### call {call['call']} ({call['label']}, {call['seconds']}s)", "",
           "Prompt (end of the last user message):", "", "~~~text", last_user[-600:], "~~~", ""]
    if call.get("error"):
        out += [f"Error: `{call['error']}`", ""]
    else:
        response = call.get("response") or ""
        suffix = f"\n... ({len(response) - 1500} more characters)" if len(response) > 1500 else ""
        out += ["Response:", "", "~~~text", response[:1500] + suffix, "~~~", ""]
    return out


def write_review(path, header, produced, log_path):
    lines = ["# Benchson generation preview", ""] + header + ["", "## Examples", ""]
    for task in _TASKS:
        if not produced[task]:
            lines += [f"(no {task} examples were produced)", ""]
        for i, (_, record) in enumerate(produced[task], start=1):
            lines += _example_section(task, i, record)
    lines += ["## Raw model outputs", "",
              "Exactly what the model returned, before any cleaning. Full text is in llm_calls.jsonl.", ""]
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines += _call_section(json.loads(line))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ main

def run(args):
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    try:
        config = expand_env(raw_config)
    except EnvironmentError as e:
        print(f"Error: {e}\nSet them in your shell first, for example:\n  export NAME=value")
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir or os.path.join("outputs", "preview", stamp))
    if os.path.abspath(out_dir).startswith(os.path.join(os.getcwd(), "data") + os.sep):
        print("Error: refusing to write previews inside data/ (that would mix them with the dataset).")
        return 2
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_run_config(config, args, out_dir)
    log_path = os.path.join(out_dir, "llm_calls.jsonl")
    budget = CallBudget(args.max_llm_calls)

    builder_llm = RecordingProvider(LLMProvider.from_config(cfg["llm_provider"]), log_path, "builder", budget)
    validator_llm = None
    if "validator_provider" in cfg:
        validator_llm = RecordingProvider(LLMProvider.from_config(cfg["validator_provider"]),
                                          log_path, "validator", budget)

    # Save the config as written (placeholders, no secrets) - never the expanded one.
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(raw_config, f, indent=2)

    print(f"Preview run -> {out_dir}")
    print(f"  model: {builder_llm.model} | examples per task: {args.count} | "
          f"max LLM calls: {args.max_llm_calls} | round-trip gate: {cfg['validation']['round_trip']}")

    started = time.time()
    stopped = None
    try:
        BenchmarkBuilder(cfg).build(builder_llm, validator_llm)
    except (CallBudgetExceeded, ProviderFailing) as e:
        stopped = str(e)
        print(f"\nStopped early: {stopped}")
    elapsed = round(time.time() - started, 1)

    produced = load_produced(cfg)
    stats = summarize_calls(log_path)
    counts = {task: len(items) for task, items in produced.items()}

    summary = {"model": builder_llm.model, "requested_per_task": args.count, "produced": counts,
               "stopped_early": stopped, "seconds": elapsed, "llm": stats}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    header = [f"- Model: `{builder_llm.model}`",
              f"- Examples per task: requested {args.count}, produced "
              + ", ".join(f"{t} {counts[t]}" for t in _TASKS),
              f"- LLM calls: {stats['calls']} (errors: {stats['errors']}), "
              f"avg {stats['avg_seconds']}s, max {stats['max_seconds']}s",
              f"- Responses with a think tag: {stats['with_think_tag']}, empty: {stats['empty_responses']}",
              f"- Calls that should return only JSON: {stats['json_expected_calls']} -> "
              f"clean {stats['json_clean']}, inside a code fence {stats['json_in_code_fence']}, "
              f"unparseable {stats['json_unparseable']} (likely cut off: {stats['json_likely_truncated']})"]
    if stopped:
        header.append(f"- STOPPED EARLY: {stopped}")
    review_path = os.path.join(out_dir, "review.md")
    write_review(review_path, header, produced, log_path)

    print("\n=== Preview summary ===")
    print(f"Examples      : " + ", ".join(f"{t} {counts[t]}/{args.count}" for t in _TASKS))
    print(f"LLM calls     : {stats['calls']} (errors {stats['errors']}), avg {stats['avg_seconds']}s, "
          f"max {stats['max_seconds']}s, total run {elapsed}s")
    print(f"Responses     : think tag {stats['with_think_tag']} | empty {stats['empty_responses']} | "
          f"avg {stats['avg_response_chars']} chars, max {stats['max_response_chars']}")
    print(f"JSON replies  : {stats['json_expected_calls']} expected -> clean {stats['json_clean']}, "
          f"code fence {stats['json_in_code_fence']}, unparseable {stats['json_unparseable']} "
          f"(likely cut off {stats['json_likely_truncated']})")
    print(f"Stopped early : {stopped or 'no'}")
    print(f"Review file   : {review_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Small-scale preview of the benchmark build.")
    parser.add_argument("--config", help="Preview config JSON (see configs/preview_*.json).")
    parser.add_argument("--count", type=int, default=2, help="Examples per task (default 2).")
    parser.add_argument("--out-dir", help="Output folder (default outputs/preview/<timestamp>).")
    parser.add_argument("--max-llm-calls", type=int, default=60,
                        help="Hard stop after this many model calls (default 60).")
    parser.add_argument("--seed", type=int, help="Override the schema-sampling seed to see other schemas.")
    parser.add_argument("--round-trip", action="store_true",
                        help="Also run the round-trip check (extra model calls).")
    parser.add_argument("--list-models", nargs="?", const="granite", metavar="TEXT",
                        help="List WatsonX catalogue model ids containing TEXT (default 'granite') and exit.")
    parser.add_argument("--env-file", default=".env",
                        help="File with KEY=VALUE secrets (default .env; ignored by git). "
                             "Variables already exported in the shell take priority.")
    args = parser.parse_args()

    loaded = load_env_file(args.env_file)
    if loaded:
        print(f"Loaded from {args.env_file}: {', '.join(loaded)}")

    if args.list_models is not None:
        list_models(args.list_models)
        return 0
    if not args.config:
        parser.error("--config is required (or use --list-models)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
