"""Builds the description-grounded JSON benchmark (tasks 2a + 2b).

Uses a strong "builder" LLM to author ground-truth reference objects, natural
language descriptions, and injected-error variants from a schema corpus.

Usage:
    uv run python src/build_benchmark.py --config configs/build_benchmark.json

Scale-only experiment extras (all optional; without them the behaviour is unchanged):
    --check-pool               count usable schemas and exit (no model, no files written)
    --smoke-total N            same pipeline/config, but only N examples spread over the tasks
    --out-dir DIR              write somewhere other than the config's output_root
    --max-llm-calls N          hard stop after N model calls
    --record-calls FILE        log every model call (prompt + answer) to FILE
    --env-file .env            load secrets for ${VAR} placeholders in the config
A config with a top-level "scale_only": true is additionally checked for safety
(standard-only, original Modify, published schemas excluded, separate output dir).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import copy
import datetime
import hashlib
import json
import platform
import subprocess

from config_utils import expand_env, load_env_file
from llm.llm_provider import LLMProvider
from data_generation.benchmark_builder import BenchmarkBuilder
from data_generation.generation_mix import split_total

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def validate_scale_only(config):
    """Returns a list of problems that make a config unsafe for the scale-only experiment."""
    problems = []
    gen = config.get("generation")
    if not isinstance(gen, dict):
        problems.append("missing 'generation' block (without it Modify uses the newer weighted strategy, "
                        "not the original one)")
    else:
        if float(gen.get("complex_ratio", 0)) != 0:
            problems.append("generation.complex_ratio must be 0 (100% standard)")
        if float(gen.get("standard_ratio", 1)) != 1:
            problems.append("generation.standard_ratio must be 1")
        if gen.get("standard_modify_strategy") != "original":
            problems.append("generation.standard_modify_strategy must be 'original'")
    excl = (config.get("exclusions") or {}).get("file")
    if not excl:
        problems.append("exclusions.file (reserved_schemas.json from the importer) is required")
    elif not os.path.isfile(excl):
        problems.append(f"exclusions.file not found: {excl}")
    out = config.get("output_root")
    if not out:
        problems.append("output_root is required (never write into data/ or the baseline)")
    else:
        out_abs = os.path.abspath(out)
        if os.path.abspath(os.path.join(os.getcwd(), "data")) == out_abs:
            problems.append("output_root must not be data/")
        if excl and os.path.isfile(excl):
            base = os.path.dirname(os.path.abspath(excl))
            if out_abs == base or out_abs.startswith(base + os.sep):
                problems.append("output_root is inside the published baseline directory")
    counts = config.get("counts", {})
    if "test" in counts:
        problems.append("counts.test must be absent (the published test set is not regenerated)")
    if config.get("reuse"):
        # Controlled schema reuse works on the ORIGINAL partition (same seed/ratio as the published build).
        if config.get("split", {}).get("test_ratio", 0.15) != 0.15:
            problems.append("split.test_ratio must stay 0.15 (the original partition defines the training pool)")
        if config.get("seed", 0) != 0:
            problems.append("seed must be 0 (the original partition seed)")
        for src in config.get("sources", []):
            if src.get("max_schema_kb", 10) != 10:
                problems.append("max_schema_kb must stay 10")
        if not config.get("reuse", {}).get("max_new_references_per_schema"):
            problems.append("reuse.max_new_references_per_schema is required (per-schema cap)")
    else:
        problems.append("missing 'reuse' block (the scale-only experiment samples several examples per schema)")
    return problems


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, timeout=20).stdout.strip()
    except Exception:
        return None


def _write_manifest(path, info):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)


def _redacted(config):
    """The raw config; it holds ${PLACEHOLDERS} only, never the expanded secrets."""
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the JSON extraction/fix benchmark.")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark build config JSON.")
    parser.add_argument("--env-file", default=None, help="KEY=VALUE secrets file for ${VAR} placeholders.")
    parser.add_argument("--check-pool", action="store_true", help="Report schema supply and exit (no model).")
    parser.add_argument("--assumed-acceptance", type=float, default=0.5,
                        help="Used by --check-pool: fraction of schemas assumed to yield an accepted example.")
    parser.add_argument("--smoke-total", type=int, default=None, help="Build only N examples in total.")
    parser.add_argument("--out-dir", default=None, help="Override output_root.")
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--record-calls", default=None)
    args = parser.parse_args(argv)

    if args.env_file:
        loaded = load_env_file(args.env_file)
        if loaded:
            print(f"Loaded from {args.env_file}: {', '.join(loaded)}")

    config_bytes = open(args.config, "rb").read()
    raw = json.loads(config_bytes.decode("utf-8"))
    config = copy.deepcopy(raw)

    # Paths may use ${VAR}; expand them first so the safety check and --check-pool see real paths
    # (the model section is expanded later, so --check-pool needs no model secrets).
    for key in ("exclusions", "output_root"):
        if key in config:
            config[key] = expand_env(config[key])
    if args.out_dir:
        config["output_root"] = args.out_dir
    if args.smoke_total is not None:
        counts = config.get("counts", {})
        tasks = list((counts.get("train") or counts or {"create": 0, "fix": 0, "modify": 0}).keys())
        config["counts"] = {"train": split_total(args.smoke_total, tasks)}
        if not args.out_dir and config.get("output_root"):
            config["output_root"] = config["output_root"].rstrip("/") + "_smoke"

    if raw.get("scale_only"):
        problems = validate_scale_only(config)
        if problems:
            sys.exit("scale_only safety check failed:\n  - " + "\n  - ".join(problems))
        print("[scale_only] safety checks passed (standard-only, original Modify, exclusions, reuse settings, separate output).")

    if args.check_pool:
        probe = {k: v for k, v in config.items() if k not in ("llm_provider", "validator_provider")}
        builder = BenchmarkBuilder(probe)
        if builder.reuse:
            report = builder.reuse_capacity_report(args.assumed_acceptance)
        else:
            report = builder.pool_report()
            need = sum(sum(c.values()) for c in report["targets_per_task"].values())
            per_task = max((max(c.values()) for c in report["targets_per_task"].values() if c), default=0)
            report["required_schemas_if_all_accepted"] = per_task
            report["required_schemas_at_assumed_acceptance"] = int(per_task / max(args.assumed_acceptance, 0.01))
            usable = report["usable_schemas"]
            report["verdict"] = ("OK" if usable >= report["required_schemas_at_assumed_acceptance"]
                                 else "TIGHT" if usable >= per_task else "NOT ENOUGH SCHEMAS")
            report["total_examples_requested"] = need
        print(json.dumps(report, indent=2, default=str))
        return report

    config = expand_env(config)
    llm_provider = LLMProvider.from_config(config["llm_provider"])
    validator_provider = None
    if "validator_provider" in config:
        validator_provider = LLMProvider.from_config(config["validator_provider"])

    budget = None
    if args.max_llm_calls is not None or args.record_calls:
        from llm.recording_provider import CallBudget, RecordingProvider
        budget = CallBudget(args.max_llm_calls)
        log_path = args.record_calls or os.devnull
        llm_provider = RecordingProvider(llm_provider, log_path, "builder", budget)
        if validator_provider is not None:
            validator_provider = RecordingProvider(validator_provider, log_path, "validator", budget)

    builder = BenchmarkBuilder(config)
    out_root = builder.data_root
    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "generation_manifest.json")
    model_params = (config.get("llm_provider", {}).get("params") or {})
    manifest = {
        "status": "running",
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config_file": os.path.abspath(args.config),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config": _redacted(raw),
        "smoke_total": args.smoke_total,
        "seed": builder.seed,
        "model": model_params.get("model"),
        "model_params": model_params.get("model_params"),
        "counts": builder.counts_by_split,
        "generation_mode": {"standard_ratio": str(builder.settings.standard_ratio),
                            "complex_ratio": str(builder.settings.complex_ratio),
                            "standard_modify_strategy": builder.settings.standard_modify_strategy,
                            "enabled": builder.settings.enabled},
        "published_baseline": builder.reserved_info,
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty_files": (_git("status", "--porcelain") or "").splitlines()},
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "max_llm_calls": args.max_llm_calls,
        "schema_reuse": builder.reuse.as_dict() if builder.reuse else None,
        "policy_note": ("Changes the original one-example-per-schema sampling policy (controlled schema reuse); "
                        "keeps the original generation logic, prompts, thresholds and schema pool. The generator "
                        "model differs from the original (gpt-oss-120b)." if builder.reuse else None),
    }
    _write_manifest(manifest_path, manifest)
    try:
        builder.build(llm_provider, validator_provider)
        manifest["status"] = "completed"
    except BaseException as e:
        manifest["status"] = f"stopped: {type(e).__name__}"
        raise
    finally:
        manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        manifest["llm_calls_used"] = budget.used if budget else None
        manifest["run_summary"] = builder.last_summary
        _write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
