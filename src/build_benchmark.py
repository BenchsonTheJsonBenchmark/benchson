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
        reuse_cfg = config.get("reuse", {})
        if reuse_cfg.get("pool_source") == "published_train":
            # Pool = schemas of the published training rows. The size restriction is an explicit experimental choice
            # (kept at 10 KB), not a claim about how the Hugging Face snapshot was generated.
            if reuse_cfg.get("anchor_max_schema_kb") != 10:
                problems.append("reuse.anchor_max_schema_kb must be 10 (documented experimental restriction)")
            anchor = reuse_cfg.get("anchor_file") or (os.path.join(os.path.dirname(os.path.abspath(excl)), "published_train_schemas.json")
                                                     if excl else None)
            if anchor and not os.path.isfile(anchor):
                problems.append(f"anchor file not found: {anchor} (python src/export_anchor_pool.py --baseline-dir <baseline>)")
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
    parser.add_argument("--smoke-openai-timeout", type=int,
                        help="For an OpenAI smoke run, set the SDK timeout in seconds and disable SDK retries.")
    parser.add_argument("--out-dir", default=None, help="Override output_root.")
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--record-calls", default=None)
    parser.add_argument("--context-guard", type=int, default=None, metavar="MAX_MODEL_LEN",
                        help="Refuse (without sending) any request with prompt_tokens + max_tokens > MAX_MODEL_LEN, "
                             "counted by the vLLM server's own tokenizer. MAX_MODEL_LEN must equal the server's "
                             "--max-model-len. Nothing else about a request changes.")
    parser.add_argument("--chunks", type=int, default=None, metavar="N",
                        help="Full run in N chunks: this job builds chunk --chunk-index, whose per-task quota is "
                             "floor(total*k/N)-floor(total*(k-1)/N) of the config counts (exact sums). Needs --out-dir.")
    parser.add_argument("--chunk-index", type=int, default=None, metavar="K")
    parser.add_argument("--parent-chunk", default=None,
                        help="Directory of the previous, validated chunk (required for K > 1; never inferred).")
    args = parser.parse_args(argv)

    if args.chunks is not None:
        if args.smoke_total is not None or not args.out_dir or args.chunk_index is None:
            parser.error("--chunks needs --chunk-index and --out-dir, and cannot be combined with --smoke-total")
        if args.chunk_index > 1 and not args.parent_chunk:
            parser.error("--parent-chunk is required for --chunk-index > 1 (explicit, validated recovery; never automatic)")
        if args.chunk_index == 1 and args.parent_chunk:
            parser.error("chunk 1 has no parent")
        if os.path.lexists(args.out_dir):
            parser.error(f"chunk output directory already exists: {args.out_dir} (never reused or overwritten)")
        if args.parent_chunk:   # refuse a missing, incomplete or tampered parent BEFORE anything is created
            from data_generation import chunking
            try:
                chunking.load_chain(args.parent_chunk)
            except chunking.ChunkError as e:
                parser.error(f"parent chunk refused: {e}")
    elif args.chunk_index is not None or args.parent_chunk:
        parser.error("--chunk-index / --parent-chunk need --chunks")

    if args.smoke_openai_timeout is not None and ((not args.smoke_total and args.chunks is None) or args.smoke_openai_timeout <= 0):
        parser.error("--smoke-openai-timeout requires a positive timeout and --smoke-total (or --chunks)")

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

    if args.chunks is not None:
        from data_generation import chunking
        totals = dict((config.get("counts") or {}).get("train") or {})
        if set(totals) != set(chunking.TASKS):
            parser.error("--chunks needs counts.train for create, fix and modify in the config")
        try:
            quota = chunking.increments(totals, args.chunks, args.chunk_index)
        except chunking.ChunkError as e:
            parser.error(str(e))
        config["counts"] = {"train": quota}
        config["chunk"] = {"index": args.chunk_index, "n_chunks": args.chunks, "totals": totals,
                           "parent_dir": os.path.abspath(args.parent_chunk) if args.parent_chunk else None,
                           "config_sha256": hashlib.sha256(config_bytes).hexdigest()}

    if raw.get("scale_only"):
        problems = validate_scale_only(config)
        if problems:
            sys.exit("scale_only safety check failed:\n  - " + "\n  - ".join(problems))
        print("[scale_only] safety checks passed (standard-only, original Modify, exclusions, reuse settings, separate output).",
              file=sys.stderr if args.check_pool else sys.stdout)

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
    if args.smoke_openai_timeout is not None:
        provider = config["llm_provider"]
        if (provider["module"], provider["class"]) != ("src.llm.openai.openai_provider", "OpenAIProvider"):
            parser.error("--smoke-openai-timeout requires OpenAIProvider")
        provider.setdefault("params", {}).update(timeout=args.smoke_openai_timeout, max_retries=0)
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

    guard = None
    if args.context_guard is not None:
        from llm.context_guard import ContextGuardProvider
        raw_provider = llm_provider   # the guard counts with the real provider and sits OUTSIDE the call recorder
        if not hasattr(getattr(raw_provider, "inner", raw_provider), "count_prompt_tokens"):
            parser.error("--context-guard needs a provider with count_prompt_tokens (OpenAIProvider on vLLM)")
        counter = getattr(raw_provider, "inner", raw_provider)
        max_tokens = ((config["llm_provider"].get("params") or {}).get("model_params") or {}).get("max_tokens")
        guard = ContextGuardProvider(llm_provider, counter, args.context_guard, max_tokens)
        llm_provider = guard

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
        "chunk": config.get("chunk"),
        "chunk_quota": config["counts"]["train"] if args.chunks is not None else None,
        "seed": builder.seed,
        "model": model_params.get("model"),
        "model_params": model_params.get("model_params"),
        **({"smoke_openai_sdk": {"timeout": args.smoke_openai_timeout, "max_retries": 0}}
           if args.smoke_openai_timeout is not None else {}),
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
        "context_guard": guard.stats if guard else {"enabled": False},
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

    if args.chunks is not None:
        # Checkpoint only a chunk that met its quota exactly; anything else stays as inspectable, unusable output.
        from data_generation import chunking
        got = {t: len(f) for t, f in chunking.collect_files(out_root).items()}
        if got != config["counts"]["train"]:
            print(f"[chunk] quota not met: produced {got}, expected {config['counts']['train']}; no checkpoint written",
                  file=sys.stderr)
            sys.exit(2)
        parent = config["chunk"]["parent_dir"]
        state = builder.chunk_state(out_root, chunking.load_state(parent)["self_sha256"] if parent else None,
                                    os.path.basename(parent) if parent else None)
        chunking.write_state(out_root, state)
        print(f"[chunk] checkpoint written: chunk {args.chunk_index}/{args.chunks}, {got}")


if __name__ == "__main__":
    main()
