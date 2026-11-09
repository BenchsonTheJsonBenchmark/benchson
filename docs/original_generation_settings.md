# Original Benchson v2 generation settings: confirmed vs unknown

> **Correction (2026-11-08).** The "Confirmed" table below comes from the git manifests of commit `e7f8b22` (the FIRST build, 2026-06-15). The published
> Hugging Face snapshot (`7962b09…`, 2026-06-29, commit `8f321fa` mentions "hardened-corpus configs", which are git-ignored and not in the repository) was
> built differently: of its 1269 train schema ids only 771 are in the pool reconstructed from `data/`. Its max_schema_kb, partition, seed and corpus are
> therefore UNKNOWN. Builder model and round-trip validator model are stated by the dataset card; the rest of the table is verified for `e7f8b22` only.
> The scale-only run no longer relies on these values to define its pool (see `docs/schema_reuse.md`, pool source `published_train`).

Sources: published manifests in git `e7f8b22:data/benchmark_*/manifest.json` (build of 2026-06-15), the
Hugging Face dataset card of `aviv1ron1/Benchson` @ `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6`
(read via the HF API on 2026-11-04), `BENCHMARK.md`, and the builder code at `e7f8b22` (identical defaults on this branch).

## Confirmed
| Setting | Value | Source |
|---|---|---|
| Builder model | `openai/gpt-oss-120b` | manifests, HF card |
| Round-trip validator model | `Qwen/Qwen3.5-397B-A17B-FP8` | manifests, HF card (used for the test split) |
| Seed | 0 | manifests |
| Schema sources | `schemas` + `jsb_schemas`, `max_schema_kb` 10 each | manifests |
| Pool / partition | 1669 eligible = 250 test pool + 1419 train pool, `test_ratio` 0.15 | manifests; **reproduced exactly** from git `e7f8b22` `data/jsb_schemas` (see `src/analyze_schema_pool.py`) |
| JSB subset sizes in the pool | Github_easy 300, Github_medium 300, Github_hard 192, Glaiveai2K 200, Kubernetes 200, Snowplow 116 | git `e7f8b22` `data/jsb_schemas` (1308 files) |
| Gate defaults | grounding 0.9, min_populated_fields 2, max_build_attempts 3, round_trip_threshold 0.9, naturalize_attempts 2, max_retries 3 | code defaults, unchanged on this branch (checked by `verify_standard_equivalence.py`) |
| Train gates | deterministic only, no round-trip | BENCHMARK.md ("Per-split gates") |
| Sizing | ~150 test / ~1000 train per task | BENCHMARK.md; HF card: 1000/150 per task |
| Prompts | include hard-coded `Reasoning: low` (create, description) and `Reasoning: high` (fix) lines (gpt-oss specific) | commit 567dde7; preserved here unchanged |
| One example per task per schema | file `<source>__<subset>__<name>.json` | builder code |

## Unknown (not in the repo; the build configs under `configs/` are git-ignored)
**Temperature and max output tokens (investigated 2026-11-04, not recoverable from the repo).** Parameter path on `main`: build config
`llm_provider.params.model_params` -> `LLMProvider.from_config` (`src/llm/llm_provider.py`) -> `OpenAIProvider.__init__` -> `_generate`
(`params = {**self.model_params, **(parameters or {})}`) -> `chat.completions.create`. The builder never passes per-call `parameters`, and
`OpenAIProvider` has no Python default for either value, so they came from the (never committed) build config or from the server's defaults.
The manifest model names come from `llm_provider.model`, which only the OpenAI-style providers set, and the OpenAI `base_url`/`model_params` support
was added just before the v2 data, so the original was most likely `OpenAIProvider` against an OpenAI-compatible server (inference, not confirmed).
Other providers' defaults (vLLM provider 0.7 / 1024, HF provider 1024, WatsonX 20) cannot be tied to the published data.
Our `0.7` / `4096` come from this branch's WatsonX preview config (October 2026, after v2) and are **our provisional values, not the original ones**;
`temperature: 0.0` in `lm_eval_tasks/*.yaml` is evaluation only. To close this, ask the original author for the build config or the server launch command.

Also unknown: top_p, `system_prompt_prefix`, `concurrency`, `validation.test` values beyond defaults,
`counts` actually configured, the JSB import config (`split`, `seed`, subset order; the importer defaults are `test`/0 and
upstream test-split sizes are consistent with that, but unverified), the serving stack, and the exact run history
(e.g. why git `e7f8b22` has 950 Create-train examples while the HF dataset has 1000).
Anything in `configs/scale_only_*.json` for these (temperature 0.7, max_tokens 4096, top_p 0.95) is a **provisional value and must be marked as an assumption in results**.

## Differences of the scale-only experiment (state both when reporting results)
1. Sampling policy: the original made at most one example per task per schema; this experiment uses controlled schema reuse (several independent,
   deduplicated examples per schema, capped; see `docs/schema_reuse.md`). Schema pool, logic, prompts and thresholds are unchanged.
2. Generator model: Granite instead of gpt-oss-120b.

## Limitation (must be stated when reporting results)
Even with identical generation logic, prompts and thresholds, the new examples come from a different model
(IBM Granite instead of gpt-oss-120b). The result is a scale-up using the same pipeline, **not an exact reproduction**
of the original generation process. Granite will also see the `Reasoning: low/high` lines, which were written for gpt-oss.
Do not change them for the scale-only run (original prompts preserved); note the effect if quality differs.
