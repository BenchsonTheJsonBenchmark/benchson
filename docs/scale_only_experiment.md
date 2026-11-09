# Scale-only experiment: 3,000 -> ~7,000 training examples (100% original-style)

> **Policy update (2026-11-04):** no pool expansion. The experiment uses **controlled schema reuse** on the original training pool;
> see `docs/schema_reuse.md` (design, config, feasibility) and `docs/original_generation_settings.md`. It preserves the original generation
> logic and schema pool but **changes the original one-example-per-schema sampling policy**, and uses **Granite instead of gpt-oss-120b**
> (a second experimental difference). Sections 2 and 4 below describe the earlier strict-exclusion design and are superseded where they conflict.

## Hugging Face experiment repository (verified 2026-11-04)

| | |
|---|---|
| Our repo (private) | `benchson-research/benchson-data-experiments` |
| Original dataset (never modified) | `aviv1ron1/Benchson` |
| Original revision (its `main`) | `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6` (the `v2` tag `58b8525c...` has identical data files) |
| Our `main` commit (baseline copy) | `1e8aad270c7f0e1f82d139bbe1e32a345188b697` |
| Content of our `main` | the six data files byte-identical to the original (create/fix/modify x train/test = 3,000 train / 450 test), plus `README.md`, `NOTICE.md`, `ORIGINAL_DATASET_CARD.md`, `PROVENANCE.json` |
| Verification | `hf_baseline.py verify-remote` passed (git blob ids, sizes, sha256 and row counts re-downloaded from our repo) |
| Planned branches (NOT created) | `scale-only-7k`, `mixed-80-20` |

The copy was made with `src/hf_baseline.py` (see `docs/publishing_workflow.md`). Record both SHAs (original `7962b09...`, ours `1e8aad2...`) in every
experiment log. Credential review of the baseline: `docs/credential_reviews.json` (one exact value, public part of a truncated RSA container, five places).
Known limitation of the original benchmark (kept unchanged): 8 schema contents occur in both train and test under different ids; no example is identical across splits.

## Commands

```bash
# 1. Import the pinned baseline (needs HF access; verifies git blob ids; refuses a non-empty dir)
python src/import_published_benchmark.py --revision 7962b0947f7d96eb3a0e6b9b928d939c7f3061a6 --out-dir data_published/7962b0947f7d
# 2. Offline checks (no model)
export PUBLISHED_SHA12=7962b0947f7d
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --check-pool --assumed-acceptance 0.7
python src/verify_standard_equivalence.py
# 3. SMOKE TEST, 4 examples (REAL model - needs approval; same pipeline; output data_scale_only/<run>_smoke)
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --smoke-total 4 --max-llm-calls 200 \
    --record-calls data_scale_only/smoke_calls.jsonl --env-file .env
# 4. FULL RUN (REAL model - needs approval; resumable; output data_scale_only/<run>)
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --max-llm-calls 40000 \
    --record-calls data_scale_only/${RUN_NAME}_calls.jsonl --env-file .env
# 5. Merge (third directory) and local release folders (nothing uploaded)
python src/merge_scale_dataset.py --baseline data_published/7962b0947f7d --new data_scale_only/${RUN_NAME} --out-dir data_scale_7k --allow-train-schema-reuse --jsonl-out hf_export_7k
```
Offline rehearsal with the mock model: edit `REPLACE_SHA12` in `configs/scale_only_mock.json`, then run step 3 with that config (the mock returns the
same object every time, so repeated attempts are rejected as duplicates - it exercises the retry/skip path, not data quality).
Expected: `data_published/` (read-only baseline), `data_scale_only/<run>/benchmark_*/train/*__a<n>.json` (~1334/1333/1333) with reports and `generation_manifest.json`,
`data_scale_7k/` (train ~2334/2333/2333, test 150 each byte-identical). Nothing leaves the machine; see `docs/publishing_workflow.md`.

## 1. What was added (all new files unless marked "changed")

| File | Purpose |
|---|---|
| `src/data_generation/published.py` | ids, schema hashes, published<->builder format conversion |
| `src/import_published_benchmark.py` | import the HF snapshot at a pinned SHA into a read-only baseline |
| `src/data_generation/benchmark_builder.py` (changed) | `exclusions` (skip published schemas), pool stats, event log (provider errors / empty / suspected truncation) |
| `src/build_benchmark.py` (changed, backward compatible) | `--check-pool`, `--smoke-total N`, `--out-dir`, `--max-llm-calls`, `--record-calls`, `--env-file`, `scale_only` safety checks, `generation_manifest.json` |
| `src/merge_scale_dataset.py` | merge baseline + new run into a third directory, with leakage/field/test-identity checks |
| `src/verify_standard_equivalence.py` + `tests/test_standard_equivalence.py` | proves standard mode == original code on `main` |
| `src/llm/openai/openai_provider.py`, `src/llm/vllm/vllm_provider.py` (changed) | opt-in `drop_empty_assistant_turn`; `VLLMProvider.model` set |
| `configs/scale_only_mock.json`, `configs/scale_only_vllm_granite.json` | configs (placeholders only) |
| `slurm/scale_only_vllm.sbatch` | UNTESTED template |
| `.gitignore` (changed) | tracks `configs/scale_only_*.json`; ignores `data_published/`, `data_scale_only/`, `data_scale_7k/`, `hf_export_7k/` |
| `tests/test_import_published.py`, `test_merge_scale.py`, `test_scale_only.py`, `test_standard_equivalence.py` | tests |

The complex-mode code from the previous task is untouched; the scale-only config simply sets
`complex_ratio: 0`. The later 80/20 run uses `configs/preview_complex_*.json`-style configs.

## 2. Findings about the existing builder (verified in code, not assumed)

* One example per task per schema (id = `<source>__<subset>__<name>`, file `<id>.json`). The three
  tasks of one schema share one reference JSON. There is no mechanism for several examples per schema,
  and none was added: 1,334 Create examples need >= 1,334 distinct, unused schemas that pass the gates.
  Create is the limiting task (grounding >= 0.9), so the schema pool must be larger than the target.
* Resume = "skip if `<id>.json` exists in the split dir". Pointing it at the published data would
  not work (published data is JSONL with stringified fields, not `<id>.json`) and could mix runs,
  so the new run writes to its own directory and the merge happens afterwards.
* Train/test separation in the builder is by schema, via a seeded stratified split. It does not know
  which schemas the published dataset used, hence the `exclusions` file (ids **and** content hashes).
* **Modify**: without a `generation` block the branch uses the newer recursive/weighted selection.
  The scale-only config therefore must (and `scale_only` validation enforces) a `generation` block
  with `standard_modify_strategy: "original"`.
* Fix: the original injected the error with the unseeded global `random`; the branch uses a seeded rng.
  Same mutation set, but not bit-reproducible against the old code (nothing to reproduce - it was random).

## 3. Verification that standard == original

`python src/verify_standard_equivalence.py` (also a unit test): exports `main`, runs old and new builders
with the offline mock and the same seed. Result on this branch: Create and Modify files byte-identical,
Fix schema/valid_json identical, same Create prompt, same default gates. The negative control
(legacy default on the branch) differs in Modify, so the check can fail.
Limitation: it uses the mock model, so it proves the logic/prompts, not model behaviour.

## 4. Procedure

```bash
# 0. (Bar) get the pinned revision SHA of aviv1ron1/Benchson from the HF page ("History") - 40 hex chars.
# 1. Import baseline (read-only, refuses a non-empty dir):
python src/import_published_benchmark.py --revision <SHA40> --out-dir data_published/<SHA12>
#    offline alternative: add --source-dir <downloaded snapshot dir>
# 2. Pool feasibility (no model):
export PUBLISHED_SHA12=<SHA12>
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --check-pool
# 3. Offline rehearsal with the mock model (free):
python src/build_benchmark.py --config configs/scale_only_mock.json --smoke-total 4   # edit REPLACE_SHA12 first
# 4. REAL smoke test, 3-4 examples (needs approval; same pipeline, output in data_scale_only/<run>_smoke):
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --smoke-total 4 \
    --max-llm-calls 200 --record-calls data_scale_only/smoke_calls.jsonl --env-file .env
# 5. REAL full run (needs approval; resumable; output data_scale_only/<run>):
python src/build_benchmark.py --config configs/scale_only_vllm_granite.json --env-file .env
# 6. Merge (new directory, baseline untouched):
python src/merge_scale_dataset.py --baseline data_published/<SHA12> --new data_scale_only/<run> \
    --out-dir data_scale_7k --jsonl-out hf_export_7k
```

## 5. Expected output

```
data_published/<SHA12>/            read-only baseline: benchmark_{create,fix,modify}/{train(1000),test(150)}/, reserved_schemas.json, published_manifest.json
data_scale_only/<run>/             new run: benchmark_*/train/ (~1334/1333/1333), _metadata/ sidecars, generation_manifest.json,
                                   _generation_reports/<ts>/{generation_summary.json, generation_failures.jsonl, generation_events.jsonl}
data_scale_7k/                     merge: train ~2334/2333/2333, test 150 each (byte-identical), merge_manifest.json
hf_export_7k/<task>/{train,test}.jsonl   optional, same stringified format as the published dataset
```

Logged: failed examples (`generation_failures.jsonl`), duplicates (summary), provider errors / empty /
suspected truncation (`generation_events.jsonl`, heuristic: JSON-looking answer that never closes or an unclosed `<think>`).

## 6. Findings from the follow-up investigation (2026-11-04)

* **Pinned revision (verified via the HF API through the desktop browser):** `aviv1ron1/Benchson` `main` =
  `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6` (last modified 2026-06-29). Tags: `v1` = `d588e33b...`, `v2` = `58b8525c...`.
  The six data files are identical in `main` and `v2` (same git blob ids, listed in `published.KNOWN_REVISIONS`); only README differs.
  The importer verifies downloaded files against these blob ids.
* **Schema pool reconstructed exactly.** git `e7f8b22` contains `data/jsb_schemas` (1308 files). Restored (git-ignored, nothing overwritten) to
  `data/jsb_schemas`; with seed 0 / test_ratio 0.15 / 10 KB the builder reproduces the manifest numbers: 1669 eligible = 250 test pool + 1419 train pool.
* **Capacity** (`python src/analyze_schema_pool.py --reserved ...`; figures below use the git `e7f8b22` data as a PROXY for HF, whose Create train has
  1000 rows instead of 950, so real numbers will be slightly lower; rerun after the real import):
  schemas unused by any published task in the train pool = **152** (+61 never-used schemas in the original test pool);
  per-task slots without a published example = Create 469 / Fix 419 / Modify 419 = **1307** upper bound even if every attempt succeeded.
  **4,000 extra examples are not reachable on the original pool without reusing schemas**; the strict policy yields <= ~456 (<= ~640 with the leftover test-pool schemas).
* The published data also reveals content leakage that already exists: 10 schema contents appear in both original train and test POOLS (different ids); 8 of them produced examples on both sides in the published files.
  New data excludes by id and by content hash, so it cannot add to this.
* Upstream JSONSchemaBench (`epfl-dlab/JSONSchemaBench`, `test` split sizes from the dataset-server): Github_easy 577, Github_medium 586, Github_hard 368,
  Glaiveai2K 513, Kubernetes 320, Snowplow 121. The v2 import took 300/300/192/200/200/116, so roughly 1,000 more schemas (before the 10 KB filter) exist
  upstream in the same subsets. `configs/import_jsonschemabench_expand.json` imports them into a **separate** folder; `--expanded-source` checks that the
  original schemas are a subset. This is an expansion of the schema pool, not synthetic data, but it needs a decision (see report) and is not run.

## 6b. Unresolved before the first real run

1. Decision on how to reach the target: (c) controlled schema reuse was chosen and is implemented (`docs/schema_reuse.md`); (a) and (b) were not pursued. It differs from the one-example-per-schema methodology and must be reported as such.
2. Run the real import on a machine with HF access, then `analyze_schema_pool.py --reserved` for exact numbers.
3. Unknown original sampling settings (investigated, still unknown; see `docs/original_generation_settings.md`); the config keeps our provisional temperature 0.7 / max_tokens 4096.
4. `exclusions` is currently strict (a schema used by any published task is excluded). Per-(task, schema) exclusion (option B) is not implemented pending your decision.
5. Truncation detection is a heuristic; the OpenAI-compatible provider does not expose `finish_reason`.

## 7. Granite 4 H-Small on the BGU Slurm cluster

* Granite 4.0 H-Small is (as far as I know; verify on the model card) a ~32B-parameter hybrid Mamba/transformer MoE (~9B active).
  Weights alone are ~64 GB in bf16 -> a GTX 1080 Ti (11 GB, no bf16) cannot run it. It needs an 80 GB-class GPU
  (A100/H100 80GB) or several smaller GPUs with tensor parallelism. I do not know which GPUs BGU offers: check
  `sinfo -o "%P %G %m"` and the cluster docs.
* Repo support: `VLLMProvider` (in-process `LLM.chat`, strictly sequential) and `OpenAIProvider(base_url=...)`
  for a `vllm serve` endpoint (supports concurrency, used in the template). `HuggingFaceProvider` (transformers) is not suitable for this size/throughput.
  `pyproject.toml` marks the `hf` and `vllm` dependency groups as conflicting - use a dedicated vLLM env.
* Whether the installed vLLM version supports Granite 4 hybrid models must be checked. No model or quantization is substituted:
  if bf16 does not fit, an FP8/quantized variant is a decision for you, not made here.
* Slurm needs: GPU partition/type, time limit, large writable HF cache, internet or pre-downloaded weights (compute nodes are often offline), a vLLM env.

## 7. Protocol: published-baseline anchored pool (approved 2026-11-08)

Section 6 stated that the pool was "reconstructed exactly". That is true for the FIRST build (git `e7f8b22`: 950 Create-train examples, 250/1419
partition) but not for the published Hugging Face snapshot, which has 1000 Create-train examples and train schemas that do not line up with that
partition. The scale-only run therefore samples from the schemas of the published training rows (`reuse.pool_source: "published_train"`,
see `docs/schema_reuse.md`). Preparation on a machine that has `data_published/<sha12>/`:

    python src/export_anchor_pool.py --baseline-dir data_published/7962b0947f7d

### What this experiment does and does not isolate (state this when reporting results)
Scale-only changes TWO things relative to the published data, so it does **not** isolate dataset quantity under otherwise identical generation conditions:
1. **Generator model**: IBM Granite 4.0 H-Small instead of `openai/gpt-oss-120b` (the prompts still contain the gpt-oss-specific `Reasoning: low/high` lines).
2. **Sampling policy**: controlled schema reuse (up to `max_new_references_per_schema` independent, deduplicated examples per schema) instead of one example per schema.
Further caveats: the pool is limited to schemas that already occur in the published train set (schema diversity does not grow), a documented 10 KB size
restriction removes larger schemas (the published train set contains some), temperature 0.7 / max tokens 4096 / top_p 0.95 are provisional,
and the published train and test sets already share some schema content (reported by the importer).
