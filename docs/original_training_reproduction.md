# Original Benchson training: reproduction assessment (before any 3K vs 7K fine-tuning)

Status: research and planning only. Nothing was trained, evaluated, installed, downloaded or submitted. Date of investigation: 2026-11-18.
Status words: VERIFIED (code, config, log or primary source), DERIVED, INFERRED, UNKNOWN, CONFLICTING.

## 1. Executive summary

- **Verdict: NOT YET.** The original training procedure is not in this repository, not in any of its branches or tags, and no fine-tuned model is published.
  The only description of it that we could reach is a secondary project summary (LoRA adapters on `ibm-granite/granite-4.1-3b`), without learning rate,
  optimizer, epochs, batch size, sequence length, loss masking, seed, framework or versions. These are UNKNOWN.
- What is verified: how the **training data are turned into prompts** (`src/export_training.py` + the evaluation's own `format_for_llm`), how the **test split is separated**
  (disjoint schema pools) and how **Benchson is evaluated** (lm-eval-harness tasks, greedy decoding, no constrained decoding).
- The 3K baseline cannot be reproduced exactly today. A fair 3K vs 7K comparison therefore requires **retraining both** under one reconstructed configuration
  (the task's "critical requirement"), unless the original pipeline owner supplies the missing configuration (section 13, blocker B1).
- Even a retraining of both is only partly "original": the reconstructed recipe will contain our own choices, to be labelled as deviations.
- Compute is not the obstacle: a 3B-parameter model with LoRA fits easily on one RTX PRO 6000; full fine-tuning is probably feasible too (section 12).

## 2. Evidence inventory

| # | Source | What it gives | Type |
|---|---|---|---|
| E1 | `src/export_training.py:1-12, 40-100` | Exporter of the train split to chat/alpaca JSONL; prompt built with the evaluation's `format_for_llm`; gold answer replaces the empty assistant turn | primary (code) |
| E2 | `src/data_generation/tasks/base_task.py:106-118` | `format_example`: chat = messages with the gold completion as last assistant turn; alpaca = instruction/input/output | primary (code) |
| E3 | `src/evaluations/create_by_schema/create_by_schema.py:43-61`, `json_error_evaluation/json_error_evaluation.py:28-46`, `modify_json_evaluation/modify_json_evaluation.py:27-45` | The exact prompts (system + user) per task | primary (code) |
| E4 | `lm_eval_tasks/benchson_create.yaml:10-17`, `benchson_fix.yaml:6-13`, `benchson_modify.yaml:6-13` | Evaluation decoding: `generate_until`, `until: ["\n\n\n\n"]`, `max_gen_toks: 2048`, `do_sample: false`, `temperature: 0.0` | primary (config) |
| E5 | `README.md:85-111`, `BENCHMARK.md:60-110` | Pipeline, train/test separation by disjoint schema pools, "train exported for SFT, test never", recommended `lm_eval --model hf ... --tasks benchson_hard --apply_chat_template` | primary (docs) |
| E6 | `pyproject.toml:15-23` | `hf` dependency group = `transformers>=4.43`, `torch>=2.12.1`, `accelerate>=0.26`; no `peft`, `trl` or training library; `hf` and `vllm` groups conflict | primary (config) |
| E7 | `git log --all -S` on `LoraConfig`, `SFTTrainer`, `peft`, `num_train_epochs`, `lora_r` | 0 commits contain any of them; `learning_rate` appears only in schema data. Branches `main`, `creation`, `run`, `improve-data-generation` and tags `v1.0.0`, `v2.0.0`, `v3.0.0` hold no trainer, notebook or training config | primary (history) |
| E8 | `README.md:102-103` and `BENCHMARK.md:101-102` reference `configs/export_training.json` | That config file is **absent** from the repository at HEAD and in the history searched | primary (docs vs tree) |
| E9 | HF profile `huggingface.co/aviv1ron1` (fetched 2026-11-18) | Models: none public. Datasets: `Benchson`, `benchson_json1`, `benchson_json2`, `SecQueRetrieval` | primary (external) |
| E10 | HF dataset card of `aviv1ron1/Benchson` @ `7962b09…` (`release/scale-only-7k/ORIGINAL_DATASET_CARD.md`) | Splits, metrics, disjoint schema pools; no model, no training statement | primary (external) |
| E11 | `docs/original_generation_settings.md:3-7` | The published snapshot was built differently from git `e7f8b22` (only 771 of 1269 train ids in the reconstructed pool) | primary (project docs) |
| E12 | Project doc "סיכום של הפרויקט עד הכניסה שלנו" (Claude project "ibm", created 2026-10-29), sections 2-3 | Describes "Experiment track A" (adapters) and "track B" (black-box benchmark run): variants, LoRA targets and r/alpha, hard-tier exact_match results, JSONSchemaBench results. States that the pipeline lives in `~/projects/granite_switch/json_schema_adapter` (own `CLAUDE.md`), job YAMLs in `~/projects/granite_switch/vela/json/`, artifacts on COS `/danieloh_cos/avivron/json-schema-adapter/` | **secondary** (a summary of a working session; not code, not logs) |

Not reachable from this session (and not in the repository): the `granite_switch` / `json_schema_adapter` repository, its job YAMLs, the COS artifacts,
the `lm_eval_results/` logs. A web search for them found only public Granite Switch pages, none with the Benchson adapter pipeline. Two GitHub and two Hugging Face
pages were outside the session's fetch allowlist (the model card of `ibm-granite/granite-4.1-3b`, the `epfl-dlab/JSONSchemaBench` card), so model size and the
JSONSchemaBench revision are not confirmed from primary pages.

## 3. Original model identification (the four models must not be confused)

| Role | Model | Status |
|---|---|---|
| Generated the ORIGINAL synthetic examples | `openai/gpt-oss-120b` (builder); round-trip validator `Qwen/Qwen3.5-397B-A17B-FP8` | VERIFIED (dataset card, `docs/original_generation_settings.md`) |
| Generated OUR 4,000 new examples | IBM Granite 4.0 H Small, served by vLLM (not a training model) | VERIFIED (our configs, release card) |
| Model fine-tuned in the original research | `ibm-granite/granite-4.1-3b` | INFERRED: one secondary source (E12). No code, config, log or model repo confirms it. Revision UNKNOWN; base vs instruct UNKNOWN (the name suggests a post-trained model; not verified) |
| Evaluation baseline | the same `granite-4.1-3b` without adapter ("base" in E12); also DeepSeek-V3.2 for the black-box benchmark | INFERRED (E12) |
| Model we intend to fine-tune for 7K | not decided | UNKNOWN: needs an explicit decision (recommended: exactly the model of the baseline, once confirmed) |

## 4. Original training procedure

| Question | Answer | Status |
|---|---|---|
| Models trained | `granite-4.1-3b` (five variants in E12: base, v1 LoRA, v2 LoRA, aLoRA, Shadow-Residual) | INFERRED |
| Separate or joint per task | UNKNOWN. The exporter writes one JSONL with a `task_type` field for all three tasks (E1), which suggests joint training, but nothing shows how the adapters were trained | UNKNOWN (exporter shape only) |
| Framework | UNKNOWN | UNKNOWN |
| Method | E12: v1/v2 = LoRA (targets q,k,v,o,gate,up,down; r=16, alpha=32); aLoRA = Activated LoRA (targets q,k,v,o; r=32, alpha=64); SR = Shadow-Residual (cross_stream, q; r=32, alpha=64) | INFERRED (secondary) |
| Training data of the variants | v1: "benchson v1 (gpt-oss synth, 5k)"; v2, aLoRA, SR: "benchson v3 train (3k)" | CONFLICTING naming: the public dataset is "v2 data scored by v3 code" (`BENCHMARK.md`). Which exact 3K was used (HF snapshot `7962b09` or the git `e7f8b22` data, which differ per E11) is UNKNOWN |
| Hyperparameters | not available | UNKNOWN |

Relevant for us: the **v2 LoRA** variant is the one that matches "train on the 3K Benchson train set" and is the natural baseline, but its configuration is only known for method, targets, r and alpha.

## 5. Training hyperparameter table

| Parameter | Original value | Source | Status | 7K proposed value |
|---|---|---|---|---|
| Base model | `ibm-granite/granite-4.1-3b` | E12 | INFERRED | same as the baseline once confirmed |
| Model revision | not recorded | none | UNKNOWN | pin one immutable commit, use it for both runs (deviation, labelled) |
| Training method | LoRA (v2) | E12 | INFERRED | same method as the reconstructed baseline |
| Training framework | not recorded | none | UNKNOWN | decision needed; nothing in this repo (E6) |
| Framework versions | not recorded | none | UNKNOWN | pin and record |
| Tokenizer | model's own | implied | INFERRED | same as base model revision |
| Chat template | model's own template applied to `messages` | E1/E2 produce `messages`; template use at train time unverified | INFERRED | same as baseline |
| Prompt format | system + user prompts of `format_for_llm` per task | E1, E3 | VERIFIED for the exporter | identical (same exporter) |
| Response format | compact `json.dumps(gold)` as the assistant turn | `src/export_training.py` (line `completion = json.dumps(record[spec["gold"]])`), E2 | VERIFIED for the exporter | identical |
| Loss masking | UNKNOWN (assistant-only or full sequence) | none | UNKNOWN | must be decided and applied equally |
| Learning rate | UNKNOWN | none | UNKNOWN | decision needed |
| Optimizer | UNKNOWN | none | UNKNOWN | decision needed |
| Scheduler | UNKNOWN | none | UNKNOWN | decision needed |
| Epochs | UNKNOWN | none | UNKNOWN | see section 11 |
| Batch size | UNKNOWN | none | UNKNOWN | decision needed |
| Gradient accumulation | UNKNOWN | none | UNKNOWN | decision needed |
| Effective batch size | UNKNOWN | none | UNKNOWN | decision needed |
| Warmup | UNKNOWN | none | UNKNOWN | decision needed |
| Weight decay | UNKNOWN | none | UNKNOWN | decision needed |
| Maximum sequence length | UNKNOWN | none | UNKNOWN | see section 6 (needed: examples are long) |
| Truncation | UNKNOWN | none | UNKNOWN | decision needed; must not silently cut JSON answers |
| Precision | UNKNOWN | none | UNKNOWN | decision needed |
| Random seed | UNKNOWN (data seed 0 is only the dataset build seed) | `docs/original_generation_settings.md` | UNKNOWN | fixed set of seeds |
| Gradient checkpointing | UNKNOWN | none | UNKNOWN | decision needed |
| LoRA configuration | v2: r=16, alpha=32, targets q,k,v,o,gate,up,down. Dropout, bias, rank-stabilization UNKNOWN | E12 | INFERRED (partial) | same r/alpha/targets; other fields recorded |
| Checkpoint saving | UNKNOWN (artifacts on COS) | E12 | UNKNOWN | save final adapter + config |
| Evaluation frequency | UNKNOWN | none | UNKNOWN | decision needed |
| Early stopping | UNKNOWN | none | UNKNOWN | none by default (do not add silently) |

About 20 of 27 parameters are UNKNOWN. Every UNKNOWN in the right-hand column is a decision of ours, not a reproduction, and has to be written down as such.

## 6. Dataset formatting and preprocessing

- Source: `data/benchmark_<task>/train/*.json` records, projected by `src/export_training.py` (E1). Default format `chat`; `alpaca` is optional. Output default
  `outputs/training_data/benchmark_train.jsonl` (the config file that sets it is missing, E8). Each line is `{"messages": [...], "task_type": ...}` (E2, E1).
- Prompt per task = the evaluation's own `format_for_llm` (E3). Create: instructions + "Description:" + "Schema:"; Fix: "correct the following JSON" with JSON and schema (and the
  description when present); Modify: JSON + instruction (+ schema). A system message is included. Training and evaluation prompts are identical by construction.
- Gold answers: Create `reference_json`, Fix `valid_json`, Modify `ground_truth`, written compactly with `json.dumps` (no indentation).
- The exporter reads per-example `.json` files; the Hugging Face dataset is stringified JSONL. For our runs the 7K needs the same conversion from the merged directory
  `data_scale_7k_20261114/` (or an equivalent verified conversion of the JSONL). Using the same exporter code for both the 3K and the 7K is part of the controlled design.
- Sequence length matters: average raw example size is about 6,200 characters in the original 3K and about 5,000 in the 7K (sum of schema, description and JSON fields,
  before the prompt template). The longest examples may exceed a 4K or 8K token limit; the original limit and truncation rule are UNKNOWN. Token lengths have not been measured
  because no tokenizer was run.
- Validation split: none known. Train/test separation is by disjoint schema pool (E5), but there is no documented validation set for training-time evaluation.
- Known data caveats that apply to both runs: three validation defects in original examples (preserved), eight schema contents present in both original train and test
  (`docs/publishing_workflow.md`), and the new examples being structurally simpler on average.

## 7. Original evaluation methodology

| Item | Finding | Status |
|---|---|---|
| Benchson held-out test | 450 examples (150 per task), lm-eval-harness tasks in `lm_eval_tasks/`; recommended headline: `exact_match` on `benchson_hard` (Github_medium, Github_hard, JSONSchemaStore `schemas`) | VERIFIED (E4, E5, `lm_eval_tasks/README.md:12-27`) |
| Command | `lm_eval --model hf --model_args pretrained=<model> --tasks benchson_hard --include_path lm_eval_tasks --apply_chat_template` | VERIFIED (`BENCHMARK.md`) |
| Decoding | greedy: `do_sample: false`, `temperature: 0.0`, `max_gen_toks: 2048`, stop at `"\n\n\n\n"` | VERIFIED (E4) |
| Constrained decoding / structured output | none in the task configs or the recommended command | VERIFIED for the Benchson harness; UNKNOWN for the JSONSchemaBench run |
| Metrics | `json_validity`, `schema_compliance`, `semantic_fidelity`, `exact_match` (all tasks); `change_fidelity` (Fix and Modify only). `exact_match` = valid + schema-compliant + every ground-truth field recovered. `json_validity` and `semantic_fidelity` saturate | VERIFIED (dataset card, README) |
| Which metric for which task | Create: validity, compliance, fidelity, exact. Fix/Modify: the same plus change_fidelity | VERIFIED |
| JSONSchemaBench | E12: `schema_compliance`, 2-shot, splits easy/medium/hard, via lm-eval, local logs in `lm_eval_results/{baseline,lora}/`. Task definition, few-shot examples, pinned dataset revision and decoding are not in this repository | INFERRED (secondary); revision UNKNOWN |
| Black-box benchmark run (track B) | granite-4.1-3b vs DeepSeek-V3.2, all 450; run from `~/projects/temp/test_benchson/v2` | INFERRED (secondary) |
| Scoring-code version | harness at git tag `v3` scores the `v2` dataset (`BENCHMARK.md`) | VERIFIED |

The evaluation prompts are fixed by E3 and E4. Any change would have to be documented as a deviation.

## 8. Available checkpoints and baseline results

- Published weights, adapters, logs, configs: **none found** (E9; no model on the HF profile; none in the repository).
- Results available only as numbers in the secondary summary (E12), not as logs:
  mean `exact_match` over 9 hard tiers: base 0.545, v1 LoRA 0.634, v2 LoRA 0.684, aLoRA 0.639; by family (v2 LoRA vs base): create 0.697 vs 0.540,
  fix 0.847 vs 0.771, modify 0.895 vs 0.731. JSONSchemaBench `schema_compliance` 2-shot: base 0.936/0.707/0.196, v1 0.931/0.681/0.236, v2 0.924/0.710/0.169 (easy/medium/hard);
  differences within about 1 to 1.5 standard errors, i.e. no transfer. Black-box: blended exact_match granite-4.1-3b 0.631, DeepSeek-V3.2 0.847.
  These are INFERRED until the original logs or a rerun confirm them. They must not be quoted as verified results.
- A checkpoint that might exist (COS, E12) is not accessible, and its training data and configuration are not verified, so it cannot serve as the baseline.

## 9. Reproducibility assessment

**NOT CURRENTLY REPRODUCIBLE.**

- Missing: training script, framework, versions, hyperparameters, sequence length, loss masking, seed, model revision, exact 3K dataset identity (CONFLICTING naming and E11), logs, checkpoints.
- Present: prompt and response construction (exporter), schema-disjoint split, evaluation tasks and decoding, dataset revisions on Hugging Face, method and LoRA shape (secondary).
- Smallest documented deviation for a fair comparison: **retrain both** 3K and 7K under one reconstructed configuration, written down before running, with every
  parameter marked "original (INFERRED)" or "our decision". Do not compare a new 7K run with the old reported numbers.

## 10. Proposed controlled 3K vs 7K experiment

Identical in A and B: base model and pinned revision, tokenizer and chat template, exporter and prompts (E1-E3), assistant-only or full-sequence loss (one choice), LoRA configuration,
optimizer, learning rate, schedule shape, effective batch size, precision, maximum length and truncation rule, seeds, evaluation prompts, decoding (E4), the 450-example test split.

Changes: only the training set.
- A: original 3,000 examples (`main`, `1e8aad27…`, byte-identical to `aviv1ron1/Benchson` @ `7962b09…`).
- B: 7,000 examples (`scale-only-7k`, commit `668a6773…`).

Use at least three seeds per arm and report mean and spread (this repository's data build seed 0 is unrelated to training seeds). Primary evaluation: Benchson test with unconstrained greedy
decoding, as in E4. Secondary and separate: JSONSchemaBench as in the original attempt (once its task definition and revision are recovered), and optionally a constrained-decoding analysis,
labelled as a different condition.

Limitation to keep in the plan: the new examples come from a different generator, use controlled schema reuse, and are structurally simpler on average. The experiment evaluates
our Scale Only expansion procedure, not training-set size in isolation.

## 11. Training budget comparison

How the original defined duration (epochs, steps, tokens): **UNKNOWN**.

| | Option 1: same epochs | Option 2: same optimizer updates |
|---|---|---|
| 3K vs 7K updates (effective batch B, E epochs) | 3K: E*ceil(3000/B); 7K: E*ceil(7000/B) (2.33x more) | equal, by construction |
| Passes per example | equal | 7K gets 0.43x the passes |
| Unique data seen | more with 7K | more with 7K |
| Compute | about 2.33x more for 7K (and fewer tokens per example in 7K, so slightly less than that) | equal |
| Risk | more updates may explain gains; also more overfitting risk on repeated schemas | 7K may be undertrained per example |

Recommendation: **primary = same number of epochs.** It matches the usual reading of "same procedure, bigger dataset", and it is the only version that does not need a
different schedule per arm. It is the answer only if the original also used epochs; that is unknown, so confirm with the pipeline owner first. **Secondary = matched optimizer updates (and, if cheap, matched tokens)**, reported
separately, so that a gain from extra compute can be told apart from a gain from extra data. The choice must be explicit in the run record. Never change it between arms.

## 12. BGU compute feasibility (estimates, not measured; nothing was run on the cluster)

Assumptions: model about 3B parameters (from the name; not confirmed from a primary page), bf16, one RTX PRO 6000 Blackwell (about 96 GB, from the hardware's published spec; verify with `nvidia-smi`).

| Question | Assessment |
|---|---|
| Memory, LoRA | weights about 6 GB plus activations and adapter state; comfortably within one GPU at moderate batch size |
| Memory, full fine-tuning | about 16 bytes per parameter with AdamW mixed precision is about 48 GB before activations; probably feasible on one 96 GB GPU, tight with long sequences. Only needed if the original was full fine-tuning (it appears not to be) |
| Multi-GPU needed | not for LoRA; likely not for full fine-tuning |
| Dependencies | `pyproject.toml` has an `hf` group (transformers, torch, accelerate) but no `peft` or `trl`, and it conflicts with the `vllm` group used for generation. Use a **separate** environment; do not touch the generation environment |
| Disk | LoRA adapter checkpoints are small (tens to hundreds of MB each); full fine-tuning is about 6 GB per bf16 checkpoint plus optimizer state about 24 GB if saved |
| GPU hours | tokens per epoch not measured. Rough character counts: 7K about 35 M characters, 3K about 18.7 M (before the prompt template, which adds text). At about 3 characters per token that is roughly 6 M (3K) and 12 M (7K) tokens per epoch. Training FLOPs about 6 x 3e9 x tokens: about 1e17 (3K) and 2e17 (7K) per epoch; at an assumed effective 100 TFLOP/s this is minutes to about an hour per epoch. Order of magnitude: single-digit GPU-hours per arm for a few epochs, before seeds and evaluation |
| Smoke test | yes: one tiny run (a few steps) to verify tokenization, loss masking, maximum length (how many examples would be truncated) and checkpoint saving, before any real run |
| Reproduce the original environment | not possible without the original versions; use the newest compatible, pin and record them |

## 13. Blockers

| # | Blocker | Class | Smallest resolution |
|---|---|---|---|
| B1 | Original training script, hyperparameters and framework are missing | CRITICAL | Ask the owner of `granite_switch/json_schema_adapter` (the user's IBM mentors / Aviv Ron) for the `CLAUDE.md`, the training script and the job YAMLs from `vela/json/`; otherwise reconstruct and retrain both arms |
| B2 | Which exact 3K dataset and which base model revision were used | CRITICAL | Same request; check the COS artifacts' metadata |
| B3 | Model identity is only a secondary claim; base vs instruct unknown | IMPORTANT | Confirm from the pipeline repository or the adapter config (`base_model_name_or_path`) |
| B4 | Training budget definition (epochs vs steps) unknown | IMPORTANT | Same request; otherwise choose same epochs (section 11) and say so |
| B5 | Loss masking, max length and truncation unknown | IMPORTANT | Same request; smoke test measures truncation; decide once for both arms |
| B6 | `configs/export_training.json` missing | IMPORTANT | Recreate from the documented defaults (format `chat`), recorded as a new file, no code change to the exporter |
| B7 | JSONSchemaBench task definition, revision and decoding not in the repository | IMPORTANT | Obtain the lm-eval task from the pipeline repository; pin a dataset commit |
| B8 | Original checkpoints and logs not accessible | IMPORTANT | Not needed if both arms are retrained |
| B9 | No validation split for training-time monitoring | OPTIONAL | Report test only at the end; avoid test-set model selection |
| B10 | Model weights and tokenizer must be downloadable on BGU | OPTIONAL (check before launch) | Test access without downloading large weights; model size not confirmed from a primary page |
| B11 | Three original defects and new-data simplicity | OPTIONAL | Already documented; keep in the plan |

## 14. Recommended next steps

1. Send the owner of the adapter pipeline one precise request (B1-B5, B7): training script, job YAMLs, adapter configs, base model id and revision, the dataset file used for the 3K, the budget definition, the JSONSchemaBench task and revision.
2. In parallel (read-only, no download): confirm model size and revision from the Hugging Face model card; record the pinned commit.
3. If the information arrives, write the reconstructed recipe table with sources; if not, write the "our decision" recipe explicitly, as a deviation.
4. Recreate `configs/export_training.json` (documented defaults), export the 3K and the 7K with the same exporter, measure token lengths with the real tokenizer (no training).
5. Prepare a separate training environment and a short smoke test, to be run only after approval (no Slurm submitted so far).

## Answer to the key question

**Can we train on our 7K dataset using exactly the same model and training procedure as the original Benchson experiment? NOT YET.**
The training procedure and its hyperparameters are not recoverable from the repository or from public sources. They would have to come from the original pipeline owner,
or both arms must be retrained under an explicitly reconstructed configuration.

---

## Addendum (2026-11-18): exhaustive Git investigation of `BenchsonTheJsonBenchmark/benchson`

Scope: every remote branch and tag, full history (read-only). Result: the first report's conclusion is confirmed and its evidence base strengthened. Corrections are listed below.

**Git coverage.** `git ls-remote --heads --tags origin` matches the local refs exactly (4 branches: `main` 8642024, `improve-data-generation` 351eef6, `creation` 8f321fa, `run` d1de51b; 3 tags: `v1.0.0` e2b34cd, `v2.0.0` ca0800e, `v3.0.0` fc95e29). The repository is not shallow; all 59 commits are present (`git rev-list --all --count`). There is no `.gitmodules`, no `.github/`, no notebook (`*.ipynb`) and no training/tuning file in any commit (160 distinct non-data paths ever existed).

**Search across all commits** (`git log --all -S`): `granite_switch` 0, `json_schema_adapter` 0, `avivron` 0, `danieloh` 0, `SFTTrainer` 0, `Trainer(` 0, `QLoRA` 0, `aLoRA` 0, `deepspeed` 0, `target_modules` 0, `lora_alpha` 0, `lora_r` 0, `gradient_accumulation` 0, `warmup` 0, `weight_decay` 0, `max_seq_length` 0, `num_train_epochs` 0, `per_device` 0. Non-zero hits were inspected: "LoRA/lora" = one argparse description in `src/generate_data.py` (be074c9, a data generator, not a trainer); "PEFT" = noise inside logo files (47016f2); "vela" = the name "Lovelace" in tests; "granite-4" = the generation model `granite-4.0-h-small` in our own Slurm scripts; "accelerate" = inference dependency of `src/llm/huggingface/huggingface_provider.py` and `pyproject.toml`; "epochs", "batch_size", "optimizer", "scheduler", "adapter" = no training-related line.

**Origin of `granite_switch/json_schema_adapter`.** It appears in zero commits and in no tree. It originates only from the project summary document "סיכום של הפרויקט עד הכניסה שלנו" (section 2), which cites local paths on the original author's machine (`~/projects/granite_switch/json_schema_adapter`, `~/projects/granite_switch/vela/json/`, COS `/danieloh_cos/avivron/json-schema-adapter/`). It is not a directory, not a submodule, and not a link in this repository. `vela` and the COS bucket are IBM internal infrastructure names. The public Granite Switch repository (checked via a public fork, `IRENEKO/granite-switch`, top level only) has `.github`, `.pre-commit`, `ci`, `docs`, `src/granite_switch`, `tests`, `tutorials` and no `json_schema_adapter`; subdirectories were not expanded. Not verified: whether an IBM-internal repository with that name exists and is accessible to us.

**Changes to the first report.**

| Item in first report | Status after the exhaustive search |
|---|---|
| Training script, framework, hyperparameters, LoRA details, seed, budget, loss masking, max length: UNKNOWN | Still UNKNOWN. Nothing in any branch, tag or commit. |
| B6: `configs/export_training.json` "absent" | Explained: `.gitignore:178` ignores `configs/*` (exceptions only for `example.json`, `preview_*`, `scale_only_*`, `import_jsonschemabench_expand.json`). The file was documented in the README (a35bb44, e7f8b22) but never tracked. Same for the original `build_benchmark.json`/`run_benchmark.json`. |
| Exporter and prompts: "VERIFIED" | Strengthened: `src/export_training.py` was added in e7f8b22 (2026-06-16) and is unchanged since; the `format_example` body is byte-identical at e7f8b22 and HEAD; the three `format_for_llm` prompt builders were not changed after e7f8b22 (the later diffs to the evaluation files only add `exact_match` and `change_fidelity`, i.e. metrics). So today's exporter produces the same training prompts as at the first v2 build. Whether the adapters were actually trained with this exporter remains UNKNOWN. |
| "Which exact 3K": CONFLICTING | New evidence: the git v2 data at e7f8b22 has create train = 950, fix 1000, modify 1000 (2,950), while the Hugging Face snapshot has 1000/1000/1000 (3,000). `data/benchmark_*/` was removed from git in 8f321fa ("generated data moved to HF"). `docs/data_generation_research_notes.md` also calls the baseline "~2K". The training dataset is therefore still UNKNOWN: HF 3,000, git 2,950, or an internal snapshot. |
| JSONSchemaBench evaluation: not in repository | Confirmed: the repository only imports JSONSchemaBench schemas (`src/import_jsonschemabench.py:27`, dataset name `epfl-dlab/JSONSchemaBench`, no revision pinned). No evaluation task for it exists in any commit. |
| Evaluation of Benchson | Unchanged: `lm_eval_tasks/` (9a974a2, c6ea010, 8f321fa, 62e0c30), greedy decoding, no constrained decoding. |
| "Branches/tags hold no trainer" (first report: stated before remote parity was checked) | Now verified against `ls-remote` and full history. |

No factual error was found in the first report's UNKNOWN table; its wording on `configs/export_training.json` ("absent, reason unknown") was incomplete, and it did not yet use the 950 vs 1,000 evidence. Verdict unchanged: NOT REPRODUCIBLE from this repository alone.
