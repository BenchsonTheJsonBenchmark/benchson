# Pre-training readiness audit: 3K original vs 7K Scale Only

Date: 2026-11-18. Branch `improve-data-generation` (HEAD `351eef6`). Scope: preparation only. No training, no inference, no Slurm, no
download of weights, no upload, no commit. The BGU cluster and Hugging Face are not reachable from the audit session.

Status words used below: **VERIFIED** (executed here, result observed), **READY BUT NOT EXECUTED** (command or code prepared, not run),
**BLOCKED** (needs an input that does not exist yet), **NOT APPLICABLE**. A check that was only read in code is called "inspected", never "verified".

## 1. Executive summary

- The existing exporter `src/export_training.py` (unmodified since `e7f8b22`) was run on the pinned original 3K and on the verified 7K Scale Only
  release. Both exports are complete, structurally identical in format, deterministic (same SHA-256 on a rerun) and free of any row-level overlap
  with the test split. **VERIFIED.**
- 3K export: 3000 rows (1000/1000/1000). 7K export: 7000 rows (2334/2333/2333). The 3000 original rows inside the 7K export are byte-identical to
  the 3K export (as a multiset). 4000 rows are new; none is identical to an original row. **VERIFIED.**
- The training prompt equals the lm-eval prompt text for the same record (system + blank line + user), and the metric code in `lm_eval_tasks/utils.py`
  agrees with `src/evaluations/metrics.py` on the tested cases. **VERIFIED by 22 new tests.**
- The new 4K examples are clearly shorter than the originals (characters, a proxy): mean total 4.8K vs 7.9K (create), 6.6K vs 9.7K (fix), 6.1K vs 8.0K
  (modify). A fixed-epoch recipe therefore gives the 7K run about 2.3x the examples but only about 1.9x the text (35.0M vs 18.65M chars).
  See section 5 and 10.
- A scientifically valid comparison is **not yet possible**, for reasons that are not in the data: the original training recipe is unknown
  (`docs/original_training_reproduction.md`), the tokenizer and base model are unconfirmed, and the JSONSchemaBench evaluation protocol is not in the repo.
- Safeguard gaps (section 10): the exporter silently skips records, has no count check and no hash output; an export for a comparison must be
  re-counted. The Test split is protected only by the exporter never opening it, not by a guard.
- The 3K and 7K train sets contain schema contents that also occur in the Test split of the original baseline (8 distinct schema contents, a
  property of the baseline already documented in `tests/test_hf_baseline.py`). The 4K new examples contain none. VERIFIED.

## 2. Dataset export compatibility (Phase A, the 16 questions)

Answers come from reading the code and, where marked, from the executed exports.

| # | Question | Answer |
|---|---|---|
| 1 | Input structure | `cwd/data/<dataset>/train/*.json`, one JSON record per file; dataset names from `config["datasets"]` (default `benchmark_create/fix/modify`). Absolute dataset paths also work (`os.path.join` discards the prefix); this was used for the exports. |
| 2 | Load both pinned HF revisions, or local JSONL? | The exporter reads only per-record `.json` files, not JSONL and not HF. The published JSONL (JSON fields stringified) must first be decoded with `published.decode_row` into per-record files. Done for the exports; the 3K was exported both from `data_published/<rev>/benchmark_*/train/*.json` and from the decoded `original_files/*/train.jsonl`, with identical output (SHA-256 equal). **VERIFIED.** |
| 3 | The three tasks | `create`, `fix`, `modify`, always in this order. |
| 4 | Chat conversion | `BaseDataTask.format_example`: the trailing empty assistant turn from the evaluation's own `format_for_llm` is replaced by the gold. Roles are exactly `system, user, assistant` in all 10,000 exported rows. **VERIFIED.** |
| 5 | Messages | One system message per task (1 distinct text per task, verified), one user prompt, one assistant target. |
| 6 | Targets | Create `reference_json`; Fix `valid_json` (never `erroneous_json`); Modify `ground_truth`. **VERIFIED** per row against the source record. |
| 7 | JSON serialization | Target: `json.dumps(gold)`, compact default separators `", "` / `": "`, `ensure_ascii=True` (non-ASCII becomes `\uXXXX`), single line. Prompt schema/data: `json.dumps(..., indent=2)`, `ensure_ascii=True`. Round trip of every target through `json.loads` equals the gold. **VERIFIED.** |
| 8 | `$ref` handling | Left unresolved and verbatim in the prompt; external refs are neither fetched nor inlined. The gold target is validated by the evaluator against the schema, so an unresolvable ref makes `schema_compliance` 0. |
| 9 | Test contamination | The exporter reads only `train/`. A probe export of the 450 test records (kept in a clearly named probe file) shares 0 lines with the 3K or 7K export. **VERIFIED.** Schema-level overlap: see section 3. |
| 10 | Shuffling | None. Order = task (create, fix, modify), then sorted file name. The trainer must shuffle; the shuffle seed is part of the missing training recipe. |
| 11 | Ordering | Deterministic. For decoded JSONL the file index follows the JSONL order, which in the 7K release is interleaved by example id (originals and new rows mixed, not "first 3000"). |
| 12 | Determinism | Two runs per input gave identical SHA-256. **VERIFIED.** |
| 13 | Seeds | The exporter has no randomness and no seed. |
| 14 | Filtering | Only silent skips: a missing dataset or directory, non-`.json` files, records without the gold key. No other filter. A malformed JSON file aborts the run (tested). |
| 15 | 3K vs 7K format differences | None in fields: identical keys per task (create 6, fix 7, modify 8 keys; see `readiness_results.json`). Content differs (section 5). |
| 16 | Row identity | `row_origin.tsv` (task, split, line, origin, line_sha256) labels each 7K row; the 3K rows are the rows labelled `original`. |

## 3. Source-to-export integrity (Phase B)

All numbers come from `outputs/training_export_readiness_20261118/` (ignored by git). Command and hashes are in `provenance.json`; the driver script is
`outputs/readiness_tools_20261118/run_readiness_export_v1.py`. Inputs: baseline revision `7962b0947f7d96eb3a0e6b9b928d939c7f3061a6`, Scale Only HF
commit `668a6773c6935e445cc04765b58e419ecdeebb0c` (local verified copy `release/scale-only-7k`, `check-release` PASSED).

| Export | Rows | SHA-256 |
|---|---|---|
| 3K from published JSON files | 3000 (1000/1000/1000) | `7f1e9a8cfff24ea75b14e42dca668ba35b1c5b66bee12fdd345091287472b945` |
| 3K from decoded published JSONL | 3000 | same |
| 7K from decoded release JSONL | 7000 (2334/2333/2333) | `679089b32d7fbf8cf10b31fe794cb52931653ad04e761ee687f27627b4cc1eab` |
| Test probe (never for training) | 450 | `6c903bcbf73926fa24eaacad088b16e8ceb5219a1cc6fbc120129828f044bab5` |

Per-row checks (all 10,000 training rows, against the independent source record): keys exactly `messages, task_type`; roles; no empty content;
target equals `json.dumps(gold)` and round-trips; schema, description, instructions and erroneous JSON appear verbatim in the prompt; the Fix prompt
does not contain the answer. Problems found: 0. Missing, repeated or filtered rows: 0 (counts equal the source, duplicates 0). Cross-task rows: 0
(`task_type` tally equals the task block).

Original-vs-expanded consistency: origin counts in the 7K release are create 1000 original + 1334 new, fix 1000 + 1333, modify 1000 + 1333. The
3000 rows labelled `original` equal the 3K export as a multiset. VERIFIED.

Contamination (executed): exact-row overlap with the test export 0 for 3K and 0 for 7K; Test files equal the baseline Test rows. Schema-content
overlap between the original train and the original test: 8 distinct schema contents (baseline property, not introduced by the expansion). New
examples whose schema content is in the Test split: 0. Note that the 3K and 7K runs both inherit those 8 shared schemas, so the comparison is not
affected differentially, but absolute Test scores slightly favour memorization on those schemas.

Target properties found in the source (measured with local `jsonschema` 3.2.0, version dependent):

| Task | Gold valid against own schema | Gold not validatable | Empty gold (`{}`/`[]`/null) |
|---|---|---|---|
| Create | 971 original, 1310 new | 29 original, 24 new | 0 |
| Fix | 985 original, 1312 new | 15 original, 21 new | 0 (17 new and 10 original have an empty erroneous input) |
| Modify | 988 original, 1304 new | 12 original, 29 new | 5 original, 13 new |

"Not validatable" means `jsonschema` raised something other than a validation error (unresolved ref, unsupported draft). No gold target failed a
validation outright in these data. The task brief expected Modify targets might violate the schema; none was found here, but the exporter does not
check it (a test pins that a violating Modify target would be exported as is). The empty `subset` counts in the raw rows are the `schemas` source,
which has no subset; this is not a missing field.

## 4. Prompt and target correctness

- Prompt formatters and `format_example` are unchanged since `e7f8b22`; evaluation files only gained metrics later (checked in the earlier phases).
- Training prompt vs `lm_eval_tasks/utils.py::doc_to_text_*`: identical text for create (with and without description), fix and modify. Tested.
- The lm-eval prompt is a single string (system and user joined by a blank line); with a chat template the model sees separate roles. The training
  data uses separate roles. Which form the original evaluation used for each model is a decoding detail the repository does not fix (`--apply_chat_template`
  is an lm-eval run option). **Risk, not a defect.** The evaluation of both models must use the same option.
- Targets: see section 3. The Create reference is the model-generated instance; Fix is the valid JSON; Modify is the modified JSON.

## 5. Length distributions (Phase C)

Unit: characters, a proxy. The tokenizer and base model are unconfirmed, so no token counts, no maximum length and no truncation estimate are given.
Source: `length_distribution.csv`. Prompt = system + user. Total = prompt + target.

| Task | Group | n | Mean total | Median | P90 | P99 | Max |
|---|---|---|---|---|---|---|---|
| create | original 3K | 1000 | 7,936 | 4,703 | 21,501 | 32,097 | 39,310 |
| create | new 4K | 1334 | 4,812 | 3,386 | 10,432 | 17,266 | 20,570 |
| create | combined 7K | 2334 | 6,151 | 3,833 | 14,258 | 29,074 | 39,310 |
| fix | original 3K | 1000 | 9,722 | 6,189 | 24,400 | 36,550 | 76,684 |
| fix | new 4K | 1333 | 6,627 | 5,100 | 14,146 | 19,899 | 40,421 |
| fix | combined 7K | 2333 | 7,954 | 5,405 | 17,677 | 31,901 | 76,684 |
| modify | original 3K | 1000 | 7,967 | 4,642 | 21,630 | 31,286 | 45,463 |
| modify | new 4K | 1333 | 6,120 | 4,236 | 14,445 | 22,121 | 34,020 |
| modify | combined 7K | 2333 | 6,912 | 4,365 | 16,905 | 28,529 | 45,463 |

P75, P95 and the prompt-only and target-only rows are in the CSV. Target means (chars): create 706 / 435 / 551, fix 796 / 577 / 671, modify 688 / 571 / 621
(original / new / combined).

Observations:

- The embedded schema is about 62 to 74 percent of the prompt on average, in every group. Prompt length is therefore driven by the schema, and the new
  schemas are smaller (mean schema 3.3K to 4.4K chars vs 5.9K to 6.4K for the originals; P99 14K to 17K vs 27K to 32K).
- Longest examples are all in the original 3K: the longest Fix example is 76,684 chars (`Github_hard`), about 2x the longest new example (40,421). The 7K
  set has the same few very long originals plus a shorter bulk.
- External refs (a `$ref` not starting with `#`): create 42 original / 38 new, fix 23 / 29, modify 22 / 44 examples. Embedded `definitions` are kept in
  the prompt.
- Effect of the lower complexity: the 7K set has about 2.3x the rows but only about 1.9x the characters of the 3K (35.0M vs 18.65M). The mean example
  is about 20 percent shorter. The distribution shift is in the schema size, the exact cause is intentionally not investigated here.
- No choice of maximum sequence length is proposed here.

## 6. Evaluation readiness (Phase D)

Inspected (code), not executed (no inference allowed):

- Harness: `lm_eval_tasks/` (lm-evaluation-harness), tasks `benchson_create`, `benchson_fix`, `benchson_modify`, groups `benchson` and `benchson_hard`
  (Github_hard, Github_medium, schemas for each task, nine subtasks); per-tier subtasks in `lm_eval_tasks/tiers/`. `BENCHMARK.md` states that the hard
  group is the headline and that `exact_match` should lead the report. Confirmed from the files; `benchson_hard` is primary.
- Data: `aviv1ron1/Benchson`, `test_split: test`, so evaluation reads the Test split only. For a pinned run the dataset revision must be given
  explicitly (the YAML has no revision). READY BUT NOT EXECUTED.
- Decoding: `output_type: generate_until`, `do_sample: false`, `temperature: 0.0`, `max_gen_toks: 2048`, `until: ["\n\n\n\n"]`. No constrained decoding.
  Generation length 2048 tokens is a risk for the longest targets only if tokens per char were low; target means are under 800 chars (proxy), so this is
  not expected to bind, but it is not verified without the tokenizer.
- Prompt: single joined string; the same function family as training (section 4).
- Independence from training: the evaluation reads only the Test split of the published dataset and has no code path that reads `train/`; the exporter
  never reads `test/`. Both are inspected and pinned by tests; there is no runtime guard. Test must not be used for training, model selection or
  checkpoint selection: nothing in the repository enforces it, so the protocol must (section 9, G).
- Environment risk: the metrics call `jsonschema.validate`, and the validity of some schemas depends on the installed version (local 3.2.0; an earlier
  cluster check used 4.26.0). The evaluation environment of both models must pin the same version.

Metric applicability matrix (from `utils.py::_score`):

| Metric | Create | Fix | Modify | Definition |
|---|---|---|---|---|
| json_validity | yes | yes | yes | output parses as JSON (code fences tolerated) |
| schema_compliance | yes | yes | yes | parsed output validates against the record schema; unresolvable schema counts as 0 |
| semantic_fidelity | yes | yes | yes | share of reference leaf values recovered by value |
| exact_match | yes | yes | yes | valid, compliant and fidelity 1.0 |
| change_fidelity | no | yes | yes | share of changed leaf paths (input vs reference) that the output gets right |

Because `exact_match` requires schema compliance, an unvalidatable schema (about 1.2 to 2.9 percent of rows by the local check) caps it at 0 for that
row, for both models alike.

## 7. JSONSchemaBench readiness (Phase E)

- `src/import_jsonschemabench.py` imports schemas from `epfl-dlab/JSONSchemaBench` (MIT) as Benchson source schemas, subsets Github_easy/medium/hard,
  Kubernetes, Snowplow, Glaiveai2K, default split `test`, no immutable revision, random sampling by seed. It is a data importer, not an evaluator.
- Benchson's `train` schemas therefore partly derive from JSONSchemaBench (tiers visible in the train rows: Github_easy/medium/hard, Kubernetes, Snowplow,
  Glaiveai2K; 1,630 of 2,334 create rows in the 7K set carry a JSONSchemaBench subset, the other 704 are `schemas`). Evaluating a model trained on them on the matching JSONSchemaBench
  schemas is **not** contamination-free unless the shared schemas are removed. Not checked against the JSONSchemaBench data (inaccessible).
- The repository contains no JSONSchemaBench evaluation implementation, no constrained-decoding setup, no metric definitions (the benchmark's native
  metrics are coverage and compliance, not Benchson's four), no pinned revision and no record of how the researchers evaluated. Comparable tasks:
  only schema-conditioned generation (Create) is conceptually close; Fix and Modify have no JSONSchemaBench counterpart.
- **BLOCKED**: whether the researchers evaluated on JSONSchemaBench, and how, is unknown. No protocol was invented. Needed from the mentors: the
  evaluation procedure and revision, if any.

## 8. BGU feasibility (Phase F)

The cluster was not accessible. Nothing about it is VERIFIED here. Known from repository files and earlier phases (secondary): partition `main`, account
`4th-year-proj`, GPU `rtx_pro_6000` (about 96 GB, from the specification), `--mem=32G`, `--cpus-per-task=8`, `--tmp=100G`, time limit 3 h in the smoke script.
The `hf` and `vllm` dependency groups conflict (`pyproject.toml`), and neither lists `peft` or `trl`; training therefore needs its own environment,
to be created later and not now.

Read-only command set to run on BGU (READY BUT NOT EXECUTED, nothing writes, nothing is submitted):

```bash
sinfo -o "%P %a %l %D %G %c %m" ; scontrol show partition main
sacctmgr -n show assoc user=$USER format=account,partition,qos,maxjobs,grptres%40 ; sshare -U 2>/dev/null | head
squeue -u $USER ; sacct -u $USER -S 2026-11-01 -o jobid,jobname,state,elapsed,maxrss,reqmem | tail -20
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # only inside an allocation already granted
df -h $HOME /tmp ; quota -s 2>/dev/null ; du -sh ~/.cache/huggingface 2>/dev/null
module avail 2>&1 | head -40 ; conda env list 2>/dev/null ; ls ~/.venv 2>/dev/null
python3 --version ; pip list 2>/dev/null | grep -i -E "^(torch|transformers|peft|trl|vllm|datasets|accelerate|jsonschema|lm.eval)"
```

Resource estimate (assumptions stated, not measured): LoRA on a model of about 3B parameters needs roughly 8 to 12 GB for weights in bf16 plus activations;
full fine-tuning needs roughly 48 GB before activations; either fits one 96 GB card. Sequence length drives activation memory; with the longest original
examples at about 77K characters (maybe 25K tokens at 3 chars per token, unmeasured) gradient checkpointing and a length policy will matter. GPU time
cannot be estimated without throughput, which needs a measured run.

## 9. Existing tests, missing safeguards and the two plans (Phases G, H)

Existing tests covering these areas: before this audit none exercised `export_training.py` or the prompt/target mapping (`grep` over `tests/` finds no
reference to it). `test_standard_equivalence.py` checks generation prompts, `test_hf_baseline.py` and `test_publish_scale.py` cover publication and
overlap. Added: `tests/test_export_training_readiness.py` (22 tests, offline, synthetic records, no exporter change): structure and roles, target field
per task, compact serialization and escaping, unresolved `$ref`, Modify target not schema-checked, null gold, ordering without shuffle, byte-identical
reruns, pinned silent skips, abort on malformed JSON, Test directory never read, prompt equality with lm-eval text for all three tasks, target equality,
published-format round trip, metric agreement between the two implementations, `exact_match` and `change_fidelity` rules, and one real-data test that the
original rows in the 7K export equal the 3K export (skipped when the untracked local release is absent).

Missing safeguards (not implemented, because exporter changes need approval):

1. Exporter prints counts but never compares them to the source and never fails on a skipped record; add a `--expect-counts` option or a post-export verifier.
2. No hash or manifest output from the exporter; the driver script here covers it ad hoc.
3. No guard that the output is not inside `data/*/test`; no `train_manifest` tying an export to a dataset revision.
4. No metric-parity check between `lm_eval_tasks/utils.py` and `src/evaluations/metrics.py` outside these tests.
5. No pin for `jsonschema` in the evaluation environment.

Plan A (faithful reproduction). Prerequisite: the mentors supply the original training pipeline, data identity and base-model details
(`docs/original_training_reproduction.md`). Missing inputs are not filled with defaults. Status: **BLOCKED**. Once it exists, apply the same recipe to 7K
unchanged, with only the data and the budget policy (below) as the declared variable.

Plan B (independent comparison). A new, documented, identical recipe for both datasets. This invents a recipe, so it needs the user's explicit approval
and is not designed here. What it must fix in advance, not defined here: one model and tokenizer, one prompt format (the exporter's), one maximum length
policy, one optimizer and schedule, one decoding and Test set, one metric set, one checkpoint policy chosen without Test.

Budget analysis (facts only):

| Policy | 3K | 7K | Consequence |
|---|---|---|---|
| Fixed epochs (same epochs) | 3000 x E examples, about 18.65M x E chars | 7000 x E, about 35.0M x E chars | 7K sees about 2.3x the examples, about 1.9x the text, so it also gets about 2.3x the optimizer updates at equal batch size |
| Fixed updates (same steps) | E epochs | about 0.43 E epochs | equal updates, 7K sees each example less often |
| Fixed token exposure | needs tokenized lengths | needs tokenized lengths | needs the tokenizer |

The mean example of the new data is about 20 percent shorter, so "same number of examples" and "same amount of text" are different controls. Which one
the original authors used is unknown. Task balance is nearly equal in both (3K exactly 1/3 each; 7K 2334/2333/2333). Tier mix differs: Github_hard
share of train rows is about 19 percent in the original 3K and about 18 percent in 7K create; `schemas` share rises (create 316 to 704 of 2334, fix 250 to 618,
modify 145 to 537), so `schemas`, which is in `benchson_hard`, is better represented in 7K. Sampling imbalance across tiers is thus a second difference besides length.

Repeated-seed protocol (to be run later): at least 3 training seeds per dataset with identical data order policy, the same set of evaluation seeds
(greedy decoding needs none), report mean and spread per metric on `benchson_hard` and on `benchson`, per-task and per-tier, and compare with the
spread, not single runs. The number of seeds is a budget decision for the user.

## 10. Reproducibility checklist (Phase I)

| Area | Item | Status |
|---|---|---|
| Dataset | 3K export complete, deterministic | VERIFIED |
| Dataset | 7K export complete, deterministic, originals equal the 3K | VERIFIED |
| Dataset | No Test row in either train export | VERIFIED (row level) |
| Dataset | Exact identity of the original 3K the authors used (3000 HF vs 2950 git vs "about 2K") | BLOCKED |
| Dataset | Input revisions and hashes recorded | VERIFIED (`provenance.json`) |
| Model | Base model, tokenizer, revision | BLOCKED (granite-4.1-3b is a secondary claim) |
| Training | Recipe (LoRA r16/alpha32 claimed, all else unknown), budget definition, seeds, max length, packing | BLOCKED |
| Training | Training environment on BGU | READY BUT NOT EXECUTED (commands in section 8; env not created) |
| Compute | Partition, GPU memory, quota, free disk | READY BUT NOT EXECUTED (cluster inaccessible) |
| Compute | Throughput and GPU time | NOT APPLICABLE until a measured run |
| Evaluation | Prompt equals training prompt, metrics agree between the two code paths | VERIFIED (tests) |
| Evaluation | Run on both models, same revision, same jsonschema | READY BUT NOT EXECUTED |
| Evaluation | Test never used for training or selection | inspected only; protocol must enforce it |
| Evaluation | JSONSchemaBench protocol | BLOCKED |
| Tests | 22 new tests, relevant existing tests, full suite by groups | VERIFIED (counts in the final report) |

## 11. Remaining blockers

Training-specific, for the mentors: the original training script and config, base model and revision, tokenizer and chat template, maximum length and
truncation policy, batch size, learning rate and schedule, epochs or steps, seed, LoRA target modules, checkpoint selection rule, the exact original
3K identity, and whether and how JSONSchemaBench was used for evaluation.

Other: BGU access for the read-only inspection; the approval to design Plan B; the choice of budget control; the seed count.

## 12. Recommended next actions

1. Send the mentors the list in section 11 (one message).
2. When BGU is reachable, run the read-only command set in section 8 and record the outputs.
3. Decide, with the mentors' answer, between Plan A and Plan B; if Plan B, approve a documented recipe before any run.
4. Approve (or decline) the small exporter safeguards in section 9, items 1 and 2.
5. Pin the evaluation environment, including `jsonschema`, and the dataset revisions before any evaluation.
6. Only then create the training environment and start with a throughput measurement, not a full run.
