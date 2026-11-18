# Training guide: reproduce the original LoRA and aLoRA, then train on 7K

This folder has the code and reference results of the original Benchson adapter runs
(June 2026): a **LoRA** ("v2 LoRA") and an **aLoRA**. Use it to train a 3K adapter and a 7K
adapter **the same way**, and compare them on the same test set. Do this for LoRA, aLoRA, or both.

```
original_training/
├── README.md                       this guide
├── scripts/
│   ├── prepare_benchson_data.py    HF dataset -> chat JSONL (one row per example)
│   ├── train_lora.py               the LoRA trainer (PEFT + HF Trainer)
│   └── train_alora.py              the aLoRA trainer (same code, aLoRA adapter)
└── reference/                      the original runs, for comparison only
    ├── train_log_v2.txt            LoRA: loss per step and per epoch
    ├── results_base.json           lm-eval scores, base model (no adapter)
    ├── results_v2.json             lm-eval scores, v2 LoRA
    └── results_alora.json          lm-eval scores, aLoRA
```

`prepare_benchson_data.py` and `train_lora.py` are the original scripts with one small change
each:

- `prepare_benchson_data.py`: the dataset id is a required `--dataset` argument instead of a
  hardcoded value. We checked: its output today is identical to the original training file
  (all 3000 rows).
- `train_lora.py`: `warmup_ratio=0.05` became `warmup_steps=0.05`. Same behavior (a float under 1
  means "fraction of total steps", and the original run saved it as `warmup_steps: 0.05`), but it
  also works on transformers 5.15 and newer, which removed `warmup_ratio`.

`train_alora.py` is new (see "LoRA vs aLoRA" below).

## The plan

```
           ┌─ 3K data ─┐                          ┌─ eval ─> scores 3K ─┐
 prepare ──┤           ├── shuffle ── train ──────┤                     ├── compare
           └─ 7K data ─┘   + filter   (same       └─ eval ─> scores 7K ─┘     + base
                                       settings)
```

Only the training data changes between the two runs. Everything else stays the same.

## The original results

`exact_match`, mean over the 9 hard tiers (Github_medium, Github_hard and schemas, for each
of create / fix / modify). Computed from the `results_*.json` files in `reference/`.

| Run | mean | create | fix | modify |
|---|---|---|---|---|
| base `granite-4.1-3b`, no adapter | 0.545 | 0.327 | 0.649 | 0.659 |
| v2 LoRA (3K) | 0.684 | 0.474 | 0.772 | 0.806 |
| aLoRA (3K) | 0.639 | 0.401 | 0.723 | 0.792 |

## The original settings

All of these are hardcoded in the scripts or are their defaults. The original runs used the
defaults.

| Setting | LoRA (`train_lora.py`) | aLoRA (`train_alora.py`) |
|---|---|---|
| Base model | `ibm-granite/granite-4.1-3b`, revision `c0650403e44e78ec0262dab1c90914c65b196c4e` | same |
| Adapter | r=16, alpha=32, dropout 0.05, on `q,k,v,o,gate,up,down_proj` | r=32, alpha=64, dropout 0.05, on `q,k,v,o_proj` only |
| Switches on | for every token | only after `<\|start_of_role\|>assistant<\|end_of_role\|>` |
| Trainable params | 31,129,600 | 20,971,520 |
| Optimizer | AdamW, lr 2e-4, cosine schedule, warmup 5%, no weight decay | same, but weight decay 0.01 |
| Batch | 2 per GPU x 8 accumulation = 16, on 1 GPU (A100 80GB) | 1 x 8 = 8 |
| Length | 2048 tokens max. **36% of the 3K rows are longer** (see Step 3) | same |
| Epochs | 3, keeps the checkpoint with the lowest eval loss | 2, keeps the last checkpoint |
| Loss | only on the answer (everything up to the assistant turn is masked) | same |
| Seed | 42 (Trainer default) | 2 |
| Time for 3K, one A100 | about 48 minutes (`train_runtime` 2851 s) | about 50 minutes (3026 s) |
| Versions (June) | PEFT 0.19.1. Eval: transformers 5.9.0, lm-eval 0.4.10 | same |

## LoRA vs aLoRA

Both adapters learn the same task. The difference is **when** the adapter is on:

```
tokens:   <system> <user: schema + task ...> <|start_of_role|>assistant<|end_of_role|> {"a": 1 ...
LoRA:     [adapter on ................................................................ on]
aLoRA:    [base model only ...........................][adapter on .................... on]
                                                       ^ invocation tokens
```

With aLoRA, the prompt is read by the plain base model, and the adapter only changes the
answer. So in a multi-step system, the prompt's KV cache can be shared with the base model.
In June, LoRA scored higher (0.684 vs 0.639).

`train_alora.py` is the same code as `train_lora.py`. Only the adapter config and the defaults
differ (see the table above, or `diff train_lora.py train_alora.py`).

The June aLoRA was trained with a different, internal trainer. `train_alora.py` reproduces its
adapter config (we checked: the same rank, targets and invocation tokens) and its Trainer
settings with plain PEFT. But it is not the same code, so expect results close to 0.639, not
identical. Do not compare your aLoRA loss values with June's. The internal trainer reported
them differently.

## Step 1: Set up

Use a separate environment. The benchson `hf` group has no `peft`, and it conflicts with
the `vllm` group.

```bash
python -m venv .venv-train && source .venv-train/bin/activate
pip install torch transformers peft datasets accelerate lm-eval jsonschema deepdiff
```

The latest versions work. We tested the whole pipeline (data prep, both trainers, lm-eval
with both adapters) on one A100 with:

| Package | Tested version | Original run (June) |
|---|---|---|
| torch | 2.15.0 | 2.6.0 |
| transformers | 5.23.0 | 5.9.0 |
| peft | 0.21.3 | 0.19.1 |
| datasets | 5.2.0 | - |
| accelerate | 1.16.0 | - |
| lm-eval | 0.4.13 | 0.4.10 |

The scripts also work with the June versions. Do not mix: if you pin old versions, pin all of
them. Note: `peft` must be 0.17 or newer for aLoRA.

Write down the exact versions you got (`pip freeze > versions.txt`). Use the same environment
for both runs.

## Step 2: Prepare the data

Run from the root of a benchson checkout. `PYTHONPATH=.:src` is needed: benchson's code imports
both `src.…` and plain module names. The script loads the `train`
split of each task and builds the prompts with benchson's own `format_for_llm`. So the training
prompts match the eval prompts exactly. It never reads the `test` split.

Replace the placeholders with your own values:

| Placeholder | Meaning |
|---|---|
| `<3K_DATASET_ID>`, `<3K_REVISION>` | the HF dataset repo and pinned revision of the original 3K data (the original run used revision `main`, unpinned) |
| `<7K_DATASET_ID>`, `<7K_REVISION>` | the HF dataset repo and branch/revision of your 7K data |

**3K (the original data):**

```bash
PYTHONPATH=.:src python original_training/scripts/prepare_benchson_data.py \
    --dataset <3K_DATASET_ID> --revision <3K_REVISION> --output data/train_3k.jsonl
```

**7K (your data):**

```bash
PYTHONPATH=.:src python original_training/scripts/prepare_benchson_data.py \
    --dataset <7K_DATASET_ID> --revision <7K_REVISION> --output data/train_7k.jsonl
```

Check the counts it prints (`[create] wrote N examples`) against what you expect.

## Step 3: Clean and shuffle (both datasets, the same way)

**Remove leakage.** Drop train rows whose schema also appears in **any** test split. Use your
A1 leakage script. The 7K includes the original 3K rows, so it has the same leaking rows unless
you filter it. Only filter train. Never touch test.

**Shuffle.** `train_lora.py` uses the **last 10%** of the file as validation, without
shuffling. The file is written create, then fix, then modify. So without a shuffle, the
validation set is only `modify`:

```
train_3k.jsonl:  [ create 1000 | fix 1000 | modify 1000 ]
                                           '-- last 300 --' = validation, all modify
```

The original run had this problem: modify trained on only 700 rows, and the best checkpoint was
picked by modify-only loss. Shuffle with a fixed seed before training:

```bash
python -c "
import random, sys
for f in sys.argv[1:]:
    rows = open(f).readlines(); random.Random(0).shuffle(rows)
    open(f.replace('.jsonl', '_shuf.jsonl'), 'w').writelines(rows)
" data/train_3k.jsonl data/train_7k.jsonl
```

**Check the length.** Examples over `--max_seq_length` (default 2048) are cut from the end.
If the prompt alone is that long, the whole answer is cut, every label is masked, and the
example teaches nothing. This hit the original run hard. We measured the original 3K file:

| Original 3K rows | create | fix | modify | total |
|---|---|---|---|---|
| fit in 2048 tokens | 687 | 576 | 658 | 1921 (64%) |
| answer partly cut | 48 | 51 | 57 | 156 (5%) |
| answer fully cut, nothing learned | 265 | 373 | 285 | **923 (31%)** |

How many rows fit at each limit (median 1334 tokens, 90th percentile 5315, 99th 7805):

| `--max_seq_length` | 2048 | 4096 | 6144 | 8192 |
|---|---|---|---|---|
| rows that fit | 1921 (64%) | 2483 (83%) | 2849 (95%) | 2983 (99.4%) |

The eval is not cut (lm-eval allows the model's full context), so the original run trained on
shorter inputs than it was tested on.

Recommendation: use `--max_seq_length 8192` for **both** arms. If you want to reproduce the
original setup exactly, keep 2048, but then also report it as a limitation. Longer sequences
need more GPU memory: if you run out, use `--per_device_train_batch_size 1
--gradient_accumulation_steps 16` (LoRA) to keep the same effective batch. We did not test 8192
for memory.

Count the rows over the limit in your files:

```bash
python -c "
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('ibm-granite/granite-4.1-3b')
for f in sys.argv[1:]:
    n = [len(tok(tok.apply_chat_template(json.loads(l)['messages'], tokenize=False)).input_ids) for l in open(f)]
    print(f, 'rows:', len(n), 'over 2048:', sum(x > 2048 for x in n), 'over 8192:', sum(x > 8192 for x in n))
" data/train_3k_shuf.jsonl data/train_7k_shuf.jsonl
```

Whatever limit you choose, use the same one for both arms.

## Step 4: Train

The same command for both. Only the data path and the output dir change.

```bash
python original_training/scripts/train_lora.py \
    --data_path data/train_3k_shuf.jsonl --output_dir adapters/lora-3k
python original_training/scripts/train_lora.py \
    --data_path data/train_7k_shuf.jsonl --output_dir adapters/lora-7k
```

**aLoRA:** the same, with `train_alora.py`:

```bash
python original_training/scripts/train_alora.py \
    --data_path data/train_3k_shuf.jsonl --output_dir adapters/alora-3k
python original_training/scripts/train_alora.py \
    --data_path data/train_7k_shuf.jsonl --output_dir adapters/alora-7k
```

Check that `adapters/alora-3k/adapter_config.json` has
`"alora_invocation_tokens": [100264, 78191, 100265]`. If it is `null`, the adapter is a plain
LoRA, and your PEFT version is too old.

Sanity check (LoRA): for 3K, the loss should fall to about 0.02 within the first epoch, and the
eval loss should be well under 0.01 by epoch 2. Compare with `reference/train_log_v2.txt`.

The 7K run takes about 2.3 times more steps (same number of epochs). Write this down when you
report results.

## Step 5: Evaluate

Run the base model once, then each adapter. Same command, same test set. Run from the root of
the benchson checkout. `PYTHONPATH=lm_eval_tasks` is needed with lm-eval 0.4.13 (it no longer
finds the tasks' `utils.py` on its own; 0.4.10 did).

```bash
export PYTHONPATH=lm_eval_tasks
lm_eval --model hf --model_args pretrained=ibm-granite/granite-4.1-3b \
  --tasks benchson_tiers --include_path lm_eval_tasks --apply_chat_template \
  --log_samples --output_path results/base

lm_eval --model hf --model_args pretrained=ibm-granite/granite-4.1-3b,peft=adapters/lora-3k \
  --tasks benchson_tiers --include_path lm_eval_tasks --apply_chat_template \
  --log_samples --output_path results/lora-3k
# ... and the same for adapters/lora-7k, adapters/alora-3k, adapters/alora-7k
# (aLoRA needs nothing extra: --apply_chat_template adds the invocation tokens)
```

## Step 6: Compare

Report `exact_match` on the 9 hard tiers, per task, like the table above. For fix and modify,
also report `change_fidelity`. Put your base run next to `reference/results_base.json`. They
should be close. If they are not, your setup differs from the original. Fix that before you
compare 3K with 7K.

Your 3K run will not match the original scores (0.684 LoRA, 0.639 aLoRA) exactly, because you shuffle and filter. That is
expected. The fair comparison is **your 3K vs your 7K**, not your 7K vs the original.

Run more than one seed if you can (for example 3 per dataset), and report the mean and spread.
The test set has only 150 rows per task, so small differences can be noise.

## Checklist

- [ ] One environment, versions written down
- [ ] Same base model revision for both runs
- [ ] 3K revision pinned
- [ ] Leakage filter applied to both, test untouched
- [ ] Both files shuffled with the same seed
- [ ] Truncation counted for both
- [ ] Same training command for 3K and 7K (`train_lora.py` or `train_alora.py`)
- [ ] Same `--max_seq_length` for both (8192 recommended)
- [ ] Base model evaluated and close to `reference/results_base.json`
