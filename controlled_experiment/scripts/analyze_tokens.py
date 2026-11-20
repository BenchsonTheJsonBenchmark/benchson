#!/usr/bin/env python3
"""Exact token-length / truncation / loss-masking analysis with the ORIGINAL Granite 4.1 3B tokenizer (CPU only).

Standalone: needs only the Python stdlib + `transformers` (+ `tokenizers`). It loads NO model weights and needs no torch.
`tokenize_and_mask` below is a verbatim copy of the function in original_training/scripts/train_lora.py (a unit test asserts
the two source texts are identical); tokenizer setup (pad=eos, right padding, response-template ids) copies train_lora.py main().

Inputs (one directory, `--inputs`), all verified by sha256 before use:
  prepare_original_3k.jsonl   output of the ORIGINAL prepare_benchson_data.py on the 3K baseline
  prepare_release_7k.jsonl    same, on the Scale-Only 7K release (origin of each row from row_origin.tsv)
  row_origin.tsv              release/scale-only-7k/row_origin.tsv
  train_A_3k.jsonl, train_B_7k.jsonl, validation.jsonl   the controlled split (split_v1)
Nothing is modified; outputs are written with exclusive-create ("x") into `--out`.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from collections import Counter

MODEL = "ibm-granite/granite-4.1-3b"
REVISION = "c0650403e44e78ec0262dab1c90914c65b196c4e"
MAX_LEN = 2048
RESPONSE_TEMPLATE = "<|start_of_role|>assistant<|end_of_role|>"
TASKS = ("create", "fix", "modify")
EXPECTED_SHA256 = {
    "prepare_original_3k.jsonl": "7f1e9a8cfff24ea75b14e42dca668ba35b1c5b66bee12fdd345091287472b945",
    "prepare_release_7k.jsonl": "2dde3ce42bfe52cda108965f802e8bcb2ba7e61afc7a8ec89656ae3c6446804d",
    "row_origin.tsv": "0bd864e3a6f431f0c515d0c6d09b4ff5e5ad9b552dc17929b91a2723c54cad4c",
    "train_A_3k.jsonl": "af79981c10e721d4ef3860022b3e21cc9b3eae20fb807a6f0948da7ec48c26ec",
    "train_B_7k.jsonl": "e8b66d9bed0c2bc909f4eec8a6a17e542219966a3943a493cf198f9dd4800141",
    "validation.jsonl": "194139cd735c4987dca6dc46dc46799e683d20fc18803b655f2293390018045b",
}
EXPECTED_COUNTS = {"prepare_original_3k.jsonl": 3000, "prepare_release_7k.jsonl": 7000, "train_A_3k.jsonl": 2684,
                   "train_B_7k.jsonl": 6267, "validation.jsonl": 300}


# --- verbatim copy of original_training/scripts/train_lora.py::tokenize_and_mask -------------------------------------
def tokenize_and_mask(example, tokenizer, max_length, response_template_ids):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = tokenized["input_ids"]
    labels = list(input_ids)

    # Mask everything before (and including) the last response template occurrence
    tlen = len(response_template_ids)
    last_match = -1
    for i in range(len(input_ids) - tlen + 1):
        if input_ids[i:i + tlen] == response_template_ids:
            last_match = i + tlen
    if last_match == -1:
        labels = [-100] * len(input_ids)
    else:
        labels[:last_match] = [-100] * last_match

    return {"input_ids": input_ids, "attention_mask": tokenized["attention_mask"], "labels": labels}
# ---------------------------------------------------------------------------------------------------------------------


def last_template_end(ids, template_ids):
    """End index of the last response-template occurrence (same scan as the original), or -1."""
    tlen, last = len(template_ids), -1
    for i in range(len(ids) - tlen + 1):
        if ids[i:i + tlen] == template_ids:
            last = i + tlen
    return last


def measure(line, tokenizer, template_ids, max_len):
    """Per-example record. Truncated numbers come from the ORIGINAL function; the untruncated tokenization (same call with
    truncation off) provides the full prompt/target lengths."""
    example = json.loads(line)
    out = tokenize_and_mask(example, tokenizer, max_len, template_ids)
    ids, labels = out["input_ids"], out["labels"]
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(text, truncation=False, padding=False)["input_ids"]
    full_end = last_template_end(full_ids, template_ids)
    retained = sum(1 for x in labels if x != -100)
    rec = {
        "task": example["task_type"], "seq_len": len(ids), "full_len": len(full_ids),
        "prompt_full": full_end if full_end != -1 else None,
        "target_full": (len(full_ids) - full_end) if full_end != -1 else None,
        "template_in_truncated": last_template_end(ids, template_ids) != -1,
        "retained": retained,
    }
    rec["class"] = classify(rec)
    return rec


def classify(r):
    if r["target_full"] is None:
        return "anomaly_no_template_untruncated"
    truncated = r["seq_len"] < r["full_len"]
    if not truncated:
        return "complete" if r["retained"] == r["target_full"] and r["retained"] > 0 else "anomaly_untruncated"
    if r["retained"] == 0:
        return "fully_masked_template_cut" if not r["template_in_truncated"] else "fully_masked_no_target_tokens"
    return "partial" if r["retained"] < r["target_full"] else "anomaly_truncated_but_complete"


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def dist(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return {"mean": round(statistics.fmean(v), 2), "median": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99), "max": v[-1]}


def summarize(recs):
    n = len(recs)
    c = Counter(r["class"] for r in recs)
    trunc = [r for r in recs if r["seq_len"] < r["full_len"]]
    lost = sum(r["target_full"] - r["retained"] for r in recs if r["target_full"] is not None and r["seq_len"] < r["full_len"])
    part = [r for r in recs if r["class"] == "partial"]
    return {
        "n": n,
        "classes": dict(sorted(c.items())),
        "complete_pct": round(100 * c["complete"] / n, 2) if n else None,
        "partial_pct": round(100 * c["partial"] / n, 2) if n else None,
        "fully_masked_pct": round(100 * (c["fully_masked_template_cut"] + c["fully_masked_no_target_tokens"]) / n, 2) if n else None,
        "any_truncated": len(trunc),
        "partial_lost_1_token": sum(1 for r in part if r["target_full"] - r["retained"] == 1),
        "supervised_tokens_retained": sum(r["retained"] for r in recs),
        "supervised_tokens_full_target": sum(r["target_full"] or 0 for r in recs),
        "supervised_tokens_lost_to_truncation": lost,
        "seq_len_after_truncation": dist([r["seq_len"] for r in recs]),
        "seq_len_untruncated": dist([r["full_len"] for r in recs]),
        "prompt_len_untruncated": dist([r["prompt_full"] for r in recs]),
        "target_len_untruncated": dist([r["target_full"] for r in recs]),
        "retained_per_example": dist([r["retained"] for r in recs]),
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path):
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def verify_inputs(d):
    shas, counts = {}, {}
    for name, want in EXPECTED_SHA256.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing input: {p}")
        shas[name] = sha256_file(p)
        if shas[name] != want:
            raise SystemExit(f"sha256 mismatch for {name}: {shas[name]} != expected {want}")
    for name, want in EXPECTED_COUNTS.items():
        counts[name] = len(read_lines(os.path.join(d, name)))
        if counts[name] != want:
            raise SystemExit(f"row count mismatch for {name}: {counts[name]} != {want}")
    return shas, counts


def tokenizer_provenance(tok, model, revision):
    prov = {"model": model, "requested_revision": revision, "tokenizer_class": type(tok).__name__,
            "vocab_size": len(tok), "eos_token": tok.eos_token, "chat_template_sha256":
            hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest(),
            "chat_template_chars": len(tok.chat_template or "")}
    try:
        import transformers, tokenizers
        prov["transformers"], prov["tokenizers"] = transformers.__version__, tokenizers.__version__
    except Exception:
        pass
    files = {}
    try:
        from huggingface_hub import try_to_load_from_cache
        for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
                   "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"):
            p = try_to_load_from_cache(model, fn, revision=revision)
            if isinstance(p, str) and os.path.isfile(p):
                files[fn] = {"sha256": sha256_file(p), "snapshot_dir": os.path.basename(os.path.dirname(p))}
    except Exception as e:  # provenance only; never blocks the analysis
        prov["file_hash_error"] = repr(e)
    prov["files"] = files
    snaps = {v["snapshot_dir"] for v in files.values()}
    prov["snapshot_dirs_seen"] = sorted(snaps)
    prov["revision_verified_by_snapshot_dir"] = bool(snaps) and snaps == {revision}
    return prov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="directory with the six verified input files")
    ap.add_argument("--out", required=True, help="output directory (created; files are never overwritten)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--online", action="store_true", help="allow Hugging Face network access (default: offline, cache only)")
    ap.add_argument("--verify-determinism", action="store_true", help="tokenize everything a second time and compare digests")
    ap.add_argument("--per-example", action="store_true", help="also write per_example.tsv")
    a = ap.parse_args()
    if not a.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.makedirs(a.out, exist_ok=True)

    shas, counts = verify_inputs(a.inputs)
    p3 = read_lines(os.path.join(a.inputs, "prepare_original_3k.jsonl"))
    p7 = read_lines(os.path.join(a.inputs, "prepare_release_7k.jsonl"))
    arm_a = read_lines(os.path.join(a.inputs, "train_A_3k.jsonl"))
    arm_b = read_lines(os.path.join(a.inputs, "train_B_7k.jsonl"))
    val = read_lines(os.path.join(a.inputs, "validation.jsonl"))

    # origin of each of the 7000 release rows: converter order is create->fix->modify, each in release file order
    with open(os.path.join(a.inputs, "row_origin.tsv"), newline="") as f:
        tsv = [r for r in csv.DictReader(f, delimiter="\t") if r["split"] == "train"]
    origin = []
    for t in TASKS:
        rows = sorted((int(r["line"]), r["origin"]) for r in tsv if r["task"] == t)
        assert [x[0] for x in rows] == list(range(1, len(rows) + 1))
        origin += [o for _, o in rows]
    assert len(origin) == len(p7) == 7000
    tasks7 = [json.loads(l)["task_type"] for l in p7]
    assert tasks7 == sorted(tasks7, key=TASKS.index), "release render is not create->fix->modify ordered"
    orig_rows_in_7k = [l for l, o in zip(p7, origin) if o == "original"]
    new_rows = [l for l, o in zip(p7, origin) if o == "new"]
    integrity = {
        "origin_counts": dict(Counter(origin)),
        "original_rows_of_7k_identical_to_3k_render": orig_rows_in_7k == p3,
        "arm_A_subset_of_3k_render": not (Counter(arm_a) - Counter(p3)),
        "arm_B_subset_of_7k_render": not (Counter(arm_b) - Counter(p7)),
        "arm_A_subset_of_arm_B": not (Counter(arm_a) - Counter(arm_b)),
        "validation_subset_of_3k_render": not (Counter(val) - Counter(p3)),
        "validation_disjoint_from_arms": not (set(val) & (set(arm_a) | set(arm_b))),
    }
    if not all(integrity[k] for k in integrity if k != "origin_counts") or integrity["origin_counts"] != {"original": 3000, "new": 4000}:
        raise SystemExit(f"input integrity check failed: {integrity}")

    from transformers import AutoTokenizer  # imported late so --help works anywhere
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tok.pad_token = tok.eos_token  # as in train_lora.py main()
    tok.padding_side = "right"
    template_ids = tok.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    prov = tokenizer_provenance(tok, a.model, a.revision)
    prov.update({"response_template": RESPONSE_TEMPLATE, "response_template_ids": template_ids, "max_len": a.max_len,
                 "padding_side": tok.padding_side, "pad_token_is_eos": tok.pad_token == tok.eos_token})
    if len(template_ids) < 1:
        raise SystemExit("response template encodes to nothing")

    cache = {}

    def rec_for(line):
        r = cache.get(line)
        if r is None:
            r = cache[line] = measure(line, tok, template_ids, a.max_len)
        return r

    groups = {"original_3k": p3, "new_4k": new_rows, "combined_7k": p7, "arm_A": arm_a, "arm_B": arm_b, "validation": val}
    results, digests = {}, {}
    for g, lines in groups.items():
        recs = [rec_for(l) for l in lines]
        results[g] = {"all": summarize(recs)}
        for t in TASKS:
            results[g][t] = summarize([r for r in recs if r["task"] == t])
        h = hashlib.sha256()
        for r in recs:
            h.update(f'{r["task"]},{r["seq_len"]},{r["full_len"]},{r["prompt_full"]},{r["target_full"]},{r["retained"]}\n'.encode())
        digests[g] = h.hexdigest()
    if a.verify_determinism:
        cache.clear()
        for g, lines in groups.items():
            h = hashlib.sha256()
            for l in lines:
                r = rec_for(l)
                h.update(f'{r["task"]},{r["seq_len"]},{r["full_len"]},{r["prompt_full"]},{r["target_full"]},{r["retained"]}\n'.encode())
            if h.hexdigest() != digests[g]:
                raise SystemExit(f"non-deterministic result for {g}")
        prov["determinism_second_pass_identical"] = True

    report = {"tokenizer": prov, "input_sha256": shas, "input_row_counts": counts, "input_integrity": integrity,
              "original_script": "original_training/scripts/train_lora.py::tokenize_and_mask (verbatim copy)",
              "per_group_result_digest_sha256": digests, "unique_rendered_rows_tokenized": len(cache), "results": results}
    with open(os.path.join(a.out, "token_stats.json"), "x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    if a.per_example:
        with open(os.path.join(a.out, "per_example.tsv"), "x") as f:
            f.write("row_sha256\ttask\tseq_len\tfull_len\tprompt_full\ttarget_full\tretained\tclass\n")
            for line, r in sorted(cache.items(), key=lambda kv: hashlib.sha256(kv[0]).hexdigest()):
                f.write(f'{hashlib.sha256(line).hexdigest()}\t{r["task"]}\t{r["seq_len"]}\t{r["full_len"]}\t{r["prompt_full"]}\t{r["target_full"]}\t{r["retained"]}\t{r["class"]}\n')
    print(f"tokenizer: {a.model} revision requested {a.revision}; verified by snapshot dir: {prov['revision_verified_by_snapshot_dir']}")
    print(f"chat_template sha256 {prov['chat_template_sha256']}; response template ids {template_ids}")
    hdr = f'{"group":<13}{"task":<8}{"n":>6}{"complete":>9}{"partial":>9}{"masked":>8}{"sup_kept":>10}{"sup_lost":>10}{"len_p99":>8}{"len_max":>8}'
    print(hdr)
    for g in groups:
        for t in ("all",) + TASKS:
            s = results[g][t]
            c = s["classes"]
            masked = c.get("fully_masked_template_cut", 0) + c.get("fully_masked_no_target_tokens", 0)
            print(f'{g:<13}{t:<8}{s["n"]:>6}{c.get("complete", 0):>9}{c.get("partial", 0):>9}{masked:>8}'
                  f'{s["supervised_tokens_retained"]:>10}{s["supervised_tokens_lost_to_truncation"]:>10}'
                  f'{s["seq_len_untruncated"]["p99"]:>8}{s["seq_len_untruncated"]["max"]:>8}')
    print(f"report: {os.path.join(a.out, 'token_stats.json')}")


if __name__ == "__main__":
    main()
