"""Build LoRA training data (chat JSONL) from the published Benchson v3 dataset.

Pulls the `train` split of a Benchson HuggingFace dataset (`--dataset`) for the
create / fix / modify families and projects each record into a chat example using
Benchson's *own* `format_for_llm` + `format_example`, so the training prompt is
byte-for-byte identical to what the benchmark uses. Only the `train` split is
touched — the held-out `test` split (disjoint schema pool) is never read.

Run from inside a benchson checkout (so `src/` is importable):

    cd <benchson> && PYTHONPATH=.:src python prepare_benchson_data.py \
        --dataset <HF_DATASET_ID> --revision <REVISION> --output data/train.jsonl

Output rows: {"messages": [...], "task_type": "create|fix|modify"} — the format
train_lora.py already consumes.
"""
import argparse
import json
import os

from datasets import load_dataset

from data_generation.tasks.base_task import BaseDataTask
from evaluations.create_by_schema.create_by_schema import CreateBySchema
from evaluations.json_error_evaluation.json_error_evaluation import ErrorJson
from evaluations.modify_json_evaluation.modify_json_evaluation import ModifyJson

# Mirrors export_training.py's _TASKS, but sources records from the HF dataset.
# Each entry: the eval whose format_for_llm defines the prompt, a record->test_case
# mapper, and the record key holding the gold answer.
TASKS = {
    "create": {
        "eval": CreateBySchema,
        "test_case": lambda r: {"data": r["schema"], "source_doc": r.get("description"),
                                "instructions": r.get("instructions")},
        "gold": "reference_json",
    },
    "fix": {
        "eval": ErrorJson,
        "test_case": lambda r: {"data": r["erroneous_json"], "schema": r["schema"],
                                "description": r.get("description")},
        "gold": "valid_json",
    },
    "modify": {
        "eval": ModifyJson,
        "test_case": lambda r: {"data": r["data"], "schema": r.get("schema"),
                                "instructions": r["instructions"]},
        "gold": "ground_truth",
    },
}


def _unstringify(row):
    """export_hf serializes dict/list fields to strings for a uniform Arrow schema.
    Parse those back; leave genuine plain-text fields (description, instructions) as-is.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="HF dataset id of the Benchson data, e.g. <HF_DATASET_ID>.")
    parser.add_argument("--output", required=True, help="Path to write chat JSONL.")
    parser.add_argument("--revision", default="main",
                        help="HF dataset revision/tag to pin (e.g. v2) for reproducibility.")
    parser.add_argument("--tasks", default="create,fix,modify",
                        help="Comma-separated subset of task families to include.")
    args = parser.parse_args()

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    formatter = BaseDataTask()
    written = {}

    with open(args.output, "w", encoding="utf-8") as out:
        for task in wanted:
            spec = TASKS[task]
            ds = load_dataset(args.dataset, task, split="train", revision=args.revision)
            evaluator = spec["eval"].__new__(spec["eval"])  # format_for_llm needs no state
            n = 0
            for row in ds:
                record = _unstringify(row)
                if spec["gold"] not in record or record[spec["gold"]] is None:
                    continue
                messages = evaluator.format_for_llm(spec["test_case"](record))
                completion = json.dumps(record[spec["gold"]])
                example = formatter.format_example(messages, completion, "chat")
                example["task_type"] = task
                out.write(json.dumps(example) + "\n")
                n += 1
            written[task] = n
            print(f"[{task}] wrote {n} examples (revision={args.revision})")

    print(f"Wrote {sum(written.values())} training examples to {args.output}")


if __name__ == "__main__":
    main()
