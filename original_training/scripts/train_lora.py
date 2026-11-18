import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer  # type: ignore[attr-defined]
from peft import LoraConfig, get_peft_model, TaskType  # type: ignore[import-untyped]
import torch


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


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


class PaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    response_template = "<|start_of_role|>assistant<|end_of_role|>"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    records = load_jsonl(args.data_path)
    n_train = int(len(records) * 0.9)
    train_ds = Dataset.from_list(records[:n_train])
    eval_ds = Dataset.from_list(records[n_train:])

    fn = lambda x: tokenize_and_mask(x, tokenizer, args.max_seq_length, response_template_ids)
    # Drop all source columns (messages, task_type, …); the tokenizer emits fresh ones.
    train_ds = train_ds.map(fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(fn, remove_columns=eval_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # fraction of total steps (= the original warmup_ratio=0.05)
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PaddingCollator(tokenizer.pad_token_id),
    )

    print(f"Training on {len(train_ds)} samples, eval on {len(eval_ds)}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
