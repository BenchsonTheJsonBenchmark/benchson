"""Read-only: explains model calls that failed with a context-length error (e.g. the pilot's 3 failures).

    python src/analyze_context_failures.py CALLS.jsonl [--max-model-len 8192 --max-tokens 4096]

For every failed call whose error mentions the context length it prints which request it was (the schema's own first
prompt, a repair prompt that grew with the previous invalid answer, a description prompt, ...), the message count and
the prompt size in characters. It reads the call log only and writes nothing.
"""
import argparse
import json
import re
import sys


def classify(messages):
    text = " ".join(m.get("content") or "" for m in messages if isinstance(m, dict))
    last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if "The JSON is invalid:" in last_user:
        return "repair_prompt (previous invalid answer included)"
    if "Generate a JSON object that conforms to the following schema" in text:
        return "initial_schema_prompt (the schema's own prompt)"
    if "containing the actual values" in last_user:
        return "description_prompt (schema + reference object)"
    if last_user.startswith("Rewrite the following data-edit instruction"):
        return "instruction_rewrite"
    return "other"


def analyze(path, max_model_len=8192, max_tokens=4096):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            call = json.loads(line)
            error = call.get("error") or ""
            if not re.search(r"context length|context window|max_model_len", error, re.I):
                continue
            messages = call.get("messages") or []
            found = re.search(r"at least (\d+) input tokens", error)
            rows.append({"call": call.get("call"), "kind": classify(messages), "messages": len(messages),
                         "prompt_chars": sum(len(m.get("content") or "") for m in messages),
                         "reported_prompt_tokens_at_least": int(found.group(1)) if found else None,
                         "allowed_prompt_tokens": max_model_len - max_tokens})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("calls_log")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args(argv)
    rows = analyze(args.calls_log, args.max_model_len, args.max_tokens)
    print(json.dumps({"context_failures": rows, "count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
