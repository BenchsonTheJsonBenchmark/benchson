"""Offline tests for analyze_tokens.py with a toy tokenizer (no transformers, no network).
Run: python3 -I -m unittest discover controlled_experiment/tests"""
import ast, json, os, sys, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import analyze_tokens as A

ORIG = os.path.join(HERE, "..", "..", "original_training", "scripts", "train_lora.py")
TEMPLATE = A.RESPONSE_TEMPLATE


def func_source(path, name):
    with open(path) as f:
        src = f.read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)


class Toy:
    """Whitespace tokenizer; each word is one token. Same call surface as the HF tokenizer pieces used by the original."""
    chat_template = "toy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return " ".join(f"<|start_of_role|>{m['role']}<|end_of_role|> {m['content']} <|end_of_text|>" for m in messages)

    def _ids(self, text):
        return text.replace("<|start_of_role|>", "S ").replace("<|end_of_role|>", " E").split()

    def __call__(self, text, truncation=False, max_length=None, padding=False):
        ids = self._ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)


def row(user_words, asst_words):
    return json.dumps({"messages": [{"role": "user", "content": " ".join(["u"] * user_words)},
                                    {"role": "assistant", "content": " ".join(["a"] * asst_words)}], "task_type": "fix"}).encode()


class T(unittest.TestCase):
    def test_function_is_verbatim_copy_of_original(self):
        self.assertEqual(func_source(ORIG, "tokenize_and_mask"), func_source(A.__file__, "tokenize_and_mask"))

    def test_classification_and_masking(self):
        tok, tid = Toy(), Toy().encode(TEMPLATE)
        # sequence layout: [S user E] u*n [<|end_of_text|>] [S assistant E] a*m [<|end_of_text|>]
        complete = A.measure(row(3, 4), tok, tid, 50)
        self.assertEqual(complete["class"], "complete")
        self.assertEqual(complete["retained"], complete["target_full"])
        full = complete["full_len"]
        partial = A.measure(row(3, 4), tok, tid, full - 2)
        self.assertEqual(partial["class"], "partial")
        self.assertEqual(partial["retained"], partial["target_full"] - 2)
        # cut inside the assistant template -> template absent -> everything masked
        cut_tpl = A.measure(row(3, 4), tok, tid, partial["prompt_full"] - 1)
        self.assertEqual(cut_tpl["class"], "fully_masked_template_cut")
        self.assertEqual(cut_tpl["retained"], 0)
        # template ends exactly at the limit -> template present but no target tokens
        at_end = A.measure(row(3, 4), tok, tid, partial["prompt_full"])
        self.assertEqual(at_end["class"], "fully_masked_no_target_tokens")
        # masking equals the original function's labels
        out = A.tokenize_and_mask(json.loads(row(3, 4)), tok, 50, tid)
        self.assertEqual(sum(1 for x in out["labels"] if x != -100), complete["target_full"])

    def test_summarize_counts(self):
        tok, tid = Toy(), Toy().encode(TEMPLATE)
        recs = [A.measure(r, tok, tid, 20) for r in (row(2, 3), row(2, 30), row(30, 2))]
        s = A.summarize(recs)
        self.assertEqual((s["n"], s["classes"].get("complete"), s["classes"].get("partial")), (3, 1, 1))
        self.assertEqual(sum(s["classes"].values()), 3)


if __name__ == "__main__":
    unittest.main()
