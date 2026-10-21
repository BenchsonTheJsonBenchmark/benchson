"""Offline stand-in for an LLM, to test the generation pipeline without any model.

It is not smart. It recognises the specific prompts Benchson's builder sends and
answers with simple, deterministic output, so the full path (schema sampling, gates,
writing files, statistics, review file) can run for free. Never use it to produce
real data.
"""

import json

from ..llm_provider import LLMProvider

_SCHEMA_MARKER = "Generate a JSON object that conforms to the following schema:"
_DOC_MARKER = "containing the actual values:\n"
_DOC_END = "\n\nYour task is"
_REWRITE_MARKER = "Rewrite the following data-edit instruction"

_MAX_DEPTH = 6


class MockProvider(LLMProvider):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = "mock"

    def _generate(self, messages, parameters=None):
        for m in messages:
            if m["role"] == "user" and _SCHEMA_MARKER in m["content"]:
                schema = _first_json(m["content"].split(_SCHEMA_MARKER, 1)[1])
                if isinstance(schema, dict):
                    return json.dumps(make_instance(schema), indent=2)

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if _DOC_MARKER in last_user:
            reference = _first_json(last_user.split(_DOC_MARKER, 1)[1].split(_DOC_END, 1)[0])
            return "This record describes: " + "; ".join(_strings(reference)) + "."

        if last_user.startswith(_REWRITE_MARKER) and "Instruction: " in last_user:
            return last_user.split("Instruction: ", 1)[1].strip()

        return "{}"


def _first_json(text):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    return None


def _strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _resolve_ref(ref, root):
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return {}
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node


def make_instance(schema, root=None, depth=0):
    """A simple example value for `schema` (best effort; not every keyword is honoured)."""
    root = root if root is not None else schema
    if not isinstance(schema, dict) or depth > _MAX_DEPTH:
        return None
    if "$ref" in schema:
        return make_instance(_resolve_ref(schema["$ref"], root), root, depth + 1)
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list) and schema[key]:
            return make_instance(schema[key][0], root, depth + 1)
    if isinstance(schema.get("allOf"), list) and schema["allOf"]:
        merged = {}
        for sub in schema["allOf"]:
            part = make_instance(sub, root, depth + 1)
            if isinstance(part, dict):
                merged.update(part)
        if merged:
            return merged

    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t is None:
        t = "object" if "properties" in schema else "array" if "items" in schema else None

    if t == "object":
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        out = {}
        for name, sub in props.items():
            if name in required or depth < 2:
                out[name] = make_instance(sub, root, depth + 1)
        for name in required:
            out.setdefault(name, "value")
        return out
    if t == "array":
        items = schema.get("items", {})
        lo = schema.get("minItems", 0) if isinstance(schema.get("minItems"), int) else 0
        hi = schema.get("maxItems") if isinstance(schema.get("maxItems"), int) else 2
        count = max(lo, min(2, hi))
        if isinstance(items, list):
            return [make_instance(s, root, depth + 1) for s in items]
        return [make_instance(items, root, depth + 1) for _ in range(count)]
    if t == "string":
        fmt = schema.get("format")
        known = {"date": "2024-03-15", "date-time": "2024-03-15T10:30:00Z",
                 "email": "user@example.com", "uri": "https://example.com",
                 "uuid": "123e4567-e89b-12d3-a456-426614174000"}
        value = known.get(fmt, "sample text")
        min_len = schema.get("minLength") if isinstance(schema.get("minLength"), int) else 0
        max_len = schema.get("maxLength") if isinstance(schema.get("maxLength"), int) else None
        value = value.ljust(min_len, "x")
        return value[:max_len] if max_len is not None else value
    if t in ("integer", "number"):
        lo = schema.get("minimum", schema.get("exclusiveMinimum"))
        hi = schema.get("maximum", schema.get("exclusiveMaximum"))
        value = 1
        if isinstance(lo, (int, float)) and not isinstance(lo, bool):
            value = max(value, int(lo) + (1 if "exclusiveMinimum" in schema else 0))
        if isinstance(hi, (int, float)) and not isinstance(hi, bool):
            value = min(value, int(hi))
        return value
    if t == "boolean":
        return True
    if t == "null":
        return None
    return "value"
