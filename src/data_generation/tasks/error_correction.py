import copy
import json
import random

from data_generation.tasks.base_task import BaseDataTask

_FORMAT_INVALIDS = {
    "email": "not-an-email",
    "date": "not-a-date",
    "date-time": "not-a-datetime",
    "uri": "not-a-uri",
    "uuid": "not-a-uuid",
}

_WRONG_TYPE_VALUES = {
    "string": 99999,
    "number": "not-a-number",
    "integer": "not-a-number",
    "boolean": "not-a-bool",
    "array": "not-an-array",
    "object": "not-an-object",
}


class ErrorCorrectionTask(BaseDataTask):

    def generate(self, schema_path, llm_provider, max_retries):
        """Returns (messages, valid_json_str) or None."""
        result = self.generate_valid_json(schema_path, llm_provider, max_retries)
        if result is None:
            return None

        schema, valid_json_str, _ = result
        valid_json = json.loads(valid_json_str)

        erroneous_json = _inject_error(valid_json, schema)
        if erroneous_json is None:
            return None

        messages = self._build_error_correction_prompt(schema, erroneous_json)
        return messages, valid_json_str


def _inject_error(valid_json, schema, rng=None):
    """Applies a single random schema-violating mutation. Returns None if no mutation is possible.

    `rng` (a random.Random) makes the choice reproducible; without it the global `random`
    module is used, as before.
    """
    return _inject_error_info(valid_json, schema, rng)[0]


def _inject_error_info(valid_json, schema, rng=None):
    """Like `_inject_error`, but returns (erroneous_or_None, {"kind", "field"}_or_None).

    The info names the mutation that was applied, for metadata. The mutation set and the
    way one is picked are exactly those of the original single-error injector.
    """
    if not isinstance(valid_json, dict):
        return None, None

    props = schema.get("properties", {})
    required = schema.get("required", [])
    erroneous = copy.deepcopy(valid_json)
    mutations = []

    # ── existing mutations ────────────────────────────────────────────────────

    for field in required:
        if field in erroneous:
            mutations.append(("remove_required", field))

    for field, ps in props.items():
        if field not in erroneous:
            continue
        val = erroneous[field]
        ptype = ps.get("type")

        if "enum" in ps:
            mutations.append(("invalid_enum", field))
        if ptype in ("number", "integer") and isinstance(val, (int, float)):
            mutations.append(("wrong_type_number", field))
        if ptype == "boolean":
            mutations.append(("wrong_type_boolean", field))
        if ptype == "string" and ps.get("minLength", 0) > 0 and isinstance(val, str) and len(val) > 0:
            mutations.append(("empty_string", field))

    # ── new mutations ─────────────────────────────────────────────────────────

    # Numeric range violations
    for field, ps in props.items():
        if field not in erroneous:
            continue
        val = erroneous[field]
        if not isinstance(val, (int, float)):
            continue
        if "minimum" in ps:
            mutations.append(("below_minimum", field))
        if "exclusiveMinimum" in ps and isinstance(ps["exclusiveMinimum"], (int, float)):
            mutations.append(("below_exclusive_minimum", field))
        if "maximum" in ps:
            mutations.append(("above_maximum", field))
        if "exclusiveMaximum" in ps and isinstance(ps["exclusiveMaximum"], (int, float)):
            mutations.append(("above_exclusive_maximum", field))

    # String too long
    for field, ps in props.items():
        if field not in erroneous or "maxLength" not in ps:
            continue
        if isinstance(erroneous[field], str):
            mutations.append(("string_too_long", field))

    # Null for non-nullable field
    for field, ps in props.items():
        if field not in erroneous:
            continue
        ptype = ps.get("type")
        if isinstance(ptype, str) and ptype not in ("null",):
            mutations.append(("null_non_nullable", field))

    # Additional property when additionalProperties is false
    if schema.get("additionalProperties") is False:
        mutations.append(("additional_property", None))

    # Array violations
    for field, ps in props.items():
        if field not in erroneous or ps.get("type") != "array":
            continue
        val = erroneous[field]
        if not isinstance(val, list):
            continue
        items_schema = ps.get("items", {})
        item_type = items_schema.get("type") if isinstance(items_schema, dict) else None
        if item_type and val:
            mutations.append(("wrong_array_item_type", field))
        min_items = ps.get("minItems", 0)
        if min_items >= 2 and len(val) >= min_items:
            mutations.append(("below_min_items", field))
        if "maxItems" in ps:
            mutations.append(("above_max_items", field))

    # Invalid format
    for field, ps in props.items():
        if field not in erroneous:
            continue
        fmt = ps.get("format")
        if fmt in _FORMAT_INVALIDS and isinstance(erroneous[field], str):
            mutations.append(("invalid_format", field))

    # Nested required field removal
    for field, ps in props.items():
        if field not in erroneous or ps.get("type") != "object":
            continue
        nested_val = erroneous[field]
        if not isinstance(nested_val, dict):
            continue
        nested_required = ps.get("required", [])
        for nf in nested_required:
            if nf in nested_val:
                mutations.append(("nested_remove_required", field, nf))

    if not mutations:
        return None, None

    chosen = (rng or random).choice(mutations)
    kind = chosen[0]
    field = chosen[1] if len(chosen) > 1 else None
    info = {"kind": kind, "field": field}
    if kind == "nested_remove_required":
        info["nested_field"] = chosen[2]

    # ── apply chosen mutation ─────────────────────────────────────────────────

    if kind == "additional_property":
        erroneous["__invalid_field__"] = "invalid"
        return erroneous, info

    if field is None:
        return erroneous, info

    if kind == "remove_required":
        del erroneous[field]

    elif kind == "invalid_enum":
        erroneous[field] = str(erroneous[field]) + "_INVALID"

    elif kind == "wrong_type_number":
        erroneous[field] = "not-a-number"

    elif kind == "wrong_type_boolean":
        erroneous[field] = "not-a-bool"

    elif kind == "empty_string":
        erroneous[field] = ""

    elif kind == "below_minimum":
        minimum = props[field]["minimum"]
        erroneous[field] = minimum - 1

    elif kind == "below_exclusive_minimum":
        ex_min = props[field]["exclusiveMinimum"]
        erroneous[field] = ex_min - 1

    elif kind == "above_maximum":
        maximum = props[field]["maximum"]
        erroneous[field] = maximum + 1

    elif kind == "above_exclusive_maximum":
        ex_max = props[field]["exclusiveMaximum"]
        erroneous[field] = ex_max + 1

    elif kind == "string_too_long":
        max_len = props[field]["maxLength"]
        erroneous[field] = "x" * (max_len + 5)

    elif kind == "null_non_nullable":
        erroneous[field] = None

    elif kind == "wrong_array_item_type":
        items_schema = props[field].get("items", {})
        item_type = items_schema.get("type") if isinstance(items_schema, dict) else None
        wrong_val = _WRONG_TYPE_VALUES.get(item_type or "", "wrong")
        erroneous[field] = [wrong_val] + erroneous[field][1:]

    elif kind == "below_min_items":
        min_items = props[field]["minItems"]
        erroneous[field] = erroneous[field][:min_items - 1]

    elif kind == "above_max_items":
        max_items = props[field]["maxItems"]
        extra = erroneous[field][0] if erroneous[field] else "extra"
        erroneous[field] = erroneous[field] + [extra] * (max_items - len(erroneous[field]) + 1)

    elif kind == "invalid_format":
        fmt = props[field].get("format")
        erroneous[field] = _FORMAT_INVALIDS.get(fmt, "invalid")

    elif kind == "nested_remove_required":
        nested_field = chosen[2]
        del erroneous[field][nested_field]

    return erroneous, info
