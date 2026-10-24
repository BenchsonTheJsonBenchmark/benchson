"""Schema-aware semantic edits for the modify-by-instruction task (2c).

`apply_change` takes a valid object and its schema and returns a *correct by
construction* modified object together with a precise natural-language instruction
describing the edit. Edits are applied programmatically (never guessed), and the
result is re-validated against the full schema — so the modified ground truth is
guaranteed to stay schema-valid (an array clear that would drop below `minItems`,
for instance, is rejected and another edit is chosen).

Coverage: array edits, scalar value changes, and field add/remove, over top-level
fields and, recursively, over every nested object-valued property (any depth).

Selection is random but complexity-aware: every candidate gets a simple score (see
`_candidate_complexity`) and candidates are tried in a weighted random order, so deeper
and structurally richer edits are chosen more often than trivial top-level ones.
"""

import copy

from evaluations import metrics

_MAX_TRIES = 40
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
             "eighth", "ninth", "tenth"]


def apply_change(reference_json, schema, rng):
    """Returns (modified_json, change_spec) or None if no valid edit was found.

    change_spec = {"op": str, "path": list, "precise_instruction": str}.

    Candidates are tried in a weighted random order driven by `rng` (higher complexity =
    more likely to come first); the first edit that changes the object and still passes
    full schema validation wins. At most `_MAX_TRIES` candidates are tried, each once.
    """
    if not isinstance(reference_json, dict):
        return None

    candidates = _weighted_order(list(_candidates(reference_json, schema)), rng)

    for cand in candidates[:_MAX_TRIES]:
        modified = copy.deepcopy(reference_json)
        try:
            cand["apply"](modified)
        except (KeyError, IndexError, TypeError):
            continue
        if modified == reference_json:
            continue
        if metrics.schema_compliance(modified, schema)[0] != 1:
            continue
        return modified, {
            "op": cand["op"],
            "path": cand["path"],
            "precise_instruction": cand["instruction"],
        }
    return None


# ---------------------------------------------------------------- complexity-aware selection

_OP_BASE_SCALAR = 1   # set/increment/decrement a scalar, toggle a bool, switch an enum
_OP_BASE_FIELD = 2    # add or delete a field
_OP_BASE_ARRAY = 3    # any array_* edit


def _candidate_complexity(candidate):
    """Simple, explainable complexity score of one edit candidate (higher = harder).

        score = operation base + (path depth - 1)

    Operation base: scalar edit 1, add/delete field 2, array edit 3. The depth term adds
    1 for every level below the top, so ["port"] scores lower than
    ["settings", "server", "network", "port"] for the same operation. Only `op` and
    `path` are read, so a stored change spec can be scored too.
    """
    op = candidate["op"]
    if op.startswith("array_"):
        base = _OP_BASE_ARRAY
    elif op in ("add_field", "delete_field"):
        base = _OP_BASE_FIELD
    else:
        base = _OP_BASE_SCALAR
    depth = max(len(candidate["path"]), 1)
    return base + (depth - 1)


def _weighted_order(candidates, rng):
    """Returns the candidates in a random order that favours higher complexity.

    Weighted sampling without replacement via an "exponential race": each candidate draws
    Exp(1) / weight, and candidates are ordered by that time (smallest first). The first
    pick is chosen with probability weight / total weight, then the next from the rest,
    and so on - so every candidate appears exactly once and nothing is ever retried.
    Only the supplied `rng` is used (never the global random module); ties keep the
    original order, so the result is fully determined by the rng state.
    """
    timed = [(rng.expovariate(1.0) / _candidate_complexity(c), i) for i, c in enumerate(candidates)]
    timed.sort()
    return [candidates[i] for _, i in timed]


# ---------------------------------------------------------------- candidate discovery

def _candidates(obj, schema):
    """Yields edit candidates for the root object and every nested object, at any depth."""
    yield from _walk_object_candidates(obj, schema, path_prefix=[])


def _walk_object_candidates(obj, schema, path_prefix):
    """Candidates for one object, then recursion into its object-valued children.

    `schema` is the sub-schema describing `obj`; each child gets the matching property
    sub-schema, so `required`/constraints are honoured at every level. Only dict values
    are followed (arrays of objects are not traversed yet). The recursion follows the
    data, which is finite, so it always terminates.
    """
    schema = _subschema(schema)
    props = schema.get("properties", {})
    required = schema.get("required", [])

    yield from _object_field_candidates(obj, props, required, path_prefix)

    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _walk_object_candidates(value, props.get(key), path_prefix + [key])


def _object_field_candidates(obj, props, required, parent_path):
    """Candidates for the fields of one object (its keys) plus add-missing-field.

    `parent_path` is the list of keys leading to this object ([] for the root).
    """
    base_path = list(parent_path)
    # Label used in the human-readable instructions, e.g. settings.server.network
    parent = ".".join(str(k) for k in parent_path) or None

    for key, value in obj.items():
        path = base_path + [key]
        ps = _subschema(props.get(key))

        if isinstance(value, list):
            yield from _array_candidates(path, value, key, parent)
        elif isinstance(value, bool):
            yield from _bool_candidates(path, value, key, parent)
        elif isinstance(value, (int, float)):
            yield from _number_candidates(path, value, ps, key, parent)

        if "enum" in ps and isinstance(ps["enum"], list):
            yield from _enum_candidates(path, value, ps["enum"], key, parent)
        elif isinstance(value, str):
            yield from _string_candidates(path, value, ps, key, parent)

        # delete an optional field
        if key not in required:
            yield _cand("delete_field", path,
                        f'Remove the "{key}" field{_in(parent)}.',
                        lambda o, p=path: _del_at(o, p))

    # add a missing optional property
    for key, raw_ps in props.items():
        if key in obj or key in required:
            continue
        ps = _subschema(raw_ps)
        value = _make_value(ps)
        if value is _NO_VALUE:
            continue
        path = base_path + [key]
        yield _cand("add_field", path,
                    f'Add a "{key}" field{_in(parent)} set to {_render(value)}.',
                    lambda o, p=path, v=value: _set_at(o, p, v))


def _array_candidates(path, value, key, parent):
    desc = f'the "{key}" list{_in(parent)}'
    n = len(value)
    yield _cand("array_clear", path, f"Remove all items from {desc}.",
                lambda o, p=path: _set_at(o, p, []))
    if n >= 1:
        yield _cand("array_remove_first", path,
                    f"Remove the {_count('first', 1)} from {desc}.",
                    lambda o, p=path: _set_at(o, p, _get_at(o, p)[1:]))
        yield _cand("array_remove_last", path,
                    f"Remove the {_count('last', 1)} from {desc}.",
                    lambda o, p=path: _set_at(o, p, _get_at(o, p)[:-1]))
        idx = n - 1
        yield _cand("array_remove_at", path,
                    f"Remove the {_ordinal(idx)} item from {desc}.",
                    lambda o, p=path, i=idx: _set_at(o, p, _get_at(o, p)[:i] + _get_at(o, p)[i + 1:]))
        yield _cand("array_append_copy", path,
                    f"Add another item to {desc}, identical to its first item.",
                    lambda o, p=path: _set_at(o, p, _get_at(o, p) + [copy.deepcopy(_get_at(o, p)[0])]))
    if n >= 2:
        yield _cand("array_remove_first_2", path,
                    f"Remove the first 2 items from {desc}.",
                    lambda o, p=path: _set_at(o, p, _get_at(o, p)[2:]))
        yield _cand("array_keep_first", path,
                    f"Keep only the first item in {desc} and remove the rest.",
                    lambda o, p=path: _set_at(o, p, _get_at(o, p)[:1]))


def _bool_candidates(path, value, key, parent):
    new = not value
    yield _cand("set_bool", path,
                f'Set "{key}"{_in(parent)} to {_render(new)}.',
                lambda o, p=path, v=new: _set_at(o, p, v))


def _number_candidates(path, value, ps, key, parent):
    is_int = isinstance(value, int) and not isinstance(value, bool)
    for delta in (1, 10):
        new = value + delta
        if not _in_range(new, ps):
            continue
        new = int(new) if is_int else float(new)
        yield _cand("increment_number", path,
                    f'Increase "{key}"{_in(parent)} by {delta}.',
                    lambda o, p=path, v=new: _set_at(o, p, v))
        break
    low = value - 1
    if _in_range(low, ps):
        low = int(low) if is_int else float(low)
        yield _cand("decrement_number", path,
                    f'Decrease "{key}"{_in(parent)} by 1.',
                    lambda o, p=path, v=low: _set_at(o, p, v))


def _enum_candidates(path, value, choices, key, parent):
    others = [c for c in choices if c != value]
    if not others:
        return
    new = others[0]
    yield _cand("set_enum", path,
                f'Change "{key}"{_in(parent)} to {_render(new)}.',
                lambda o, p=path, v=new: _set_at(o, p, v))


def _string_candidates(path, value, ps, key, parent):
    # only plain strings — avoid format/pattern fields whose edits tend to break validation
    if ps.get("format") or ps.get("pattern"):
        return
    new = (value + " (updated)")
    max_len = ps.get("maxLength")
    if isinstance(max_len, int):
        new = new[:max_len]
    if new == value:
        return
    yield _cand("set_string", path,
                f'Change "{key}"{_in(parent)} to {_render(new)}.',
                lambda o, p=path, v=new: _set_at(o, p, v))


# ---------------------------------------------------------------- helpers

_NO_VALUE = object()


def _cand(op, path, instruction, apply_fn):
    return {"op": op, "path": list(path), "instruction": instruction, "apply": apply_fn}


def _subschema(ps):
    return ps if isinstance(ps, dict) else {}


def _in(parent):
    return f' inside "{parent}"' if parent else ""


def _ordinal(i):
    return _ORDINALS[i] if 0 <= i < len(_ORDINALS) else f"{i + 1}th"


def _count(word, n):
    return f"{word} item" if n == 1 else f"{word} {n} items"


def _in_range(value, ps):
    if "minimum" in ps and value < ps["minimum"]:
        return False
    if "maximum" in ps and value > ps["maximum"]:
        return False
    if "exclusiveMinimum" in ps and isinstance(ps["exclusiveMinimum"], (int, float)) and value <= ps["exclusiveMinimum"]:
        return False
    if "exclusiveMaximum" in ps and isinstance(ps["exclusiveMaximum"], (int, float)) and value >= ps["exclusiveMaximum"]:
        return False
    return True


def _make_value(ps):
    """Best-effort valid value for adding a missing optional field."""
    if "enum" in ps and isinstance(ps["enum"], list) and ps["enum"]:
        return ps["enum"][0]
    t = ps.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "boolean":
        return True
    if t == "integer":
        return int(ps.get("minimum", 1))
    if t == "number":
        return float(ps.get("minimum", 1))
    if t == "string":
        n = ps.get("minLength", 0) or 0
        if ps.get("format") or ps.get("pattern"):
            return _NO_VALUE
        return "value" if n <= 5 else "v" * n
    if t == "array" and (ps.get("minItems", 0) or 0) == 0:
        return []
    return _NO_VALUE


def _render(value):
    import json
    return json.dumps(value)


def _get_at(obj, path):
    cur = obj
    for k in path:
        cur = cur[k]
    return cur


def _set_at(obj, path, value):
    cur = obj
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def _del_at(obj, path):
    cur = obj
    for k in path[:-1]:
        cur = cur[k]
    del cur[path[-1]]
