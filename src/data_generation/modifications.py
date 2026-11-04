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

Three entry points, kept separate on purpose:

- `apply_change(..., strategy="original")`: the original Benchson behaviour - one edit,
  uniform choice, top level plus one nesting level. Used by the "standard" mode.
- `apply_change(..., strategy="recursive_weighted")` (the default): one edit, any depth,
  complexity-weighted.
- `apply_changes(...)`: the "complex" mode - 2-3 compatible edits at once (different types,
  depths and branches, including objects inside arrays), with replay/verification helpers.
"""

import copy

from evaluations import metrics

_MAX_TRIES = 40
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
             "eighth", "ninth", "tenth"]


STRATEGY_ORIGINAL = "original"
STRATEGY_WEIGHTED = "recursive_weighted"


def apply_change(reference_json, schema, rng, strategy=STRATEGY_WEIGHTED):
    """Returns (modified_json, change_spec) or None if no valid edit was found.

    change_spec = {"op": str, "path": list, "precise_instruction": str}.

    `strategy="recursive_weighted"` (default): candidates are tried in a weighted random
    order driven by `rng` (higher complexity = more likely to come first).
    `strategy="original"`: the original behaviour - candidates from the top level and one
    nesting level only, shuffled uniformly with `rng.shuffle`.

    In both cases the first edit that changes the object and still passes full schema
    validation wins. At most `_MAX_TRIES` candidates are tried, each once.
    """
    if not isinstance(reference_json, dict):
        return None

    if strategy == STRATEGY_ORIGINAL:
        candidates = list(_candidates(reference_json, schema, max_depth=1))
        rng.shuffle(candidates)
    elif strategy == STRATEGY_WEIGHTED:
        candidates = _weighted_order(list(_candidates(reference_json, schema)), rng)
    else:
        raise ValueError(f"unknown modify strategy: {strategy!r}")

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


# ---------------------------------------------------------------- complex mode: several edits at once

_MAX_MULTI_POOL = 200   # candidates inspected per selection pass


def apply_changes(reference_json, schema, rng, n_ops=(2, 3), attempts=5):
    """Complex mode: applies several compatible edits at once.

    Returns (modified_json, spec) or None. The edits are chosen so that they

    - never touch the same place or an ancestor/descendant of each other (no conflicts),
    - differ in kind where possible (scalar change / field add-remove / array edit),
    - spread over different top-level branches where possible,
    - may sit inside objects that are list items (path contains the list index),
    - leave the object schema-valid after EVERY step (full-schema validation).

    `n_ops=(lo, hi)`: each attempt draws a target count in [lo, hi]; an attempt that
    reaches at least `lo` compatible edits is accepted (the achieved count is recorded in
    the spec), otherwise it is retried, up to `attempts` times. Only `rng` is used.

    spec = {"op": "multi", "path": [], "precise_instruction": str, "n_operations": int,
            "operations": [{"op", "path", "instruction", ["value"], ["index"]}, ...]}
    """
    lo, hi = n_ops
    if not isinstance(reference_json, dict) or lo < 1 or hi < lo:
        return None
    pool = list(_candidates(reference_json, schema, arrays=True, root=schema))
    if len(pool) < lo:
        return None

    for _ in range(attempts):
        target = rng.randint(lo, hi)
        chosen, modified = _select_compatible(reference_json, schema, pool, target, rng)
        if len(chosen) >= lo:
            operations = [_operation_record(c) for c in chosen]
            return modified, {
                "op": "multi",
                "path": [],
                "precise_instruction": _join_instructions(operations),
                "n_operations": len(operations),
                "operations": operations,
            }
    return None


def _candidate_category(candidate):
    op = candidate["op"]
    if op.startswith("array_"):
        return "array"
    if op in ("add_field", "delete_field"):
        return "field"
    return "scalar"


def paths_conflict(a, b):
    """True when one path equals, contains or is contained in the other."""
    n = min(len(a), len(b))
    return list(a[:n]) == list(b[:n])


def _select_compatible(original, schema, pool, target, rng):
    """Greedy, multi-pass selection. Pass 0 wants a new kind AND a new top-level branch,
    pass 1 a new kind, pass 2 anything compatible - so variety is preferred, not required."""
    ordered = _weighted_order(list(pool), rng)[:_MAX_MULTI_POOL]
    current = copy.deepcopy(original)
    chosen, categories, branches = [], set(), set()

    for pass_no in (0, 1, 2):
        for cand in ordered:
            if len(chosen) >= target:
                return chosen, current
            if any(cand is c for c in chosen):
                continue
            if any(paths_conflict(cand["path"], c["path"]) for c in chosen):
                continue
            category, branch = _candidate_category(cand), cand["path"][0]
            if pass_no == 0 and (category in categories or branch in branches):
                continue
            if pass_no == 1 and category in categories:
                continue
            trial = copy.deepcopy(current)
            try:
                cand["apply"](trial)
            except (KeyError, IndexError, TypeError):
                continue
            if trial == current or metrics.schema_compliance(trial, schema)[0] != 1:
                continue
            current = trial
            chosen.append(cand)
            categories.add(category)
            branches.add(branch)
    return chosen, current


def _operation_record(candidate):
    record = {"op": candidate["op"], "path": candidate["path"], "instruction": candidate["instruction"]}
    record.update(candidate.get("args", {}))
    return record


def _join_instructions(operations):
    parts = [f"({i}) {op['instruction']}" for i, op in enumerate(operations, start=1)]
    text = "Make all of the following changes to the JSON: " + " ".join(parts)
    if any(isinstance(step, int) for op in operations for step in op["path"]):
        text += " (List positions in square brackets are zero-based: items[0] is the first item.)"
    return text


def replay_operation(obj, op):
    """Redoes one recorded operation on `obj` (in place) from its data alone.

    Deliberately independent of the lambdas used while generating, so that replaying a
    stored spec is a genuine second implementation to cross-check the generator against.
    """
    kind, path = op["op"], op["path"]
    if kind in ("set_bool", "increment_number", "decrement_number", "set_enum", "set_string",
                "add_field"):
        _set_at(obj, path, copy.deepcopy(op["value"]))
    elif kind == "delete_field":
        _del_at(obj, path)
    else:
        items = _get_at(obj, path)
        if kind == "array_clear":
            items = []
        elif kind == "array_remove_first":
            items = items[1:]
        elif kind == "array_remove_last":
            items = items[:-1]
        elif kind == "array_remove_at":
            i = op["index"]
            items = items[:i] + items[i + 1:]
        elif kind == "array_append_copy":
            items = items + [copy.deepcopy(items[0])]
        elif kind == "array_remove_first_2":
            items = items[2:]
        elif kind == "array_keep_first":
            items = items[:1]
        else:
            raise ValueError(f"unknown operation: {kind!r}")
        _set_at(obj, path, items)


def verify_modification(original, modified, spec, schema):
    """Programmatic checks of a modify example. Returns {"ok", "checks", "problems", "manual_review"}.

    checks (True/False, or None when it cannot be checked for this spec):
      modified_differs        the ground truth really differs from the input
      schema_valid            the ground truth validates against the full schema
      operation_count         spec["n_operations"] matches the recorded operations
      paths_disjoint          no two operations touch the same place
      each_operation_effective  every operation, alone, changes the input
      replay_matches          redoing the recorded operations reproduces the ground truth
      no_unintended_changes   every changed leaf lies on one of the operation paths
    "manual_review" lists what cannot be established programmatically.
    """
    operations = spec.get("operations") or [spec]
    multi = "operations" in spec
    checks = {
        "modified_differs": modified != original,
        "schema_valid": metrics.schema_compliance(modified, schema)[0] == 1,
        "operation_count": (spec.get("n_operations", len(operations)) == len(operations)),
        "paths_disjoint": not any(paths_conflict(a["path"], b["path"])
                                  for i, a in enumerate(operations) for b in operations[i + 1:]),
    }

    replayable = multi and all(_replayable(op) for op in operations)
    if replayable:
        effective, replayed = True, copy.deepcopy(original)
        for op in operations:
            single = copy.deepcopy(original)
            try:
                replay_operation(single, op)
                replay_operation(replayed, op)
            except (KeyError, IndexError, TypeError, ValueError):
                effective, replayed = False, None
                break
            effective = effective and single != original
        checks["each_operation_effective"] = effective
        checks["replay_matches"] = replayed == modified
    else:
        checks["each_operation_effective"] = None
        checks["replay_matches"] = None

    paths = [tuple(op["path"]) for op in operations]
    changed = metrics.changed_paths(original, modified)
    # A changed leaf is intended when it lies under an operation path, or when it is the
    # empty container that an operation filled (e.g. {} gaining a field): then the leaf's
    # own path is a prefix of the operation path.
    checks["no_unintended_changes"] = all(
        any(paths_conflict(leaf, p) for p in paths) for leaf in changed)

    problems = [name for name, ok in checks.items() if ok is False]
    manual = ["instruction wording matches the intended operations (heuristic and round-trip "
              "checks only)"]
    return {"ok": not problems, "checks": checks, "problems": problems, "manual_review": manual}


def _replayable(op):
    kind = op["op"]
    if kind in ("set_bool", "increment_number", "decrement_number", "set_enum", "set_string",
                "add_field"):
        return "value" in op
    if kind == "array_remove_at":
        return "index" in op
    return True


def instruction_covers_operations(instruction, operations):
    """Heuristic: does the (possibly rewritten) instruction mention every operation?

    For each operation the field name must appear, and a string/number value it sets must
    appear too. This catches a rewrite that dropped a change; it cannot prove the wording
    is right (that needs the round-trip check or a manual look). Returns (ok, missing).
    """
    text = instruction.lower()
    missing = []
    for op in operations:
        key = next((str(k) for k in reversed(op["path"]) if isinstance(k, str)), None)
        if key is not None and key.lower() not in text:
            missing.append(f"{op['op']}:{key}")
            continue
        value = op.get("value")
        if isinstance(value, str) and value.strip() and value.lower() not in text:
            missing.append(f"{op['op']}:value")
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and op["op"] in ("set_enum", "add_field") and str(value) not in text:
            missing.append(f"{op['op']}:value")
    return not missing, missing


# ---------------------------------------------------------------- candidate discovery

_MAX_ARRAY_ELEMENTS = 3   # objects inside an array: only the first few elements are traversed


class _Walk:
    """Traversal options.

    max_depth: how many levels of nested objects to descend into (None = unlimited;
        1 = the original "top level + one nested level").
    arrays:    also descend into the objects inside arrays (path gets the list index).
    root:      the full schema, used to follow local "$ref"s; None = refs are not followed.
    """
    __slots__ = ("max_depth", "arrays", "root")

    def __init__(self, max_depth=None, arrays=False, root=None):
        self.max_depth, self.arrays, self.root = max_depth, arrays, root


def _candidates(obj, schema, max_depth=None, arrays=False, root=None):
    """Yields edit candidates for the root object and every nested object, at any depth."""
    yield from _walk_object_candidates(obj, schema, [], _Walk(max_depth, arrays, root), 0)


def _walk_object_candidates(obj, schema, path_prefix, opts=None, depth=0):
    """Candidates for one object, then recursion into its object-valued children.

    `schema` is the sub-schema describing `obj`; each child gets the matching property
    sub-schema, so `required`/constraints are honoured at every level. With
    `opts.arrays`, objects inside arrays are followed too. The recursion follows the
    data, which is finite, so it always terminates.
    """
    opts = opts or _Walk()
    schema = _resolve(schema, opts)
    props = schema.get("properties", {})
    required = schema.get("required", [])

    yield from _object_field_candidates(obj, props, required, path_prefix, opts)

    if opts.max_depth is not None and depth >= opts.max_depth:
        return
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _walk_object_candidates(value, props.get(key), path_prefix + [key],
                                               opts, depth + 1)
        elif opts.arrays and isinstance(value, list):
            items = _resolve(props.get(key), opts).get("items")
            for idx, element in enumerate(value[:_MAX_ARRAY_ELEMENTS]):
                if isinstance(element, dict):
                    yield from _walk_object_candidates(element, items, path_prefix + [key, idx],
                                                       opts, depth + 1)


def _object_field_candidates(obj, props, required, parent_path, opts=None):
    """Candidates for the fields of one object (its keys) plus add-missing-field.

    `parent_path` is the list of keys leading to this object ([] for the root).
    """
    opts = opts or _Walk()
    base_path = list(parent_path)
    # Label used in the human-readable instructions, e.g. settings.server.network
    parent = _label(parent_path)

    for key, value in obj.items():
        path = base_path + [key]
        ps = _resolve(props.get(key), opts)

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
        ps = _resolve(raw_ps, opts)
        value = _make_value(ps)
        if value is _NO_VALUE:
            continue
        path = base_path + [key]
        yield _cand("add_field", path,
                    f'Add a "{key}" field{_in(parent)} set to {_render(value)}.',
                    lambda o, p=path, v=value: _set_at(o, p, v), value=value)


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
                    lambda o, p=path, i=idx: _set_at(o, p, _get_at(o, p)[:i] + _get_at(o, p)[i + 1:]),
                    index=idx)
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
                lambda o, p=path, v=new: _set_at(o, p, v), value=new)


def _number_candidates(path, value, ps, key, parent):
    is_int = isinstance(value, int) and not isinstance(value, bool)
    for delta in (1, 10):
        new = value + delta
        if not _in_range(new, ps):
            continue
        new = int(new) if is_int else float(new)
        yield _cand("increment_number", path,
                    f'Increase "{key}"{_in(parent)} by {delta}.',
                    lambda o, p=path, v=new: _set_at(o, p, v), value=new)
        break
    low = value - 1
    if _in_range(low, ps):
        low = int(low) if is_int else float(low)
        yield _cand("decrement_number", path,
                    f'Decrease "{key}"{_in(parent)} by 1.',
                    lambda o, p=path, v=low: _set_at(o, p, v), value=low)


def _enum_candidates(path, value, choices, key, parent):
    others = [c for c in choices if c != value]
    if not others:
        return
    new = others[0]
    yield _cand("set_enum", path,
                f'Change "{key}"{_in(parent)} to {_render(new)}.',
                lambda o, p=path, v=new: _set_at(o, p, v), value=new)


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
                lambda o, p=path, v=new: _set_at(o, p, v), value=new)


# ---------------------------------------------------------------- helpers

_NO_VALUE = object()


def _cand(op, path, instruction, apply_fn, **args):
    """One edit candidate. `args` (value / index) is what a replay needs to redo the edit."""
    return {"op": op, "path": list(path), "instruction": instruction, "apply": apply_fn,
            "args": args}


def _label(path):
    """Human-readable location of an object: settings.server, items[0].address, or None."""
    out = ""
    for step in path:
        out += f"[{step}]" if isinstance(step, int) else ("." if out else "") + str(step)
    return out or None


def _resolve(ps, opts):
    """The sub-schema as a dict; follows local "$ref"s only when the traversal has a root."""
    if not isinstance(ps, dict):
        return {}
    root = opts.root if opts is not None else None
    for _ in range(5):
        ref = ps.get("$ref")
        if root is None or not isinstance(ref, str) or not ref.startswith("#/"):
            break
        node = root
        for part in ref[2:].split("/"):
            node = node.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            break
        ps = {**{k: v for k, v in ps.items() if k != "$ref"}, **node}
    return ps


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
