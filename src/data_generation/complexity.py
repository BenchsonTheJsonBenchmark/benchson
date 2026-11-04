"""Complexity measurements and metadata records for generated examples.

Everything here is a plain, explainable count - no learned score. Instance complexity
(how rich the JSON is) and task complexity (how much work the instruction/error set
demands) are measured separately and are not assumed to move together.

The records produced here go into a sidecar file next to each dataset (never into the
training records), so the benchmark/training data format is unchanged.
"""

from evaluations import metrics
from data_generation.modifications import _resolve, _Walk


def path_depth(path):
    """Depth of a location: a top-level key is 1, a key inside it 2, and so on."""
    return len(path)


def _leaf_kind(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return "empty_container"


def _object_arrays(value):
    """Counts arrays that contain at least one object, and the objects in them."""
    arrays = items = 0
    if isinstance(value, dict):
        for v in value.values():
            a, i = _object_arrays(v)
            arrays, items = arrays + a, items + i
    elif isinstance(value, list):
        objects = [v for v in value if isinstance(v, dict)]
        if objects:
            arrays, items = arrays + 1, items + len(objects)
        for v in value:
            a, i = _object_arrays(v)
            arrays, items = arrays + a, items + i
    return arrays, items


def _constrained_leaves(reference, schema):
    """Leaves whose schema node carries a value constraint (enum, range, length, pattern, format).

    Best effort: follows properties/items and local $refs only. Treat as a lower bound.
    """
    opts = _Walk(root=schema)
    keys = ("enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
            "minLength", "maxLength", "pattern", "format", "multipleOf")

    def walk(value, sch):
        sch = _resolve(sch, opts)
        if isinstance(value, dict) and value:
            props = sch.get("properties", {}) if isinstance(sch.get("properties"), dict) else {}
            return sum(walk(v, props.get(k)) for k, v in value.items())
        if isinstance(value, list) and value:
            items = sch.get("items")
            return sum(walk(v, items) for v in value)
        return 1 if any(k in sch for k in keys) else 0

    return walk(reference, schema)


def reference_metrics(reference, schema=None):
    """Instance complexity of a (create) reference object."""
    leaves = list(metrics._iter_leaves(reference))
    arrays, array_items = _object_arrays(reference)
    out = {
        "leaf_count": len(leaves),
        "depth": max((path_depth(p) for p, _ in leaves), default=0),
        "distinct_leaf_types": len({_leaf_kind(v) for _, v in leaves}),
        "object_arrays": arrays,
        "object_array_items": array_items,
        "checkable_strings": sum(
            1 for _, v in leaves
            if isinstance(v, str) and len(v.strip()) >= 4 and any(c.isalpha() for c in v)),
    }
    if schema is not None:
        out["constrained_leaves"] = _constrained_leaves(reference, schema)
    return out


def meets_create_thresholds(reference, schema, thresholds):
    """Is the reference rich enough to count as a complex Create example?

    Returns (ok, reasons). `thresholds` keys: min_leaves, min_depth, min_distinct_types,
    min_object_arrays. This only measures the INSTANCE; the description still has to pass
    the unchanged grounding (and, if enabled, round-trip) gates.
    """
    m = reference_metrics(reference)
    reasons = []
    if m["leaf_count"] < thresholds.get("min_leaves", 0):
        reasons.append(f"leaves {m['leaf_count']} < {thresholds['min_leaves']}")
    if m["depth"] < thresholds.get("min_depth", 0):
        reasons.append(f"depth {m['depth']} < {thresholds['min_depth']}")
    if m["distinct_leaf_types"] < thresholds.get("min_distinct_types", 0):
        reasons.append(f"types {m['distinct_leaf_types']} < {thresholds['min_distinct_types']}")
    if m["object_arrays"] < thresholds.get("min_object_arrays", 0):
        reasons.append(f"object arrays {m['object_arrays']} < {thresholds['min_object_arrays']}")
    return not reasons, reasons


def schema_can_support(schema, thresholds, max_depth=8):
    """Cheap pre-check, before spending model calls: could ANY valid instance be this rich?

    Looks at how many properties and how deep the schema can go (following local $refs,
    guarding against cycles). Open-ended objects (additionalProperties/patternProperties
    schemas) are treated as unbounded, so this only rules out schemas that clearly cannot
    reach the thresholds. Returns (ok, reason_or_None).
    """
    opts = _Walk(root=schema)
    state = {"props": 0, "depth": 0, "open": False, "visits": 0}

    def visit(sch, depth, seen):
        state["visits"] += 1
        sch = _resolve(sch, opts)
        if not isinstance(sch, dict) or depth > max_depth or state["visits"] > 5000:
            return
        state["depth"] = max(state["depth"], depth)
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(sch.get(key), list):
                for sub in sch[key]:
                    visit(sub, depth, seen)
        props = sch.get("properties")
        if isinstance(props, dict):
            state["props"] += len(props)
            for sub in props.values():
                ident = id(sub)
                if ident in seen:
                    continue
                visit(sub, depth + 1, seen | {ident})
        for key in ("additionalProperties", "patternProperties"):
            extra = sch.get(key)
            if isinstance(extra, dict) and extra:
                state["open"] = True
        items = sch.get("items")
        if isinstance(items, dict):
            visit(items, depth + 1, seen)

    visit(schema, 0, frozenset())
    if state["open"]:
        return True, None
    if state["depth"] < thresholds.get("min_depth", 0):
        return False, f"schema depth {state['depth']} < {thresholds['min_depth']}"
    if state["props"] < thresholds.get("min_leaves", 0):
        return False, f"schema has {state['props']} properties < {thresholds['min_leaves']}"
    return True, None


# ------------------------------------------------------------------ operation / error metadata

def operations_metrics(paths, kinds, original=None, changed_leaf_paths=None):
    """Shared metadata for a set of edits/errors given as paths and kinds."""
    branches = {p[0] for p in paths if p}
    return {
        "count": len(paths),
        "types": sorted(set(kinds)),
        "type_sequence": list(kinds),
        "max_depth": max((path_depth(p) for p in paths), default=0),
        "affected_fields": len(changed_leaf_paths) if changed_leaf_paths is not None else len(paths),
        "distinct_branches": len(branches),
        "different_branches": len(branches) > 1,
        "inside_array_items": any(isinstance(s, int) for p in paths for s in p),
    }


def modify_metrics(original, modified, spec):
    ops = spec.get("operations") or [spec]
    return operations_metrics([op["path"] for op in ops], [op["op"] for op in ops],
                              changed_leaf_paths=metrics.changed_paths(original, modified))


def fix_metrics(valid, erroneous, errors):
    return operations_metrics([e["path"] for e in errors], [e["kind"] for e in errors],
                              changed_leaf_paths=metrics.changed_paths(valid, erroneous))


def single_fix_metrics(valid, erroneous, info):
    """Metadata for a standard (single-error) fix, where only the mutation kind is known.

    The error location comes from the mutation info (field, plus nested field when there is
    one); the number of affected fields is still measured from the real difference.
    """
    changed = metrics.changed_paths(valid, erroneous)
    path = [info[k] for k in ("field", "nested_field") if info and info.get(k) is not None]
    if not path:
        path = list(next(iter(sorted(changed, key=lambda p: [str(s) for s in p])), ()))
    out = operations_metrics([path], [info["kind"] if info else "unknown"], changed_leaf_paths=changed)
    out["count"] = 1
    return out
