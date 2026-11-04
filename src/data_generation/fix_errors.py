"""Complex mode of the fix task: several schema violations in one JSON instance.

`inject_errors` takes a schema-valid object and corrupts it in 2-3 places at once. The
violations differ in kind (wrong type / out of range / missing required field / bad enum /
bad array / ...), sit at different depths and in different branches, and may be inside
objects that are list items. The standard single-error injector in
tasks/error_correction.py is untouched; this module only adds the multi-error path.

Every error is a *record* (kind, path, original value, corrupted value) so that the
expected correction is known and can be checked by code, not guessed:

- each error alone must already make the object invalid (so none is "decorative"),
- no two errors touch the same place or an ancestor/descendant of each other (so fixing
  one can never hide, undo or change another),
- putting the recorded original values back must reproduce the valid object exactly,
- nothing outside the recorded places may differ (`verify_fix`).
"""

import copy

import jsonschema

from evaluations import metrics
from data_generation.modifications import _get_at, _resolve, _set_at, _Walk, paths_conflict
from data_generation.tasks.error_correction import _FORMAT_INVALIDS, _WRONG_TYPE_VALUES

_MAX_ARRAY_ELEMENTS = 3
_MAX_POOL = 200
_ABSENT = object()
_KIND_WEIGHT = {"null_non_nullable": 0.4, "wrong_type_string": 0.6}


def inject_errors(valid_json, schema, rng, n_errors=(2, 3), attempts=5):
    """Returns (erroneous_json, errors) or None.

    `n_errors=(lo, hi)`: each attempt draws a target in [lo, hi]; an attempt with at least
    `lo` compatible errors is accepted. `errors` is the list of error records.
    """
    lo, hi = n_errors
    if not isinstance(valid_json, dict) or lo < 1 or hi < lo:
        return None
    root = _Walk(arrays=True, root=schema)
    pool = list(_walk(valid_json, schema, [], root))
    if len(pool) < lo:
        return None

    for _ in range(attempts):
        target = rng.randint(lo, hi)
        chosen = _select(valid_json, schema, pool, target, rng)
        if len(chosen) >= lo:
            erroneous = copy.deepcopy(valid_json)
            for cand in chosen:
                cand["apply"](erroneous)
            errors = [_error_record(valid_json, erroneous, c) for c in chosen]
            return erroneous, errors
    return None


# ------------------------------------------------------------------ candidate discovery

def _walk(obj, schema, path, opts, depth=0):
    sch = _resolve(schema, opts)
    props = sch.get("properties", {}) if isinstance(sch.get("properties"), dict) else {}
    required = sch.get("required", []) if isinstance(sch.get("required"), list) else []

    for field in required:
        if field in obj:
            yield _cand("remove_required", path + [field], path,
                        lambda o, p=path + [field]: _del(o, p))

    for key, value in obj.items():
        if key not in props:
            continue
        ps = _resolve(props[key], opts)
        yield from _field_candidates(path + [key], value, ps)

    if sch.get("additionalProperties") is False:
        extra = path + ["__invalid_field__"]
        yield _cand("additional_property", extra, path,
                    lambda o, p=extra: _set_at(o, p, "invalid"))

    for key, value in obj.items():
        if key not in props:
            continue
        if isinstance(value, dict):
            yield from _walk(value, props[key], path + [key], opts, depth + 1)
        elif isinstance(value, list):
            items = _resolve(props[key], opts).get("items")
            for idx, element in enumerate(value[:_MAX_ARRAY_ELEMENTS]):
                if isinstance(element, dict):
                    yield from _walk(element, items, path + [key, idx], opts, depth + 1)


def _field_candidates(path, value, ps):
    """Violations of one field's own constraints (same kinds as the standard injector)."""
    ptype = ps.get("type")
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)

    def put(new):
        return lambda o, p=path, v=new: _set_at(o, p, copy.deepcopy(v))

    if "enum" in ps:
        yield _cand("invalid_enum", path, path, put(str(value) + "_INVALID"))
    if ptype in ("number", "integer") and is_num:
        yield _cand("wrong_type_number", path, path, put("not-a-number"))
    if ptype == "boolean" and isinstance(value, bool):
        yield _cand("wrong_type_boolean", path, path, put("not-a-bool"))
    if ptype == "string" and isinstance(value, str):
        yield _cand("wrong_type_string", path, path, put(99999))
        if ps.get("minLength", 0) > 0 and value:
            yield _cand("empty_string", path, path, put(""))
        if isinstance(ps.get("maxLength"), int):
            yield _cand("string_too_long", path, path, put("x" * (ps["maxLength"] + 5)))
    if is_num:
        if isinstance(ps.get("minimum"), (int, float)):
            yield _cand("below_minimum", path, path, put(ps["minimum"] - 1))
        if isinstance(ps.get("exclusiveMinimum"), (int, float)) and not isinstance(ps["exclusiveMinimum"], bool):
            yield _cand("below_exclusive_minimum", path, path, put(ps["exclusiveMinimum"] - 1))
        if isinstance(ps.get("maximum"), (int, float)):
            yield _cand("above_maximum", path, path, put(ps["maximum"] + 1))
        if isinstance(ps.get("exclusiveMaximum"), (int, float)) and not isinstance(ps["exclusiveMaximum"], bool):
            yield _cand("above_exclusive_maximum", path, path, put(ps["exclusiveMaximum"] + 1))
    if isinstance(ptype, str) and ptype != "null":
        yield _cand("null_non_nullable", path, path, put(None))
    if isinstance(value, list) and ptype == "array":
        items = ps.get("items")
        item_type = items.get("type") if isinstance(items, dict) else None
        if isinstance(item_type, list):
            item_type = next((t for t in item_type if isinstance(t, str) and t != "null"), None)
        if isinstance(item_type, str) and value:
            wrong = _WRONG_TYPE_VALUES.get(item_type, "wrong")
            yield _cand("wrong_array_item_type", path, path, put([wrong] + value[1:]))
        min_items = ps.get("minItems", 0)
        if isinstance(min_items, int) and min_items >= 1 and len(value) >= min_items:
            yield _cand("below_min_items", path, path, put(value[:min_items - 1]))
        max_items = ps.get("maxItems")
        if isinstance(max_items, int):
            extra = value[0] if value else "extra"
            yield _cand("above_max_items", path, path,
                        put(value + [copy.deepcopy(extra)] * (max_items - len(value) + 1)))
    fmt = ps.get("format")
    if fmt in _FORMAT_INVALIDS and isinstance(value, str):
        yield _cand("invalid_format", path, path, put(_FORMAT_INVALIDS[fmt]))


def _cand(kind, path, error_path, apply_fn):
    return {"kind": kind, "path": list(path), "error_path": list(error_path), "apply": apply_fn}


def _del(obj, path):
    cur = obj
    for k in path[:-1]:
        cur = cur[k]
    del cur[path[-1]]


# ------------------------------------------------------------------ selection

def _select(valid, schema, pool, target, rng):
    """Greedy multi-pass choice, like the modify selection: variety preferred, never required.

    Deeper violations are favoured (exponential race with weight = path length), and the
    two cheapest-to-spot kinds (null / plain wrong string type) are down-weighted so they
    do not crowd out the more informative violations."""
    timed = sorted((rng.expovariate(1.0) / (max(len(c["path"]), 1) * _KIND_WEIGHT.get(c["kind"], 1.0)), i)
                   for i, c in enumerate(pool))
    ordered = [pool[i] for _, i in timed][:_MAX_POOL]
    chosen, kinds, branches = [], set(), set()

    for pass_no in (0, 1, 2):
        for cand in ordered:
            if len(chosen) >= target:
                return chosen
            if any(cand is c for c in chosen):
                continue
            if any(paths_conflict(cand["path"], c["path"]) for c in chosen):
                continue
            if pass_no == 0 and (cand["kind"] in kinds or cand["path"][0] in branches):
                continue
            if pass_no == 1 and cand["kind"] in kinds:
                continue
            trial = copy.deepcopy(valid)
            try:
                cand["apply"](trial)
            except (KeyError, IndexError, TypeError):
                continue
            if metrics.schema_compliance(trial, schema)[0] != 0:   # must break the schema by itself
                continue
            chosen.append(cand)
            kinds.add(cand["kind"])
            branches.add(cand["path"][0])
    return chosen


# ------------------------------------------------------------------ records and verification

def _value_at(obj, path):
    try:
        return _get_at(obj, path)
    except (KeyError, IndexError, TypeError):
        return _ABSENT


def _error_record(valid, erroneous, cand):
    before, after = _value_at(valid, cand["path"]), _value_at(erroneous, cand["path"])
    return {
        "kind": cand["kind"],
        "path": cand["path"],
        "error_path": cand["error_path"],
        "original_present": before is not _ABSENT,
        "original": None if before is _ABSENT else copy.deepcopy(before),
        "corrupted_present": after is not _ABSENT,
        "corrupted": None if after is _ABSENT else copy.deepcopy(after),
    }


def apply_error_record(obj, record):
    """Redoes one recorded corruption on `obj` (in place) from the record's data alone."""
    if record["corrupted_present"]:
        _set_at(obj, record["path"], copy.deepcopy(record["corrupted"]))
    else:
        _del(obj, record["path"])


def restore_error_record(obj, record):
    """Puts the recorded original value back (the correction a solver is expected to make)."""
    if record["original_present"]:
        _set_at(obj, record["path"], copy.deepcopy(record["original"]))
    else:
        _del(obj, record["path"])


def _violation_paths(obj, schema):
    try:
        validator = jsonschema.validators.validator_for(schema)(schema)
        return [tuple(e.absolute_path) for e in validator.iter_errors(obj)]
    except Exception:
        return None


def _covers(found, intended):
    intended = tuple(intended)
    return (found == intended
            or (len(found) > len(intended) and found[:len(intended)] == intended)
            or (0 < len(found) < len(intended) and intended[:len(found)] == found))


def verify_fix(valid, erroneous, errors, schema):
    """Programmatic checks of a complex fix example. Returns {"ok", "checks", "problems", "manual_review"}.

    checks (True/False, or None when it cannot be evaluated):
      original_valid                the target validates against the schema
      corrupted_invalid             the corrupted object violates the schema
      error_count                   there are as many records as claimed (>= 1)
      paths_disjoint                no two errors touch the same place
      each_error_alone_invalid      every recorded error breaks the schema on its own
      each_error_reported           the validator reports a violation at/near every error
      restore_recovers_valid        putting the recorded originals back gives exactly `valid`
      replay_matches                redoing the recorded corruptions on `valid` gives `erroneous`
      no_unintended_changes         every differing leaf lies on an error path
    """
    checks = {
        "original_valid": metrics.schema_compliance(valid, schema)[0] == 1,
        "corrupted_invalid": metrics.schema_compliance(erroneous, schema)[0] == 0,
        "error_count": len(errors) >= 1,
        "paths_disjoint": not any(paths_conflict(a["path"], b["path"])
                                  for i, a in enumerate(errors) for b in errors[i + 1:]),
    }

    alone, replayed, restored = True, copy.deepcopy(valid), copy.deepcopy(erroneous)
    try:
        for rec in errors:
            single = copy.deepcopy(valid)
            apply_error_record(single, rec)
            alone = alone and metrics.schema_compliance(single, schema)[0] == 0
            apply_error_record(replayed, rec)
            restore_error_record(restored, rec)
    except (KeyError, IndexError, TypeError):
        alone, replayed, restored = False, None, None
    checks["each_error_alone_invalid"] = alone
    checks["replay_matches"] = replayed == erroneous
    checks["restore_recovers_valid"] = restored == valid

    found = _violation_paths(erroneous, schema)
    checks["each_error_reported"] = None if found is None else all(
        any(_covers(f, rec["error_path"]) for f in found) for rec in errors)

    paths = [tuple(r["path"]) for r in errors]
    checks["no_unintended_changes"] = all(
        any(paths_conflict(leaf, p) for p in paths)
        for leaf in metrics.changed_paths(valid, erroneous))

    problems = [name for name, ok in checks.items() if ok is False]
    return {"ok": not problems, "checks": checks, "problems": problems,
            "manual_review": ["the error set is realistic/natural for the schema's domain"]}
