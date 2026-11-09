"""Controlled schema reuse for the scale-only experiment.

The original Benchson build made at most one example per task per schema. When the schema pool
cannot grow, this module lets the builder make several *independent* examples per schema:

* every attempt mints a fresh reference object with the unchanged standard prompt;
* a reference that equals one already known for that schema (published or new) is rejected;
* a per-schema cap, a per-schema failure limit and suspension keep the run balanced and unblocked.

Nothing here changes prompts, gates or token limits. It is active only with a `reuse` config block.
"""
import threading

from data_generation.published import TASKS, canon_hash, reference_hash, task_keys


class ReuseSettings:
    DEFAULTS = {
        "enabled": True,
        # at most this many NEW references (each feeds up to three tasks) per schema
        "max_new_references_per_schema": 2,
        # consecutive attempts that produced nothing before the schema is suspended
        "max_failures_per_schema": 3,
        # passes a suspended schema sits out, and how often it may be suspended before it is retired
        "suspend_passes": 1,
        "max_suspensions": 1,
        # hard bound on attempts (successful or not) per schema in one run
        "max_attempts_per_schema": 6,
        # name of a provider parameter (e.g. "seed") that receives a per-attempt integer; null = unused
        "sampling_seed_param": None,
        "reject_cross_schema_duplicates": False,
        "dedupe_pool_by_content": True,
        "pool_sanity_min_fraction": 0.95,
        # Where the schema pool comes from. "reconstructed" (default, unchanged) rebuilds the original partition from
        # data/; "published_train" uses the schemas of the published training rows (see data_generation/anchor.py).
        "pool_source": "reconstructed",
        # published_train only: the anchor file (default: published_train_schemas.json next to the exclusions file)
        "anchor_file": None,
        # published_train only: the size restriction, an explicit experimental choice (REQUIRED in that mode; no default)
        "anchor_max_schema_kb": None,
    }
    POOL_SOURCES = ("reconstructed", "published_train")

    def __init__(self, **values):
        self.__dict__.update(self.DEFAULTS)
        unknown = set(values) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown reuse setting(s): {sorted(unknown)}")
        self.__dict__.update(values)
        for key in ("max_new_references_per_schema", "max_failures_per_schema", "max_attempts_per_schema"):
            if int(self.__dict__[key]) < 1:
                raise ValueError(f"reuse.{key} must be >= 1")
        if int(self.max_suspensions) < 0 or int(self.suspend_passes) < 0:
            raise ValueError("reuse.suspend_passes / max_suspensions must be >= 0")
        if self.pool_source not in self.POOL_SOURCES:
            raise ValueError(f"reuse.pool_source must be one of {self.POOL_SOURCES}, not {self.pool_source!r}")
        if self.pool_source == "published_train":
            kb = self.anchor_max_schema_kb
            if isinstance(kb, bool) or not isinstance(kb, (int, float)) or kb <= 0:
                raise ValueError("reuse.anchor_max_schema_kb is required (a positive number) with pool_source "
                                 "'published_train': the size restriction is an explicit experimental choice")
        elif self.anchor_file is not None or self.anchor_max_schema_kb is not None:
            raise ValueError("reuse.anchor_file / anchor_max_schema_kb apply only to pool_source 'published_train'")

    @classmethod
    def from_config(cls, config):
        block = config.get("reuse")
        if not block:
            return None
        if not isinstance(block, dict):
            raise ValueError("reuse must be an object")
        block = {k: v for k, v in block.items() if not k.startswith("_")}
        settings = cls(**block)
        return settings if settings.enabled else None

    def as_dict(self):
        return {k: getattr(self, k) for k in self.DEFAULTS}


class SchemaLedger:
    """Per-schema bookkeeping: attempts, accepted references, failures, suspension, retirement."""

    def __init__(self, settings, published_refs=None):
        self.s = settings
        self.published_refs = published_refs or {}
        self.entries = {}
        self.events = []   # suspended / retired / capped, in order
        self._lock = threading.Lock()

    def entry(self, schema_id, schema_hash=None):
        if schema_id not in self.entries:
            self.entries[schema_id] = {
                "schema_id": schema_id, "schema_hash": schema_hash,
                "published_references": len(self.published_refs.get(schema_hash, ())),
                "attempts": 0, "accepted_references": 0, "failed_attempts": 0, "consecutive_failures": 0,
                "suspensions": 0, "suspended_until_pass": 0, "status": "active", "reason": None,
                "examples": {t: 0 for t in TASKS}}
        return self.entries[schema_id]

    def eligible(self, schema_id, pass_no):
        e = self.entries[schema_id]
        return (e["status"] != "retired" and e["accepted_references"] < self.s.max_new_references_per_schema
                and e["attempts"] < self.s.max_attempts_per_schema and pass_no >= e["suspended_until_pass"])

    def begin_attempt(self, schema_id):
        with self._lock:
            e = self.entries[schema_id]
            e["attempts"] += 1
            if e["status"] == "suspended":   # the suspension is over: it is being tried again
                e["status"] = "active"
            return e["attempts"]

    def finish_attempt(self, schema_id, accepted_tasks, pass_no, why_empty=None):
        """accepted_tasks: list of tasks that got a new example. Returns an event dict or None."""
        with self._lock:
            e = self.entries[schema_id]
            event = None
            if accepted_tasks:
                e["accepted_references"] += 1
                e["consecutive_failures"] = 0
                for t in accepted_tasks:
                    e["examples"][t] += 1
                if e["accepted_references"] >= self.s.max_new_references_per_schema:
                    e["status"], e["reason"] = "capped", "per-schema reference cap reached"
                    event = {"event": "capped", "schema_id": schema_id}
            else:
                e["failed_attempts"] += 1
                e["consecutive_failures"] += 1
                if e["consecutive_failures"] >= self.s.max_failures_per_schema:
                    e["consecutive_failures"] = 0
                    if e["suspensions"] < self.s.max_suspensions:
                        e["suspensions"] += 1
                        e["suspended_until_pass"] = pass_no + 1 + self.s.suspend_passes
                        e["status"], e["reason"] = "suspended", why_empty
                        event = {"event": "suspended", "schema_id": schema_id, "until_pass": e["suspended_until_pass"],
                                 "reason": why_empty}
                    else:
                        e["status"], e["reason"] = "retired", why_empty or "repeated failures"
                        event = {"event": "retired", "schema_id": schema_id, "reason": e["reason"]}
            if e["status"] not in ("retired", "capped") and e["attempts"] >= self.s.max_attempts_per_schema:
                e["status"], e["reason"] = "retired", "max attempts per schema reached"
                event = {"event": "retired", "schema_id": schema_id, "reason": e["reason"]}
            if event:
                event["pass"] = pass_no
                self.events.append(event)
            return event

    def retire(self, schema_id, reason, pass_no=None):
        """Takes a schema out of the run for good (idempotent). Returns the event, or None if already retired."""
        with self._lock:
            e = self.entries[schema_id]
            if e["status"] == "retired":
                return None
            e["status"], e["reason"] = "retired", reason
            event = {"event": "retired", "schema_id": schema_id, "reason": reason, "pass": pass_no}
            self.events.append(event)
            return event

    def order_for_pass(self, schema_ids, pass_no, rng):
        """Eligible schemas for this pass: fewest total (published + new) references first, ties shuffled."""
        ids = [i for i in schema_ids if self.eligible(i, pass_no)]
        rng.shuffle(ids)
        ids.sort(key=lambda i: self.entries[i]["published_references"] + self.entries[i]["accepted_references"])
        return ids

    def live(self, schema_ids):
        """Schemas that may still be tried now or after a suspension (not retired, capped or out of attempts)."""
        return [i for i in schema_ids
                if self.entries[i]["status"] in ("active", "suspended")
                and self.entries[i]["accepted_references"] < self.s.max_new_references_per_schema
                and self.entries[i]["attempts"] < self.s.max_attempts_per_schema]


class DuplicateIndex:
    """Remembers references and task contents (published + new) and flags repeats by JSON content."""

    def __init__(self, published_refs=None, published_keys=None):
        self.published_refs = {k: set(v) for k, v in (published_refs or {}).items()}
        self.published_keys = published_keys or {t: {"exact": set(), "content": set()} for t in TASKS}
        self.new_refs = {}                       # schema_hash -> {reference_hash: schema_id}
        self.new_keys = {t: {"exact": {}, "content": {}} for t in TASKS}
        self.stats = {"rejected_reference_same_as_published": 0, "rejected_reference_same_as_new": 0,
                      "rejected_task_exact_duplicate": 0, "cross_schema_content_duplicates": 0}
        self._lock = threading.Lock()

    def reference_kind(self, schema_hash, ref_hash):
        with self._lock:
            if ref_hash in self.published_refs.get(schema_hash, ()):
                return "published"
            if ref_hash in self.new_refs.get(schema_hash, {}):
                return "new"
        return None

    def note_rejected_reference(self, kind):
        with self._lock:
            self.stats[f"rejected_reference_same_as_{kind}"] += 1

    def register_reference(self, schema_hash, ref_hash, schema_id):
        """False when this schema already has that reference (a concurrent attempt got there first)."""
        with self._lock:
            known = self.new_refs.setdefault(schema_hash, {})
            if ref_hash in known or ref_hash in self.published_refs.get(schema_hash, ()):
                return False
            known[ref_hash] = schema_id
            return True

    def check_task(self, task, record, reject_cross_schema=False):
        """None if new; otherwise "exact" or "cross_schema" (content equal under another schema)."""
        exact, content = task_keys(task, record)
        with self._lock:
            if exact in self.published_keys[task]["exact"] or exact in self.new_keys[task]["exact"]:
                self.stats["rejected_task_exact_duplicate"] += 1
                return "exact"
            if content in self.published_keys[task]["content"] or content in self.new_keys[task]["content"]:
                self.stats["cross_schema_content_duplicates"] += 1
                return "cross_schema" if reject_cross_schema else None
        return None

    def register_task(self, task, record, schema_id):
        exact, content = task_keys(task, record)
        with self._lock:
            self.new_keys[task]["exact"][exact] = schema_id
            self.new_keys[task]["content"].setdefault(content, schema_id)
