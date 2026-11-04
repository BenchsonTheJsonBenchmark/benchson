"""Standard/complex generation mix: configuration and integer quota allocation.

Config (all optional; adapt to configs/preview_complex_*.json):

    "generation": {
      "standard_ratio": 0.8,
      "complex_ratio": 0.2,
      "standard_modify_strategy": "original",
      "complex": {
        "modify": {"min_operations": 2, "max_operations": 3, "attempts": 5},
        "fix":    {"min_errors": 2, "max_errors": 3, "attempts": 5},
        "create": {"min_leaves": 10, "min_depth": 3, "min_distinct_types": 3, "min_object_arrays": 0},
        "max_failures_per_example": 5
      },
      "report_dir": null
    }

No "generation" block => `enabled` is False and the builder behaves exactly as before
(every example is built the legacy way, nothing extra is written).

Quotas are integers, computed with exact fractions and a fixed tie-break, so the same
request always gives the same allocation (no randomness, no float rounding surprises).
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

TASKS = ("create", "fix", "modify")
MODES = ("standard", "complex")

DEFAULT_COMPLEX = {
    "modify": {"min_operations": 2, "max_operations": 3, "attempts": 5},
    "fix": {"min_errors": 2, "max_errors": 3, "attempts": 5},
    "create": {"min_leaves": 10, "min_depth": 3, "min_distinct_types": 3, "min_object_arrays": 0},
    "max_failures_per_example": 5,
}

MODIFY_STRATEGIES = ("original", "recursive_weighted")


def _fraction(value, name):
    if isinstance(value, Fraction):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"generation.{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"generation.{name} must be a finite number >= 0, got {value!r}")
    return Fraction(str(value)).limit_denominator(100000)


def validate_ratios(standard_ratio, complex_ratio):
    """Returns the ratios as exact fractions; they must sum to 1."""
    s, c = _fraction(standard_ratio, "standard_ratio"), _fraction(complex_ratio, "complex_ratio")
    if abs(float(s + c) - 1.0) > 1e-6:
        raise ValueError(
            f"standard_ratio + complex_ratio must equal 1 (got {standard_ratio} + {complex_ratio})")
    return s, c


def _largest_remainder(total, shares, order):
    """Splits `total` into integers proportional to `shares` (exact Fractions).

    Floors first, then hands the leftover units to the largest fractional remainders.
    Ties go to the key that comes first in `order`, so the result is deterministic.
    """
    weight = sum(shares.values())
    if total <= 0 or weight == 0:
        return {k: 0 for k in shares}
    exact = {k: Fraction(total) * v / weight for k, v in shares.items()}
    base = {k: math.floor(v) for k, v in exact.items()}
    leftover = total - sum(base.values())
    ranked = sorted(exact, key=lambda k: (-(exact[k] - base[k]), order.index(k)))
    for k in ranked[:leftover]:
        base[k] += 1
    return base


def allocate_mix(task_counts, standard_ratio, complex_ratio):
    """{task: n} -> {task: {"standard": n_s, "complex": n_c}}, summing to n per task.

    The complex total over ALL tasks follows the requested ratio (so a tiny run such as
    4 examples at 75/25 really contains 1 complex example), then those complex examples
    are shared out between tasks in proportion to their sizes. Ties break by the order
    create, fix, modify (standard before complex for the mode split).
    """
    s_ratio, c_ratio = validate_ratios(standard_ratio, complex_ratio)
    tasks = [t for t in TASKS if t in task_counts] + [t for t in task_counts if t not in TASKS]
    sizes = {t: int(task_counts[t]) for t in tasks}
    total = sum(sizes.values())

    by_mode = _largest_remainder(total, {"standard": s_ratio, "complex": c_ratio}, list(MODES))
    complex_by_task = _largest_remainder(
        by_mode["complex"], {t: Fraction(n) for t, n in sizes.items()}, tasks)
    return {t: {"standard": sizes[t] - complex_by_task[t], "complex": complex_by_task[t]}
            for t in tasks}


def split_total(total, tasks):
    """Spreads `total` examples evenly over `tasks`; the remainder goes to the earlier tasks."""
    tasks = list(tasks)
    if not tasks:
        return {}
    base, extra = divmod(int(total), len(tasks))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(tasks)}


@dataclass(frozen=True)
class GenerationSettings:
    enabled: bool = False
    standard_ratio: Fraction = Fraction(1)
    complex_ratio: Fraction = Fraction(0)
    standard_modify_strategy: str = "recursive_weighted"
    complex: dict = field(default_factory=lambda: {k: (dict(v) if isinstance(v, dict) else v)
                                                   for k, v in DEFAULT_COMPLEX.items()})
    report_dir: str = None

    @classmethod
    def from_config(cls, config):
        block = config.get("generation")
        if block is None:
            return cls()
        if not isinstance(block, dict):
            raise ValueError("generation must be an object")
        s_raw = block.get("standard_ratio")
        c_raw = block.get("complex_ratio")
        if s_raw is None and c_raw is None:
            s_raw, c_raw = 0.8, 0.2
        elif s_raw is None:
            s_raw = 1 - float(c_raw)
        elif c_raw is None:
            c_raw = 1 - float(s_raw)
        s, c = validate_ratios(round(float(s_raw), 6), round(float(c_raw), 6))

        strategy = block.get("standard_modify_strategy", "original")
        if strategy not in MODIFY_STRATEGIES:
            raise ValueError(f"generation.standard_modify_strategy must be one of {MODIFY_STRATEGIES}")

        complex_cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_COMPLEX.items()}
        for key, override in (block.get("complex") or {}).items():
            if isinstance(complex_cfg.get(key), dict) and isinstance(override, dict):
                complex_cfg[key].update(override)
            else:
                complex_cfg[key] = override
        for lo, hi, name in (("min_operations", "max_operations", "modify"),
                             ("min_errors", "max_errors", "fix")):
            sect = complex_cfg[name]
            if not (1 <= sect[lo] <= sect[hi]):
                raise ValueError(f"generation.complex.{name}: need 1 <= {lo} <= {hi}")
        return cls(True, s, c, strategy, complex_cfg, block.get("report_dir"))

    def mix(self, task_counts):
        """Per-task standard/complex quotas for one split."""
        if not self.enabled:
            return {t: {"standard": int(n), "complex": 0} for t, n in task_counts.items()}
        return allocate_mix(task_counts, self.standard_ratio, self.complex_ratio)

    @property
    def modify_strategy(self):
        return self.standard_modify_strategy
