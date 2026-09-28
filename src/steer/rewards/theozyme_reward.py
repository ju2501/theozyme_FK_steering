from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Score:
    value: float
    valid: bool
    passed: bool
    terms: tuple
    error: str = ""

    def to_dict(self):
        return {"reward": self.value if self.valid else None, "valid": self.valid,
                "passed": self.passed, "terms": list(self.terms), "error": self.error}


class TheozymeReward:
    """Higher is better; zero means all constraints are within tolerance.

    Score = minus weighted mean squared, tolerance-adjusted violations.
    Fixed-only constraints remain diagnostics and are excluded from the steering
    objective. Pass/fail includes ALL constraints. Invalid structures get -inf.
    """
    def __init__(self, task):
        self.task = task

    def __call__(self, coordinates):
        terms, numerator, denominator = [], 0.0, 0.0
        try:
            for c in self.task.constraints:
                measured, violation = c.measure(coordinates)
                active = self.task.is_steerable(c)
                terms.append({"name": c.name, "kind": c.kind, "measured": measured,
                              "target": c.target, "violation": violation, "steerable": active})
                if active:
                    numerator += c.weight * violation ** 2
                    denominator += c.weight
            value = -numerator / denominator if denominator else 0.0
            if not math.isfinite(value):
                raise ValueError("Nonfinite reward")
        except (ValueError, OverflowError) as exc:
            return Score(float("-inf"), False, False, tuple(terms), str(exc))
        return Score(value, True, all(t["violation"] <= 1e-12 for t in terms), tuple(terms))
