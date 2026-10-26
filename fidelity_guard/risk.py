"""Risk aggregation and routing (ENABLEMENT.md Section 11)."""
from __future__ import annotations

from .types import Decision, LayerResult

DEFAULT_WEIGHTS = {
    "L1_scope_bounded": 0.6,
    "L3_residual": 0.6,
    "L4_findings": 1.0,
    "L5_quant": 0.7,
}
DEFAULT_THRESHOLDS = {"release": 0.2, "block": 0.6}


def aggregate(results: list[LayerResult], weights: dict[str, float] | None = None,
              required: set[str] | None = None) -> float:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    required = required or set()
    if any(r.hard_fail for r in results):
        return 1.0
    prod = 1.0
    for r in results:
        if r.ran:
            prod *= 1.0 - weights.get(r.layer, 1.0) * r.s
    risk = 1.0 - prod
    if any((not r.ran) and r.layer in required for r in results):
        risk = max(risk, DEFAULT_THRESHOLDS["release"])  # reduced assurance -> human review
    return risk


def route(risk: float, thresholds: dict[str, float] | None = None) -> Decision:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if risk < t["release"]:
        return Decision.RELEASE
    if risk < t["block"]:
        return Decision.REVIEW
    return Decision.BLOCK
