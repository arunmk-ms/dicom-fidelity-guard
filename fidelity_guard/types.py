from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Operation(str, Enum):
    DENOISE = "DENOISE"
    DEID = "DEID"
    CROP = "CROP"
    SEGMENT = "SEGMENT"
    DISPLAY = "DISPLAY"


class Decision(str, Enum):
    RELEASE = "RELEASE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


def logistic(z: float) -> float:
    z = max(min(z, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class LayerResult:
    layer: str
    status: str  # "pass" | "fail" | "not_run"
    s: float = 0.0
    hard_fail: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    evidence_map: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def ran(self) -> bool:
        return self.status != "not_run"

    @classmethod
    def not_run(cls, layer: str, reason: str) -> "LayerResult":
        return cls(layer=layer, status="not_run", reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status,
            "s": round(float(self.s), 4),
            "hard_fail": self.hard_fail,
            "metrics": _jsonable(self.metrics),
            "reason": self.reason,
        }


@dataclass
class Report:
    operation: Operation
    decision: Decision
    risk: float
    results: list[LayerResult]
    source_uids: list[str]
    output_uids: list[str]

    def layer(self, name: str) -> LayerResult:
        return next(r for r in self.results if r.layer == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "decision": self.decision.value,
            "risk": round(float(self.risk), 4),
            "source_sop_instance_uids": self.source_uids,
            "output_sop_instance_uids": self.output_uids,
            "layers": [r.to_dict() for r in self.results],
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
