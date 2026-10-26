"""L1 - change-scope enforcement (ENABLEMENT.md Section 5)."""
from __future__ import annotations

import hashlib

import numpy as np

from ..types import LayerResult, logistic


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def exact_scope(src_stored: np.ndarray, out_stored: np.ndarray, allowed: np.ndarray | None) -> LayerResult:
    """Pixels outside the allowed mask must be bit-identical (hard fail otherwise).

    allowed: bool array (frames, rows, cols) or None (= nothing may change).
    """
    name = "L1_scope_exact"
    if src_stored.shape != out_stored.shape:
        return LayerResult(name, "fail", 1.0, True,
                           {"source_shape": src_stored.shape, "output_shape": out_stored.shape},
                           "pixel matrix shape changed")
    if allowed is None:
        allowed = np.zeros(src_stored.shape[:3], dtype=bool)
    outside = ~allowed
    if src_stored.ndim == 4:  # color: broadcast mask over samples
        outside = np.broadcast_to(outside[..., None], src_stored.shape)
    changed = (src_stored != out_stored) & outside
    n_changed = int(changed.sum())
    h_src, h_out = _sha256(src_stored[outside]), _sha256(out_stored[outside])
    ok = n_changed == 0 and h_src == h_out
    evidence = changed.any(axis=-1) if changed.ndim == 4 else changed
    return LayerResult(
        name, "pass" if ok else "fail", 0.0 if ok else 1.0, not ok,
        {"pixels_changed_outside_allowed": n_changed,
         "allowed_fraction": float(allowed.mean()),
         "sha256_outside_source": h_src, "sha256_outside_output": h_out},
        "" if ok else f"{n_changed} pixels changed outside the allowed region",
        evidence_map=evidence.astype(np.float32),
    )


def bounded_change(src: np.ndarray, out: np.ndarray, sigma: float, mask: np.ndarray,
                   k: float = 6.0, percentile: float = 99.9) -> LayerResult:
    """For whole-image operations: changes should be on the order of the noise level.

    q = P99.9(|O - S| / sigma) within the body mask; s = logistic((q - k) / 0.5).
    """
    name = "L1_scope_bounded"
    if src.shape != out.shape:
        return LayerResult(name, "fail", 1.0, True, {}, "image grid changed")
    z = np.abs(out - src) / sigma
    q = float(np.percentile(z[mask], percentile))
    s = logistic((q - k) / 0.5)
    return LayerResult(name, "fail" if s >= 0.5 else "pass", s, False,
                       {"p999_abs_change_over_sigma": q, "k": k}, evidence_map=z.astype(np.float32))
