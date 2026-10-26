"""L5 - quantitative invariants (ENABLEMENT.md 7.3).

Paired source/output comparison of ROI means in automatically selected homogeneous patches.
The reference build uses homogeneous patches instead of organ masks so that it needs no
segmentation model; organ ROIs (e.g., TotalSegmentator) are a drop-in replacement.
"""
from __future__ import annotations

import numpy as np

from ..types import LayerResult, logistic

ABS_TOL = {"CT": 5.0}  # HU
REL_TOL = 0.05  # other modalities: 5 % of the source ROI mean


def quantitative_invariants(src: np.ndarray, out: np.ndarray, mask: np.ndarray, spacing_mm: float,
                            modality: str, sigma: float, patch_mm: float = 8.0,
                            homogeneous_quantile: float = 0.3, corr_factor: float = 3.0) -> LayerResult:
    name = "L5_quant"
    p = max(int(round(patch_mm / spacing_mm)), 4)
    rows = []
    while True:
        rows = []
        for k in range(src.shape[0]):
            for r in range(0, src.shape[1] - p + 1, p):
                for c in range(0, src.shape[2] - p + 1, p):
                    if not mask[k, r:r + p, c:c + p].all():
                        continue
                    ps, po = src[k, r:r + p, c:c + p], out[k, r:r + p, c:c + p]
                    rows.append((k, r, c, ps.mean(), po.mean(), ps.std()))
        if len(rows) >= 12 or p <= 3:
            break
        p = max(p // 2, 3)  # small images (e.g., 64x64 MR): fall back to smaller patches
    if len(rows) < 4:
        if not mask.any():
            return LayerResult.not_run(name, "empty body mask")
        # last resort: whole-body mean comparison (global drift only)
        ms_, mo_ = float(src[mask].mean()), float(out[mask].mean())
        tol_ = ABS_TOL.get(modality, max(REL_TOL * abs(ms_), sigma))
        g = abs(mo_ - ms_) / tol_
        s = logistic((g - 1.0) / 0.2)
        return LayerResult(name, "fail" if s >= 0.5 else "pass", s, False,
                           {"n_rois": 0, "mode": "whole-body", "mean_delta": mo_ - ms_, "global_ratio": g})
    arr = np.array(rows)
    homo = arr[arr[:, 5] <= np.quantile(arr[:, 5], homogeneous_quantile)]
    ms, mo, sd = homo[:, 3], homo[:, 4], homo[:, 5]
    delta = mo - ms
    tol = np.full_like(ms, ABS_TOL[modality]) if modality in ABS_TOL else np.maximum(REL_TOL * np.abs(ms), sigma)
    allowance = corr_factor * sd / p  # sampling error of a p x p mean with correlated noise
    ratio_local = np.abs(delta) / (tol + allowance)
    worst = int(np.argmax(ratio_local))
    global_ratio = abs(float(delta.mean())) / float(tol.mean())
    m = max(float(ratio_local.max()), global_ratio)
    s = logistic((m - 1.0) / 0.2)
    return LayerResult(
        name, "fail" if s >= 0.5 else "pass", s, False,
        {"n_rois": int(len(homo)), "mean_delta": float(delta.mean()),
         "max_abs_delta": float(np.abs(delta).max()),
         "worst_roi": {"frame": int(homo[worst, 0]), "row": int(homo[worst, 1]), "col": int(homo[worst, 2]),
                       "source_mean": float(ms[worst]), "output_mean": float(mo[worst])},
         "max_local_ratio": float(ratio_local.max()), "global_ratio": global_ratio,
         "tolerance": "5 HU" if modality in ABS_TOL else "5 % relative"},
    )
