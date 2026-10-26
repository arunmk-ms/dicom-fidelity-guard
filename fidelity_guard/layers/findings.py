"""L4 - finding preservation with an independent, training-free detector (ENABLEMENT.md 7.2).

The reference detector is a multi-scale center-surround (difference-of-Gaussians) contrast
detector. It shares no weights or training data with any AI model under test. The same detector
and the same normalizers (computed on the source) are applied to source and output; the finding
sets are matched with the Hungarian algorithm. Unmatched findings are confirmed with a hysteresis
test so that near-threshold structures are not reported as added/removed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from ..types import LayerResult
from .residual import blob_response, interior


def detect(resp: np.ndarray, signed: np.ndarray, inner: np.ndarray, thr: float, spacing_mm: float):
    findings = []
    lab, n = ndimage.label((resp > thr) & inner)
    for i in range(1, n + 1):
        comp = lab == i
        idx = np.argwhere(comp)
        peak = tuple(idx[np.argmax(resp[comp])])
        area_mm2 = float(comp.sum()) * spacing_mm ** 2
        findings.append({
            "frame": int(peak[0]), "row": int(peak[1]), "col": int(peak[2]),
            "peak": float(resp[peak]), "polarity": 1 if signed[peak] > 0 else -1,
            "area_mm2": area_mm2, "eq_diam_mm": float(2 * np.sqrt(area_mm2 / np.pi)),
        })
    return findings


def _local_max(resp: np.ndarray, f: dict, rad: int) -> float:
    k, r, c = f["frame"], f["row"], f["col"]
    win = resp[k, max(r - rad, 0): r + rad + 1, max(c - rad, 0): c + rad + 1]
    return float(win.max()) if win.size else 0.0


def match(fs: list[dict], fo: list[dict], spacing_mm: float, min_dist_mm: float = 5.0):
    if not fs or not fo:
        return [], list(range(len(fs))), list(range(len(fo)))
    cost = np.full((len(fs), len(fo)), 1e6)
    for i, a in enumerate(fs):
        for j, b in enumerate(fo):
            if a["frame"] != b["frame"] or a["polarity"] != b["polarity"]:
                continue
            d = spacing_mm * np.hypot(a["row"] - b["row"], a["col"] - b["col"])
            if d <= max(min_dist_mm, 0.5 * max(a["eq_diam_mm"], b["eq_diam_mm"])):
                cost[i, j] = d
    ri, ci = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(ri, ci) if cost[i, j] < 1e6]
    mi = {i for i, _ in pairs}
    mj = {j for _, j in pairs}
    return pairs, [i for i in range(len(fs)) if i not in mi], [j for j in range(len(fo)) if j not in mj]


def finding_preservation(src: np.ndarray, out: np.ndarray, mask: np.ndarray, spacing_mm: float,
                         sigma: float, det_thr: float = 7.0, hysteresis: float = 0.5,
                         min_dist_mm: float = 5.0) -> LayerResult:
    name = "L4_findings"
    inner = interior(mask, spacing_mm)
    if inner.sum() < 50:
        inner = mask
    rs, ss, norms = blob_response(src, inner, spacing_mm, sigma=sigma)
    ro, so, _ = blob_response(out, inner, spacing_mm, normalizer=norms)
    fs = detect(rs, ss, inner, det_thr, spacing_mm)
    fo = detect(ro, so, inner, det_thr, spacing_mm)
    pairs, un_s, un_o = match(fs, fo, spacing_mm, min_dist_mm)
    rad = max(int(round(min_dist_mm / spacing_mm)), 1)

    removed = [fs[i] for i in un_s if _local_max(ro, fs[i], rad) < hysteresis * fs[i]["peak"]]
    added = [fo[j] for j in un_o if _local_max(rs, fo[j], rad) < hysteresis * fo[j]["peak"]]
    rel_changes = [abs(fo[j]["peak"] - fs[i]["peak"]) / fs[i]["peak"] for i, j in pairs]

    changed = len(added) + len(removed)
    s = 1.0 if changed else 0.0
    return LayerResult(
        name, "fail" if changed else "pass", s, False,
        {"n_source_findings": len(fs), "n_output_findings": len(fo), "n_matched": len(pairs),
         "n_added": len(added), "n_removed": len(removed),
         "added": added[:10], "removed": removed[:10],
         "max_rel_peak_change_matched": max(rel_changes) if rel_changes else 0.0,
         "detector": "multi-scale DoG contrast (training-free)", "threshold": det_thr},
        "" if not changed else f"{len(added)} added / {len(removed)} removed findings",
    )
