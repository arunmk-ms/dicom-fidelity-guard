"""L3 - residual statistics (ENABLEMENT.md Section 7.1).

The residual R = S - O should look like the noise that was removed. Localized blob-like
structure in R means anatomy was added (negative blob) or removed (positive blob).
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy import ndimage

from ..types import LayerResult, logistic

DEFAULT_RADII_MM = (2.0, 4.0, 7.0)


def _robust_sd(x: np.ndarray) -> float:
    med = np.median(x)
    return max(1.4826 * float(np.median(np.abs(x - med))), 1e-9)


@lru_cache(maxsize=64)
def white_noise_dog_sd(s: float) -> float:
    """SD of the DoG response to unit-variance white noise (used as a physical noise floor)."""
    rng = np.random.default_rng(0)
    n = rng.standard_normal((256, 256))
    dog = ndimage.gaussian_filter(n, s) - ndimage.gaussian_filter(n, 3 * s)
    return float(dog[32:-32, 32:-32].std())


def blob_response(img: np.ndarray, mask: np.ndarray, spacing_mm: float,
                  radii_mm=DEFAULT_RADII_MM, normalizer: list[float] | None = None,
                  sigma: float = 0.0):
    """Center-surround (difference of Gaussians) response normalized by a robust SD.

    The normalizer is max(robust SD of the response, sigma x white-noise DoG SD) so that nearly
    noiseless inputs cannot inflate the statistic.
    Returns (response map = max |z| over scales, signed map at the arg-max scale, normalizers).
    """
    best = np.zeros(img.shape)
    signed = np.zeros(img.shape)
    norms = []
    for i, r in enumerate(radii_mm):
        s = max(r / np.sqrt(2) / spacing_mm, 0.7)
        dog = np.stack([ndimage.gaussian_filter(f, s) - ndimage.gaussian_filter(f, 3 * s) for f in img])
        if normalizer is not None:
            sd = normalizer[i]
        else:
            sd = max(_robust_sd(dog[mask]), sigma * white_noise_dog_sd(round(s, 2)))
        norms.append(sd)
        z = dog / sd
        upd = np.abs(z) > best
        best[upd] = np.abs(z)[upd]
        signed[upd] = z[upd]
    return best, signed, norms


def interior(mask: np.ndarray, spacing_mm: float, margin_mm: float = 10.0) -> np.ndarray:
    it = max(int(round(margin_mm / spacing_mm)), 1)
    return np.stack([ndimage.binary_erosion(m, iterations=it) if m.sum() > 0 else m for m in mask])


def residual_stats(src: np.ndarray, out: np.ndarray, mask: np.ndarray, spacing_mm: float,
                   sigma: float, blob_thr: float = 7.0, edge_thr: float = 0.45) -> LayerResult:
    name = "L3_residual"
    R = src - out
    inner = interior(mask, spacing_mm)
    if inner.sum() < 50:
        inner = mask
    resp, signed, _ = blob_response(R, inner, spacing_mm, sigma=sigma)
    resp = np.where(inner, resp, 0.0)
    lab, n = ndimage.label(resp > blob_thr)
    blobs = []
    for i in range(1, n + 1):
        comp = lab == i
        idx = np.argwhere(comp)
        peak = tuple(idx[np.argmax(resp[comp])])
        blobs.append({"frame": int(peak[0]), "row": int(peak[1]), "col": int(peak[2]),
                      "peak": float(resp[peak]),
                      "kind": "removed" if signed[peak] > 0 else "added"})
    m_blob = float(resp.max()) if resp.size else 0.0

    grad = np.stack([np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1)) for f in src])
    a, b = np.abs(R)[inner], grad[inner]
    c_edge = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    # Lag-1 autocorrelation of the residual (informational in the reference build).
    rx = R[:, :, 1:][inner[:, :, 1:]]
    rx0 = R[:, :, :-1][inner[:, :, 1:]]
    rho1 = float(np.corrcoef(rx, rx0)[0, 1]) if rx.std() > 0 and rx0.std() > 0 else 0.0

    s = max(logistic((m_blob - blob_thr) / 0.5), logistic((c_edge - edge_thr) / 0.05))
    return LayerResult(
        name, "fail" if s >= 0.5 else "pass", s, False,
        {"max_blob_response": m_blob, "blob_threshold": blob_thr, "n_blobs": len(blobs),
         "blobs": blobs[:20], "edge_correlation": c_edge, "edge_threshold": edge_thr,
         "lag1_autocorrelation": rho1},
        evidence_map=(resp / max(blob_thr, 1e-9)).astype(np.float32),
    )
