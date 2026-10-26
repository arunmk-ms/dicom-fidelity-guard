"""Common pre-processing (ENABLEMENT.md Section 4)."""
from __future__ import annotations

import numpy as np
import pywt
from pydicom.dataset import Dataset
from scipy import ndimage
from skimage.filters import threshold_otsu


def stored_pixels(ds: Dataset) -> np.ndarray:
    """Stored pixel values, always shaped (frames, rows, cols[, samples])."""
    arr = ds.pixel_array
    frames = int(ds.get("NumberOfFrames", 1) or 1)
    if frames == 1:
        arr = arr[None, ...]
    return arr


def to_physical(ds: Dataset) -> np.ndarray:
    """Physical values (HU for CT, rescaled units otherwise) as float64 (frames, rows, cols).

    Color images are converted to luminance so that statistical layers operate on one channel.
    """
    arr = stored_pixels(ds).astype(np.float64)
    if int(ds.get("SamplesPerPixel", 1)) == 3:
        arr = arr @ np.array([0.299, 0.587, 0.114])
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def pixel_spacing_mm(ds: Dataset) -> tuple[float, float]:
    """(row spacing, column spacing) in mm; falls back to ImagerPixelSpacing, then 1 mm."""
    for kw in ("PixelSpacing", "ImagerPixelSpacing"):
        v = ds.get(kw)
        if v is not None and len(v) == 2:
            return float(v[0]), float(v[1])
    return 1.0, 1.0


def estimate_noise(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Donoho-Johnstone robust noise estimate from the finest diagonal wavelet subband.

    sigma = median(|HH|) / 0.6745, computed per frame (db4, one level); median across frames.
    """
    sigmas = []
    for k in range(img.shape[0]):
        _, (_, _, hh) = pywt.dwt2(img[k], "db4")
        vals = hh.ravel()
        if mask is not None:
            m = mask[k][::2, ::2]
            h, w = min(m.shape[0], hh.shape[0]), min(m.shape[1], hh.shape[1])
            sel = hh[:h, :w][m[:h, :w]]
            if sel.size > 16:
                vals = sel
        sigmas.append(np.median(np.abs(vals)) / 0.6745)
    return max(float(np.median(sigmas)), 1e-6)


def body_mask(img: np.ndarray, modality: str) -> np.ndarray:
    """Largest connected foreground component per frame, holes filled."""
    out = np.zeros(img.shape, dtype=bool)
    for k in range(img.shape[0]):
        f = img[k]
        if modality == "CT":
            fg = f > -500
        else:
            try:
                fg = f > threshold_otsu(f)
            except ValueError:
                fg = np.ones_like(f, dtype=bool)
        lab, n = ndimage.label(fg)
        if n == 0:
            out[k] = True
            continue
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        largest = lab == (int(np.argmax(sizes)) + 1)
        out[k] = ndimage.binary_fill_holes(largest)
    if out.sum() < 0.05 * out.size:
        out[:] = True
    return out
