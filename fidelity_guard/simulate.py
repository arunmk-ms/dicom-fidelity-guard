"""Stand-in AI operations and controlled defect injection (ENABLEMENT.md Section 12.1).

These functions let the Guard be exercised on real DICOM files without a commercial AI product:
* ``denoise`` is a non-local-means denoiser standing in for an AI enhancement model.
* ``deidentify`` / ``crop`` are correct reference implementations of those operations.
* ``add_lesion`` / ``shift_values`` / ``perturb_ipp`` create the hallucination types to detect.
"""
from __future__ import annotations

import copy

import numpy as np
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from skimage.restoration import denoise_nl_means

from .layers.deid import PROFILE
from .preprocess import stored_pixels


def _stored_dtype(ds: Dataset) -> np.dtype:
    bits = int(ds.BitsAllocated)
    signed = int(ds.get("PixelRepresentation", 0)) == 1
    return np.dtype(f"<{'i' if signed else 'u'}{bits // 8}")


def with_pixels(ds: Dataset, phys: np.ndarray, new_uids: bool = True) -> Dataset:
    """Copy ds and replace its pixels with physical values (inverse rescale, clip, cast)."""
    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise ValueError("with_pixels supports single-sample images only")
    if not ds.file_meta.TransferSyntaxUID.is_little_endian or ds.file_meta.TransferSyntaxUID.is_compressed:
        raise ValueError("use an uncompressed little-endian source")
    d = copy.deepcopy(ds)
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    dt = _stored_dtype(ds)
    lo, hi = 0, 2 ** int(ds.BitsStored) - 1
    if dt.kind == "i":
        lo, hi = -(2 ** (int(ds.BitsStored) - 1)), 2 ** (int(ds.BitsStored) - 1) - 1
    stored = np.clip(np.rint((phys - intercept) / slope), lo, hi).astype(dt)
    if int(ds.get("NumberOfFrames", 1) or 1) == 1:
        stored = stored[0]
    d.PixelData = stored.tobytes()
    if new_uids:
        d.SOPInstanceUID = generate_uid()
        d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
        d.SeriesInstanceUID = generate_uid()
    return d


def denoise(phys: np.ndarray, sigma: float, strength: float = 0.8) -> np.ndarray:
    return np.stack([
        denoise_nl_means(f, h=strength * sigma, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True)
        for f in phys
    ])


def add_lesion(phys: np.ndarray, frame: int, row: int, col: int, radius_px: float, contrast: float) -> np.ndarray:
    out = phys.copy()
    rr, cc = np.mgrid[: phys.shape[1], : phys.shape[2]]
    d = np.hypot(rr - row, cc - col)
    out[frame] += contrast * 0.5 * (1 - np.tanh((d - radius_px) / 1.0))
    return out


def shift_values(phys: np.ndarray, delta: float, center=None, radius_px: float | None = None) -> np.ndarray:
    out = phys.copy()
    if center is None:
        return out + delta
    k, r, c = center
    rr, cc = np.mgrid[: phys.shape[1], : phys.shape[2]]
    out[k][np.hypot(rr - r, cc - c) <= radius_px] += delta
    return out


def deidentify(ds: Dataset, mask: np.ndarray | None = None, uid_map: dict | None = None) -> Dataset:
    """Correct reference de-identification: PS3.15 subset + redaction of masked pixels."""
    uid_map = {} if uid_map is None else uid_map
    d = copy.deepcopy(ds)
    for kw, action in PROFILE.items():
        if kw not in d:
            continue
        if action == "X":
            delattr(d, kw)
        elif action in ("Z", "D"):
            setattr(d, kw, "")
        elif action == "U":
            old = str(getattr(d, kw))
            uid_map.setdefault(old, generate_uid())
            setattr(d, kw, uid_map[old])
    d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
    d.remove_private_tags()
    d.PatientIdentityRemoved = "YES"
    if mask is not None:
        px = stored_pixels(d).copy()
        m = mask if px.ndim == 3 else np.broadcast_to(mask[..., None], px.shape)
        px[m] = 0
        if int(d.get("NumberOfFrames", 1) or 1) == 1:
            px = px[0]
        d.PixelData = px.tobytes()
        d.BurnedInAnnotation = "NO"
    return d


def crop(ds: Dataset, r0: int, c0: int, rows: int, cols: int) -> Dataset:
    d = copy.deepcopy(ds)
    px = stored_pixels(ds)[:, r0:r0 + rows, c0:c0 + cols]
    if int(ds.get("NumberOfFrames", 1) or 1) == 1:
        px = px[0]
    d.PixelData = np.ascontiguousarray(px).tobytes()
    d.Rows, d.Columns = rows, cols
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    dr, dc = (float(x) for x in ds.PixelSpacing)
    d.ImagePositionPatient = [round(float(v), 6) for v in ipp + c0 * dc * iop[:3] + r0 * dr * iop[3:]]
    d.SOPInstanceUID = generate_uid()
    d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
    d.SeriesInstanceUID = generate_uid()
    return d


def make_series(ds: Dataset, n: int, spacing_mm: float) -> list[Dataset]:
    """Stack copies of a real slice along its normal to form a geometrically valid series."""
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    normal = np.cross(iop[:3], iop[3:])
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    series_uid, out = generate_uid(), []
    for k in range(n):
        d = copy.deepcopy(ds)
        d.ImagePositionPatient = [round(float(v), 6) for v in ipp + k * spacing_mm * normal]
        d.SliceLocation = round(float(ipp @ normal + k * spacing_mm), 6)
        d.InstanceNumber = k + 1
        d.SOPInstanceUID = generate_uid()
        d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
        d.SeriesInstanceUID = series_uid
        out.append(d)
    return out


def perturb_ipp(series: list[Dataset], index: int, mm: float) -> list[Dataset]:
    out = [copy.deepcopy(d) for d in series]
    iop = np.array([float(x) for x in out[index].ImageOrientationPatient])
    normal = np.cross(iop[:3], iop[3:])
    ipp = np.array([float(x) for x in out[index].ImagePositionPatient])
    out[index].ImagePositionPatient = [round(float(v), 6) for v in ipp + mm * normal]
    return out
