"""L9 - metadata integrity: identity tiers, provenance and geometry (ENABLEMENT.md Section 9)."""
from __future__ import annotations

import numpy as np
from pydicom.dataset import Dataset

from ..preprocess import pixel_spacing_mm, stored_pixels
from ..types import LayerResult

TIER0 = ["PatientName", "PatientID", "PatientBirthDate", "AccessionNumber",
         "StudyInstanceUID", "FrameOfReferenceUID"]


def identity(src: Dataset, out: Dataset, provenance: dict | None = None) -> LayerResult:
    """Tier-0 attributes must match the source unless a cited source of truth gives the new value.

    provenance: {keyword: {"value": str, "source": str, "resolved_value": str}} where
    resolved_value is what the connector (FHIR/HL7/MWL/MPI) returned.
    """
    name = "L9_identity"
    provenance = provenance or {}
    problems = []
    for kw in TIER0:
        sv, ov = src.get(kw), out.get(kw)
        if sv is None and ov is None:
            continue
        if str(sv) == str(ov):
            continue
        p = provenance.get(kw)
        if p is None:
            problems.append(f"{kw} changed without provenance")
        elif str(p.get("resolved_value")) != str(ov):
            problems.append(f"{kw} does not match cited source {p.get('source')}")
    if str(out.get("SOPInstanceUID")) == str(src.get("SOPInstanceUID")):
        problems.append("derived object reuses the source SOPInstanceUID (would overwrite the original)")
    return LayerResult(name, "fail" if problems else "pass", 1.0 if problems else 0.0, bool(problems),
                       {"problems": problems}, "; ".join(problems))


def _geom(ds: Dataset):
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    return ipp, iop[:3], iop[3:]


def crop_check(src: Dataset, out: Dataset, tol_mm: float = 0.01) -> LayerResult:
    """Output must be S[r0:r0+Rows, c0:c0+Cols] with (r0, c0) derived from the written IPP.

    P(c, r) = IPP + c * dc * X + r * dr * Y   (PS3.3 C.7.6.2.1.1), dr, dc = PixelSpacing.
    """
    name = "L9_crop_geometry"
    problems = []
    ipp_s, xs, ys = _geom(src)
    ipp_o, xo, yo = _geom(out)
    dr, dc = pixel_spacing_mm(src)
    if not (np.allclose(xs, xo, atol=1e-4) and np.allclose(ys, yo, atol=1e-4)):
        problems.append("ImageOrientationPatient changed")
    if pixel_spacing_mm(out) != (dr, dc):
        problems.append("PixelSpacing changed")
    off = ipp_o - ipp_s
    c0f, r0f = off @ xs / dc, off @ ys / dr
    c0, r0 = int(round(c0f)), int(round(r0f))
    expected_ipp = ipp_s + c0 * dc * xs + r0 * dr * ys
    ipp_err = float(np.linalg.norm(ipp_o - expected_ipp))
    if ipp_err > tol_mm:
        problems.append(f"IPP is not on the source pixel grid (error {ipp_err:.3f} mm)")
    S, O = stored_pixels(src), stored_pixels(out)
    rows, cols = O.shape[1], O.shape[2]
    if r0 < 0 or c0 < 0 or r0 + rows > S.shape[1] or c0 + cols > S.shape[2]:
        problems.append("crop window outside the source image")
        identical = False
    else:
        identical = bool(np.array_equal(S[:, r0:r0 + rows, c0:c0 + cols], O))
        if not identical:
            problems.append("cropped pixels differ from the source at the location implied by IPP")
    return LayerResult(name, "fail" if problems else "pass", 1.0 if problems else 0.0, bool(problems),
                       {"r0": r0, "c0": c0, "ipp_error_mm": ipp_err, "pixels_identical": identical,
                        "problems": problems}, "; ".join(problems))


def series_geometry(outputs: list[Dataset], sources: list[Dataset] | None = None,
                    tol_mm: float = 0.5) -> LayerResult:
    """Slice-stack consistency; when sources are given, positions must match the source geometry."""
    name = "L9_series_geometry"
    problems = []
    if len(outputs) < 2 or any("ImagePositionPatient" not in d for d in outputs):
        return LayerResult.not_run(name, "single image or no IPP")
    geo = [_geom(d) for d in outputs]
    x0, y0 = geo[0][1], geo[0][2]
    n = np.cross(x0, y0)
    if any(not (np.allclose(g[1], x0, atol=1e-4) and np.allclose(g[2], y0, atol=1e-4)) for g in geo):
        problems.append("ImageOrientationPatient differs between slices")
    if len({str(d.get("FrameOfReferenceUID")) for d in outputs}) > 1:
        problems.append("multiple FrameOfReferenceUIDs in one series")
    d = np.array([g[0] @ n for g in geo])
    order = np.argsort(d)
    ds = np.sort(d)
    steps = np.diff(ds)
    if np.any(steps <= 1e-6):
        problems.append("duplicate slice positions")
    cv = float(steps.std() / steps.mean()) if steps.mean() > 0 else float("inf")
    step_dev = float(np.abs(steps - np.median(steps)).max()) if len(steps) else 0.0
    if step_dev > tol_mm:
        problems.append(f"non-uniform slice spacing (step deviates {step_dev:.2f} mm from median)")
    max_dev = 0.0
    worst = None
    if sources is not None and len(sources) == len(outputs):
        sgeo = [_geom(s) for s in sources]
        s_ipp = [sgeo[i][0] for i in np.argsort([g[0] @ n for g in sgeo])]
        o_ipp = [geo[i][0] for i in order]
        devs = [float(np.linalg.norm(a - b)) for a, b in zip(o_ipp, s_ipp)]
        max_dev = max(devs)
        worst = int(np.argmax(devs))
        if max_dev > tol_mm:
            problems.append(f"slice {worst} deviates {max_dev:.2f} mm from the source geometry")
    return LayerResult(name, "fail" if problems else "pass", 1.0 if problems else 0.0, bool(problems),
                       {"spacing_cv": cv, "max_step_deviation_mm": step_dev, "max_deviation_mm": max_dev, "worst_slice": worst,
                        "problems": problems}, "; ".join(problems))
