"""Validation experiment on real DICOM images (ENABLEMENT.md Section 12).

Runs the Fidelity Guard on correct ("clean") AI outputs and on outputs with injected
hallucinations, then writes results/validation_results.md and results/validation_trials.csv.

Usage:  python experiments/run_validation.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from pydicom.data import get_testdata_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fidelity_guard import __version__, verify  # noqa: E402
from fidelity_guard import simulate as sim  # noqa: E402
from fidelity_guard.layers.residual import interior  # noqa: E402
from fidelity_guard.preprocess import body_mask, estimate_noise, pixel_spacing_mm, to_physical  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DENOISE_CORPUS = ["CT_small.dcm", "693_UNCR.dcm", "MR_small.dcm", "emri_small.dcm",
                  "MR-SIEMENS-DICOM-WithOverlays.dcm", "MR2_UNCR.dcm"]
CLEAN_STRENGTHS = [0.5, 0.7, 0.9, 1.1]
RADII_MM = [2.5, 4.0, 7.0]


def contrasts(modality: str, S: np.ndarray, mask: np.ndarray) -> list[tuple[str, float]]:
    if modality == "CT":
        return [("20HU", 20.0), ("50HU", 50.0), ("100HU", 100.0)]
    ref = float(np.median(S[mask]))
    return [("5%", 0.05 * ref), ("10%", 0.10 * ref), ("20%", 0.20 * ref)]


def shifts(modality: str, S: np.ndarray, mask: np.ndarray) -> list[tuple[str, float, bool]]:
    if modality == "CT":
        return [("global +10HU", 10.0, False), ("global +25HU", 25.0, False),
                ("regional +15HU", 15.0, True), ("regional +40HU", 40.0, True)]
    ref = float(np.median(S[mask]))
    return [("global +8%", 0.08 * ref, False), ("global +15%", 0.15 * ref, False),
            ("regional +10%", 0.10 * ref, True), ("regional +25%", 0.25 * ref, True)]


def random_site(rng, inner: np.ndarray, margin_px: int):
    cand = np.argwhere(inner)
    cand = cand[(cand[:, 1] > margin_px) & (cand[:, 2] > margin_px) &
                (cand[:, 1] < inner.shape[1] - margin_px) & (cand[:, 2] < inner.shape[2] - margin_px)]
    if len(cand) == 0:
        return None
    return tuple(int(x) for x in cand[rng.integers(len(cand))])


def run(quick: bool):
    rng = np.random.default_rng(20261022)
    rows = []
    positions = 1 if quick else 2
    corpus = DENOISE_CORPUS[:3] if quick else DENOISE_CORPUS

    def record(image, op, category, defect, param, expected_flag, rep, extra=None):
        flagged = rep.decision.value != "RELEASE"
        row = {"image": image, "operation": op, "category": category, "defect": defect, "param": param,
               "expected": "flag" if expected_flag else "release", "decision": rep.decision.value,
               "risk": round(rep.risk, 4), "correct": flagged == expected_flag}
        for r in rep.results:
            row[f"s_{r.layer}"] = round(r.s, 3) if r.ran else "not_run"
        row.update(extra or {})
        rows.append(row)
        return row

    t0 = time.time()
    for fn in corpus:
        ds = pydicom.dcmread(get_testdata_file(fn))
        S = to_physical(ds)
        mod = str(ds.Modality)
        mask = body_mask(S, mod)
        sigma = estimate_noise(S, mask)
        sp = float(np.mean(pixel_spacing_mm(ds)))
        inner = interior(mask, sp, 12.0)
        print(f"[{time.time() - t0:6.1f}s] {fn}: {S.shape}, sigma={sigma:.2f}, spacing={sp:.2f} mm", flush=True)

        clean = {}
        for st in CLEAN_STRENGTHS:
            clean[st] = sim.denoise(S, sigma, st)
            rep = verify("DENOISE", [ds], [sim.with_pixels(ds, clean[st])])
            record(fn, "DENOISE", "clean", "none", f"nlm h={st}sigma", False, rep)
        O = clean[0.7]

        for cname, cval in contrasts(mod, S, mask):
            for rmm in RADII_MM:
                for _ in range(positions):
                    site = random_site(rng, inner, int(rmm / sp) + 2)
                    if site is None:
                        continue
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    k, r, c = site
                    cnr = abs(cval) / sigma
                    o_add = sim.add_lesion(O, k, r, c, rmm / sp, sign * cval)
                    rep = verify("DENOISE", [ds], [sim.with_pixels(ds, o_add)])
                    record(fn, "DENOISE", "added lesion", f"{cname} r={rmm}mm", f"CNR={cnr:.1f}", True, rep)
                    s_les = sim.add_lesion(S, k, r, c, rmm / sp, sign * cval)
                    src = sim.with_pixels(ds, s_les, new_uids=False)
                    rep = verify("DENOISE", [src], [sim.with_pixels(ds, O)])
                    record(fn, "DENOISE", "removed lesion", f"{cname} r={rmm}mm", f"CNR={cnr:.1f}", True, rep)

        for name, delta, regional in shifts(mod, S, mask):
            if regional:
                site = random_site(rng, inner, int(15 / sp) + 2)
                o_sh = sim.shift_values(O, delta, site, 15 / sp)
            else:
                o_sh = sim.shift_values(O, delta)
            rep = verify("DENOISE", [ds], [sim.with_pixels(ds, o_sh)])
            record(fn, "DENOISE", "quantitative drift", name, "", True, rep)

    # Geometry: real CT slice stacked into a 6-slice series; clean output vs perturbed IPP.
    ct = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    series = sim.make_series(ct, 6, 2.5)
    Sct = to_physical(ct)
    sig_ct = estimate_noise(Sct, body_mask(Sct, "CT"))
    outs = [sim.with_pixels(d, sim.denoise(to_physical(d), sig_ct, 0.7)) for d in series]
    for d, o in zip(series, outs):
        o.SeriesInstanceUID = outs[0].SeriesInstanceUID
    rep = verify("DENOISE", series, outs)
    record("CT_small.dcm x6", "DENOISE", "clean", "series geometry intact", "", False, rep)
    for mm in [0.2, 0.6, 2.0, 12.0]:
        rep = verify("DENOISE", series, sim.perturb_ipp(outs, 3, mm))
        record("CT_small.dcm x6", "DENOISE", "geometry error", "slice 3 IPP shifted", f"{mm} mm",
               mm > 0.5, rep)

    # De-identification on real CT and MR headers (pixel redaction mask on a corner box).
    for fn in ["CT_small.dcm", "MR_small.dcm", "693_UNCR.dcm", "MR2_UNCR.dcm"]:
        ds = pydicom.dcmread(get_testdata_file(fn))
        if not str(ds.get("PatientID", "")):
            ds.PatientID = "MRN12345"
        if not str(ds.get("PatientName", "")):
            ds.PatientName = "Doe^Jane"
        rows_, cols_ = int(ds.Rows), int(ds.Columns)
        m = np.zeros((1, rows_, cols_), dtype=bool)
        m[0, : rows_ // 10, : cols_ // 3] = True
        good = sim.deidentify(ds, m)
        rep = verify("DEID", [ds], [good], allowed_mask=[m])
        record(fn, "DEID", "clean", "correct de-identification", "", False, rep)
        bad = sim.deidentify(ds, m)
        bad.PatientName = ds.PatientName
        rep = verify("DEID", [ds], [bad], allowed_mask=[m])
        record(fn, "DEID", "PHI leak", "PatientName retained", "", True, rep)
        bad = sim.deidentify(ds, m)
        bad.add_new(0x00091010, "LO", f"orig id {ds.PatientID}")
        rep = verify("DEID", [ds], [bad], allowed_mask=[m])
        record(fn, "DEID", "PHI leak", "PatientID copied into private tag", "", True, rep)
        bad = sim.deidentify(ds, m)
        px = bad.pixel_array.copy()
        px[rows_ // 2, cols_ // 2] += 1
        bad.PixelData = px.tobytes()
        rep = verify("DEID", [ds], [bad], allowed_mask=[m])
        record(fn, "DEID", "out-of-scope pixel edit", "1 pixel changed outside mask", "", True, rep)
        bad = sim.deidentify(ds, m)
        bad.PatientIdentityRemoved = "NO"
        rep = verify("DEID", [ds], [bad], allowed_mask=[m])
        record(fn, "DEID", "PHI leak", "PatientIdentityRemoved not set", "", True, rep)

    us = pydicom.dcmread(get_testdata_file("US1_UNCR.dcm"))
    m = np.zeros((1, int(us.Rows), int(us.Columns)), dtype=bool)
    m[0, :60, :] = True
    rep = verify("DEID", [us], [sim.deidentify(us, m)], allowed_mask=[m])
    record("US1_UNCR.dcm", "DEID", "reduced assurance", "US de-id while OCR layer unavailable", "",
           True, rep, {"note": "expected REVIEW: OCR layer is required for US but not installed"})

    # Crop on real CT and MR.
    for fn in ["CT_small.dcm", "693_UNCR.dcm", "MR2_UNCR.dcm", "MR_small.dcm"]:
        ds = pydicom.dcmread(get_testdata_file(fn))
        R_, C_ = int(ds.Rows), int(ds.Columns)
        r0, c0, h, w = R_ // 4, C_ // 5, R_ // 2, C_ // 2
        good = sim.crop(ds, r0, c0, h, w)
        record(fn, "CROP", "clean", "correct crop", "", False, verify("CROP", [ds], [good]))
        bad = sim.crop(ds, r0, c0, h, w)
        ipp = [float(x) for x in bad.ImagePositionPatient]
        iop = [float(x) for x in bad.ImageOrientationPatient]
        dc = float(ds.PixelSpacing[1])
        bad.ImagePositionPatient = [ipp[i] + dc * iop[i] for i in range(3)]
        record(fn, "CROP", "geometry error", "IPP off by one column", "", True, verify("CROP", [ds], [bad]))
        bad = sim.crop(ds, r0, c0, h, w)
        px = bad.pixel_array.copy()
        px[h // 2, w // 2] = px[h // 2, w // 2] + 50
        bad.PixelData = px.tobytes()
        record(fn, "CROP", "out-of-scope pixel edit", "pixel altered inside crop", "", True,
               verify("CROP", [ds], [bad]))
        bad = sim.crop(ds, r0, c0, h, w)
        bad.PatientID = "SOMEONE-ELSE"
        record(fn, "CROP", "identity error", "PatientID changed without provenance", "", True,
               verify("CROP", [ds], [bad]))

    return rows, time.time() - t0


def summarize(rows, elapsed, quick):
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    lead = ["image", "operation", "category", "defect", "param", "expected", "decision", "risk", "correct"]
    keys = lead + [k for k in keys if k not in lead]
    with open(out_dir / "validation_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def rate(sel):
        n = len(sel)
        k = sum(r["correct"] for r in sel)
        return n, k, (k / n if n else float("nan"))

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[(r["operation"], r["category"], r["expected"])].append(r)
    lines = [
        "# Fidelity Guard - validation on real DICOM images",
        "",
        f"- Run (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        f"- Guard version: {__version__}; Python {platform.python_version()}; pydicom {pydicom.__version__}; "
        f"numpy {np.__version__}",
        f"- Mode: {'quick' if quick else 'full'}; trials: {len(rows)}; runtime: {elapsed:.0f} s",
        "- Images: real clinical DICOM files distributed with pydicom (CT, MR, multi-frame MR, US).",
        "- Stand-in AI model: non-local-means denoiser (scikit-image). Defects injected per ENABLEMENT.md 12.1.",
        "- A trial is *correct* when a clean output is RELEASED or a defective output is flagged (REVIEW or BLOCK).",
        "",
        "## Summary by category",
        "",
        "| Operation | Category | Expected | Trials | Correct | Rate |",
        "|-----------|----------|----------|-------:|--------:|-----:|",
    ]
    for (op, cat, exp), sel in sorted(by_cat.items()):
        n, k, p = rate(sel)
        lines.append(f"| {op} | {cat} | {exp} | {n} | {k} | {p:.1%} |")

    clean = [r for r in rows if r["expected"] == "release"]
    defect = [r for r in rows if r["expected"] == "flag"]
    n_c, k_c, p_c = rate(clean)
    n_d, k_d, p_d = rate(defect)
    lines += ["", f"**Overall specificity (clean released):** {k_c}/{n_c} = {p_c:.1%}  ",
              f"**Overall sensitivity (defects flagged):** {k_d}/{n_d} = {p_d:.1%}", ""]

    lines += ["## Lesion detection by contrast and size (DENOISE)", "",
              "| Image | Defect | Contrast / radius | Trials | Flagged |", "|---|---|---|---:|---:|"]
    grp = defaultdict(list)
    for r in rows:
        if r["category"] in ("added lesion", "removed lesion"):
            grp[(r["image"], r["category"], r["defect"])].append(r)
    for (img, cat, d), sel in sorted(grp.items()):
        n, k, _ = rate(sel)
        lines.append(f"| {img} | {cat} | {d} ({sel[0]['param']}) | {n} | {k} |")

    misses = [r for r in rows if not r["correct"]]
    lines += ["", "## Incorrect decisions", ""]
    if not misses:
        lines.append("None.")
    else:
        lines += ["| Image | Category | Defect | Param | Decision | Risk |", "|---|---|---|---|---|---:|"]
        for r in misses:
            lines.append(f"| {r['image']} | {r['category']} | {r['defect']} | {r['param']} | {r['decision']} | {r['risk']} |")
    lines += ["", "Per-trial layer scores are in `validation_trials.csv`.", ""]
    (out_dir / "validation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rows, elapsed = run(a.quick)
    summarize(rows, elapsed, a.quick)
