"""L8 - verifiable de-identification (ENABLEMENT.md Section 8).

Implements a representative subset of the DICOM PS3.15 Annex E Basic Application Level
Confidentiality Profile, a PHI-leak scan of every string element in the output (including
sequences and private tags), the PatientIdentityRemoved check, and optional OCR re-detection
of burned-in text when Tesseract is installed.
"""
from __future__ import annotations

import re
import shutil

import numpy as np
from pydicom.dataset import Dataset

from ..types import LayerResult

# keyword -> PS3.15 action code (subset). X=remove, Z=zero/dummy, D=dummy, U=replace UID.
PROFILE: dict[str, str] = {
    "PatientName": "Z", "PatientID": "Z", "PatientBirthDate": "Z", "PatientSex": "Z",
    "OtherPatientIDs": "X", "OtherPatientNames": "X", "PatientAddress": "X",
    "PatientTelephoneNumbers": "X", "PatientBirthName": "X", "PatientMotherBirthName": "X",
    "MilitaryRank": "X", "EthnicGroup": "X", "PatientReligiousPreference": "X",
    "AccessionNumber": "Z", "StudyID": "Z", "ReferringPhysicianName": "Z",
    "InstitutionName": "X", "InstitutionAddress": "X", "InstitutionalDepartmentName": "X",
    "StationName": "X", "OperatorsName": "X", "PerformingPhysicianName": "X",
    "NameOfPhysiciansReadingStudy": "X", "PhysiciansOfRecord": "X",
    "RequestingPhysician": "X", "DeviceSerialNumber": "X",
    "StudyInstanceUID": "U", "SeriesInstanceUID": "U", "SOPInstanceUID": "U",
    "FrameOfReferenceUID": "U",
}
PHI_KEYWORDS = [k for k, a in PROFILE.items() if a in ("Z", "X", "D") and k not in ("PatientSex", "StudyID")]
UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
SAFE_RETAINED = ("Manufacturer", "ManufacturerModelName", "SoftwareVersions", "Modality", "BodyPartExamined")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _phi_tokens(src: Dataset) -> set[str]:
    toks: set[str] = set()
    for kw in PHI_KEYWORDS:
        v = src.get(kw)
        if v is None or v == "":
            continue
        vals = v if isinstance(v, (list, tuple)) or getattr(v, "VM", 1) > 1 else [v]
        for item in vals:
            text = str(item)
            for part in re.split(r"[\^ ,]+", text) + [text]:
                t = _norm(part)
                if len(t) >= 3 and len(set(t)) > 1:  # skip placeholders such as 000000
                    toks.add(t)
    # Tokens that also occur in attributes the profile retains (e.g., Manufacturer "TOSHIBA_MEC"
    # vs InstitutionName "TOSHIBA") are not patient-identifying and would cause false leaks.
    retained = set()
    for kw in SAFE_RETAINED:
        if kw in src:
            retained |= _words(str(src.get(kw)))
    return toks - retained


def _words(text: str) -> set[str]:
    words = {w for w in re.split(r"[^a-z0-9]+", text.lower()) if w}
    words.add(_norm(text))
    return words


def _iter_strings(ds: Dataset, path: str = ""):
    for elem in ds:
        if elem.VR == "SQ":
            for i, item in enumerate(elem.value):
                yield from _iter_strings(item, f"{path}{elem.keyword or elem.tag}[{i}].")
        elif elem.VR in ("PN", "LO", "SH", "ST", "LT", "UT", "DA", "DT", "TM", "AE", "CS", "UC", "IS", "DS") or elem.tag.is_private:
            if elem.tag == 0x7FE00010:
                continue
            yield f"{path}{elem.keyword or str(elem.tag)}", str(elem.value)


def header_deid(src: Dataset, out: Dataset) -> LayerResult:
    name = "L8_deid_header"
    violations = []
    for kw, action in PROFILE.items():
        sv, ov = src.get(kw), out.get(kw)
        if action == "X" and ov is not None and str(ov) != "":
            violations.append(f"{kw}: must be removed")
        elif action in ("Z", "D") and ov not in (None, "") and sv not in (None, "") and str(ov) == str(sv):
            violations.append(f"{kw}: original value retained")
        elif action == "U" and sv is not None:
            if ov is None or str(ov) == str(sv) or not UID_RE.match(str(ov)):
                violations.append(f"{kw}: UID not replaced")
    if str(out.get("PatientIdentityRemoved", "")).upper() != "YES":
        violations.append("PatientIdentityRemoved (0012,0062) is not YES")

    tokens = _phi_tokens(src)
    leaks = []
    for path, text in _iter_strings(out):
        words, n = _words(text), _norm(text)
        for t in tokens:
            # whole-word match; substring match only for long alphanumeric tokens (embedded IDs)
            if t in words or (len(t) >= 6 and not t.isdigit() and t in n):
                leaks.append(f"{path} contains source PHI token")
                break
    bad = violations + leaks
    return LayerResult(name, "fail" if bad else "pass", 1.0 if bad else 0.0, bool(bad),
                       {"profile_violations": violations, "phi_leaks": leaks[:20],
                        "n_phi_tokens_checked": len(tokens)},
                       "; ".join(bad[:5]))


def burned_in_text(src: Dataset, out_pixels: np.ndarray, out: Dataset) -> LayerResult:
    """OCR re-detection of burned-in PHI. Returns not_run when Tesseract is unavailable."""
    name = "L8_deid_ocr"
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return LayerResult.not_run(name, "pytesseract not installed")
    if shutil.which("tesseract") is None:
        return LayerResult.not_run(name, "tesseract binary not found")
    import pytesseract
    from PIL import Image

    tokens = _phi_tokens(src)
    hits = []
    for k in range(out_pixels.shape[0]):
        f = out_pixels[k].astype(np.float64)
        if f.ndim == 3:
            f = f.mean(-1)
        lo, hi = np.percentile(f, [1, 99])
        views = [f, np.clip((f - lo) / max(hi - lo, 1e-9), 0, 1) * 255]
        views.append(255 - views[1])
        for v in views:
            v = (v - v.min()) / max(np.ptp(v), 1e-9) * 255
            img = Image.fromarray(v.astype(np.uint8)).resize((v.shape[1] * 2, v.shape[0] * 2))
            text = pytesseract.image_to_string(img, config="--psm 11")
            n = _norm(text)
            hits += [t for t in tokens if t in n]
    hits = sorted(set(hits))
    return LayerResult(name, "fail" if hits else "pass", 1.0 if hits else 0.0, bool(hits),
                       {"phi_tokens_found_in_pixels": len(hits)})
