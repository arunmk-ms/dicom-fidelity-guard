"""Provenance-preserving output and signed fidelity certificate (ENABLEMENT.md Section 14).

The reference build emits the certificate as JSON signed with HMAC-SHA256. A production build
encodes the same content as a DICOM Comprehensive SR with a Digital Signatures Sequence.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
from datetime import datetime, timezone

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from . import __version__
from .types import Report

PRIVATE_SCHEME = "99FGUARD"  # private coding scheme; map to CID 7203 codes after standards review
DERIVATION_CODES = {
    "DENOISE": ("FG-DENOISE", "AI denoising"),
    "CROP": ("FG-CROP", "Region-of-interest crop"),
    "SEGMENT": ("FG-SEGMENT", "AI segmentation"),
    "DISPLAY": ("FG-DISPLAY", "Display transform"),
}


def mark_derived(out: Dataset, src: Dataset, report: Report, model_name: str, model_version: str) -> Dataset:
    """Return a copy of out marked as DERIVED and linked to src. Source is never modified."""
    d = copy.deepcopy(out)
    if str(d.get("SOPInstanceUID")) == str(src.get("SOPInstanceUID")):
        d.SOPInstanceUID = generate_uid()
        d.file_meta.MediaStorageSOPInstanceUID = d.SOPInstanceUID
    if str(d.get("SeriesInstanceUID")) == str(src.get("SeriesInstanceUID")):
        d.SeriesInstanceUID = generate_uid()
    it = list(d.get("ImageType", ["DERIVED", "SECONDARY"]))
    it[0:2] = ["DERIVED", "SECONDARY"]
    d.ImageType = it
    ref = Dataset()
    ref.ReferencedSOPClassUID = src.SOPClassUID
    ref.ReferencedSOPInstanceUID = src.SOPInstanceUID
    d.SourceImageSequence = Sequence([ref])
    d.DerivationDescription = (f"{model_name} {model_version}; Fidelity Guard {__version__}; "
                               f"R={report.risk:.3f}; {report.decision.value}")[:1024]
    code = DERIVATION_CODES.get(report.operation.value)
    if code:
        c = Dataset()
        c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = code[0], PRIVATE_SCHEME, code[1]
        d.DerivationCodeSequence = Sequence([c])
    return d


def certificate(report: Report, model_name: str, model_version: str, key: bytes,
                policy_version: str = "reference-1") -> dict:
    body = {
        "type": "AI Image Fidelity Verification",
        "guard_version": __version__,
        "policy_version": policy_version,
        "model": {"name": model_name, "version": model_version},
        "issued_utc": datetime.now(timezone.utc).isoformat(),
        **report.to_dict(),
    }
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["signature"] = {"alg": "HMAC-SHA256",
                         "value": hmac.new(key, payload, hashlib.sha256).hexdigest()}
    return body


def verify_certificate(cert: dict, key: bytes) -> bool:
    body = {k: v for k, v in cert.items() if k != "signature"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hmac.compare_digest(cert["signature"]["value"], hmac.new(key, payload, hashlib.sha256).hexdigest())
