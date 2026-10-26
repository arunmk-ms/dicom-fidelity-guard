"""Command-line interface.

    python -m fidelity_guard --op DENOISE --source in1.dcm in2.dcm --output out1.dcm out2.dcm \
        [--mask mask.npy] [--cert cert.json] [--key-env FGUARD_KEY]

Exit code: 0 RELEASE, 1 REVIEW, 2 BLOCK.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pydicom

from . import Decision, verify
from .certificate import certificate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fidelity_guard", description="Verify an AI-modified DICOM output.")
    ap.add_argument("--op", required=True, choices=["DENOISE", "DEID", "CROP", "SEGMENT", "DISPLAY"])
    ap.add_argument("--source", nargs="+", required=True, help="source DICOM files (same order as --output)")
    ap.add_argument("--output", nargs="+", required=True, help="AI-modified DICOM files")
    ap.add_argument("--mask", nargs="*", help="DEID: .npy boolean masks (frames, rows, cols) of allowed pixel edits")
    ap.add_argument("--cert", help="write a signed fidelity certificate (JSON) to this path")
    ap.add_argument("--key-env", default="FGUARD_KEY", help="environment variable holding the HMAC key")
    ap.add_argument("--model", default="unknown", help="model name for the certificate")
    ap.add_argument("--model-version", default="unknown")
    a = ap.parse_args(argv)
    if len(a.source) != len(a.output):
        ap.error("--source and --output must have the same number of files")

    sources = [pydicom.dcmread(p) for p in a.source]
    outputs = [pydicom.dcmread(p) for p in a.output]
    masks = [np.load(p).astype(bool) for p in a.mask] if a.mask else None
    rep = verify(a.op, sources, outputs, allowed_mask=masks)

    print(f"Decision: {rep.decision.value}   risk={rep.risk:.3f}")
    for r in rep.results:
        flag = " HARD" if r.hard_fail else ""
        print(f"  {r.layer:<22} {r.status:<8} s={r.s:.3f}{flag}  {r.reason}")

    if a.cert:
        key = os.environ.get(a.key_env)
        if not key:
            print(f"warning: ${a.key_env} not set; certificate not written", file=sys.stderr)
        else:
            with open(a.cert, "w", encoding="utf-8") as f:
                json.dump(certificate(rep, a.model, a.model_version, key.encode()), f, indent=2)
            print(f"Certificate: {a.cert}")
    return {Decision.RELEASE: 0, Decision.REVIEW: 1, Decision.BLOCK: 2}[rep.decision]


if __name__ == "__main__":
    sys.exit(main())
