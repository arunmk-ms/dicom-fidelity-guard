# DICOM Fidelity Guard — Reference Implementation

This is a working reduction to practice of the DICOM Fidelity Guard (see [`../README.md`](../README.md) and [`../ENABLEMENT.md`](../ENABLEMENT.md)). It checks whether an AI-modified DICOM output is faithful to its source, then **releases** it, sends it to **review**, or **blocks** it, and issues a signed fidelity certificate.

## Implemented

| Layer | What it checks | Module |
|---|---|---|
| L1 | Pixels outside the allowed mask are bit-identical; changes inside the body are bounded relative to noise | `layers/scope.py` |
| L3 | The residual S−O contains no blob structure and no anatomical edges | `layers/residual.py` |
| L4 | Findings are preserved, using a training-free DoG detector with Hungarian matching | `layers/findings.py` |
| L5 | ROI means are preserved (5 HU on CT, 5% on other modalities) | `layers/quant.py` |
| L8 | De-identification: PS3.15 profile actions, full-header PHI-token scan, OCR (optional) | `layers/deid.py` |
| L9 | Identity tags, crop IPP, series geometry | `layers/geometry.py` |
| — | Noisy-OR risk with hard fails, routing, DERIVED marking, HMAC-signed certificate | `risk.py`, `certificate.py` |

Not implemented: L2 (raw-data physics), L6 (frequency), L7 (uncertainty), L10 (shadow), OCR (needs Tesseract).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pydicom numpy scipy scikit-image PyWavelets pillow pytest
# optional, for burned-in text OCR: pip install pytesseract, plus the Tesseract binary
```

## Use on your own DICOM files

```powershell
$env:FGUARD_KEY = "<secret>"
.\.venv\Scripts\python.exe -m fidelity_guard --op DENOISE `
    --source original.dcm --output ai_output.dcm --cert cert.json
```

Operations are `DENOISE`, `DEID` (pass `--mask m.npy` for the allowed redaction region), `CROP`, `SEGMENT` and `DISPLAY`. The exit code is 0 for RELEASE, 1 for REVIEW and 2 for BLOCK. In Python:

```python
from fidelity_guard import verify
report = verify("DENOISE", [src_ds], [out_ds])
print(report.decision, report.risk, [r.to_dict() for r in report.results])
```

Input requirements: uncompressed or decodable pixel data; for DENOISE the source and output must share one image grid.

## Tests and validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests                  # 11 tests on real pydicom DICOMs
.\.venv\Scripts\python.exe experiments\run_validation.py        # full run, about 5 min
.\.venv\Scripts\python.exe experiments\run_validation.py --quick
```

The validation script writes `results/validation_results.md` and `results/validation_trials.csv`, which contains per-trial layer scores.

**Latest full run:** 234 trials on 7 real DICOM files (CT, MR, multi-frame MR, US). Every clean output was released (34/34) and every injected defect was flagged (200/200). The smallest lesion caught had a contrast-to-noise ratio of 3.7. For details, failure cases found during development and limitations, see ENABLEMENT.md §12.5.

## Caveats

- The "AI model" is a non-local-means denoiser standing in for a commercial product.
- The lesions are synthetic and the corpus is small.
- The training-free L4 adds little beyond L3 because it is linear (ENABLEMENT §7.2 item 7). Use a trained detector in production.
- This is research code and not a medical device.
