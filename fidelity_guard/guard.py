"""Fidelity Guard orchestrator (ENABLEMENT.md Section 15)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydicom.dataset import Dataset

from .layers.deid import burned_in_text, header_deid
from .layers.findings import finding_preservation
from .layers.geometry import crop_check, identity, series_geometry
from .layers.quant import quantitative_invariants
from .layers.residual import residual_stats
from .layers.scope import bounded_change, exact_scope
from .preprocess import body_mask, estimate_noise, pixel_spacing_mm, stored_pixels, to_physical
from .risk import aggregate, route
from .types import LayerResult, Operation, Report


def _worst(results: list[LayerResult]) -> LayerResult:
    ran = [r for r in results if r.ran]
    if not ran:
        return results[0]
    w = max(ran, key=lambda r: (r.hard_fail, r.s))
    w.metrics = {**w.metrics, "instances_checked": len(ran)}
    return w


def verify(operation: Operation | str, sources: list[Dataset], outputs: list[Dataset],
           allowed_mask: Optional[list[np.ndarray]] = None, provenance: Optional[dict] = None,
           weights: Optional[dict] = None, thresholds: Optional[dict] = None) -> Report:
    """Verify AI-modified DICOM outputs against their sources and decide release/review/block.

    sources[i] is the source of outputs[i]. allowed_mask[i] is a bool array (frames, rows, cols)
    giving where outputs[i] may differ from sources[i] (DEID only).
    """
    op = Operation(operation)
    if len(sources) != len(outputs):
        raise ValueError("sources and outputs must be paired")
    results: list[LayerResult] = []
    required: set[str] = set()

    if op in (Operation.DENOISE, Operation.CROP, Operation.SEGMENT, Operation.DISPLAY):
        results.append(_worst([identity(s, o, provenance) for s, o in zip(sources, outputs)]))

    if op == Operation.DENOISE:
        S = np.concatenate([to_physical(s) for s in sources])
        outs = [to_physical(o) for o in outputs]
        O = np.concatenate(outs) if all(x.shape[1:] == S.shape[1:] for x in outs) else None
        if O is None or O.shape != S.shape:
            results.append(LayerResult("L1_scope_bounded", "fail", 1.0, True, {}, "image grid changed"))
        else:
            modality = str(sources[0].get("Modality", "OT"))
            spacing = float(np.mean(pixel_spacing_mm(sources[0])))
            mask = body_mask(S, modality)
            sigma = estimate_noise(S, mask)
            results.append(bounded_change(S, O, sigma, mask))
            results.append(residual_stats(S, O, mask, spacing, sigma))
            results.append(finding_preservation(S, O, mask, spacing, sigma))
            q = quantitative_invariants(S, O, mask, spacing, modality, sigma)
            results.append(q)
            results[-1].metrics["noise_sigma"] = sigma
            required |= {"L3_residual", "L4_findings", "L5_quant"}
        if len(outputs) > 1:
            results.append(series_geometry(outputs, sources))

    elif op == Operation.DEID:
        results.append(_worst([header_deid(s, o) for s, o in zip(sources, outputs)]))
        scopes = []
        for i, (s, o) in enumerate(zip(sources, outputs)):
            m = allowed_mask[i] if allowed_mask is not None else None
            scopes.append(exact_scope(stored_pixels(s), stored_pixels(o), m))
        results.append(_worst(scopes))
        ocr = _worst([burned_in_text(s, stored_pixels(o), o) for s, o in zip(sources, outputs)])
        results.append(ocr)
        if any(str(s.get("BurnedInAnnotation", "")).upper() == "YES" or
               str(s.get("Modality", "")) in ("US", "SC", "OT") for s in sources):
            required.add("L8_deid_ocr")

    elif op == Operation.CROP:
        results.append(_worst([crop_check(s, o) for s, o in zip(sources, outputs)]))

    elif op in (Operation.SEGMENT, Operation.DISPLAY):
        results.append(_worst([exact_scope(stored_pixels(s), stored_pixels(o), None)
                               for s, o in zip(sources, outputs)]))

    risk = aggregate(results, weights, required)
    return Report(op, route(risk, thresholds), risk, results,
                  [str(s.get("SOPInstanceUID")) for s in sources],
                  [str(o.get("SOPInstanceUID")) for o in outputs])
