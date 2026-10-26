"""End-to-end tests of the Fidelity Guard on real DICOM files shipped with pydicom."""
import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file

from fidelity_guard import Decision, verify
from fidelity_guard import simulate as sim
from fidelity_guard.certificate import certificate, mark_derived, verify_certificate
from fidelity_guard.preprocess import body_mask, estimate_noise, to_physical

KEY = b"test-key"


@pytest.fixture(scope="module")
def ct():
    ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    phys = to_physical(ds)
    sigma = estimate_noise(phys, body_mask(phys, "CT"))
    clean = sim.with_pixels(ds, sim.denoise(phys, sigma, 0.8))
    return ds, phys, sigma, clean


def _phi_ds(name="MR_small.dcm"):
    ds = pydicom.dcmread(get_testdata_file(name))
    ds.PatientID = ds.get("PatientID") or "MRN12345"
    ds.PatientName = ds.get("PatientName") or "Doe^Jane"
    return ds


def test_clean_denoise_released(ct):
    ds, _, _, clean = ct
    assert verify("DENOISE", [ds], [clean]).decision == Decision.RELEASE


@pytest.mark.parametrize("contrast", [50.0, 100.0])
def test_added_lesion_blocked(ct, contrast):
    ds, phys, sigma, _ = ct
    out = sim.add_lesion(sim.denoise(phys, sigma, 0.8), 0, 64, 64, 6, contrast)
    rep = verify("DENOISE", [ds], [sim.with_pixels(ds, out)])
    assert rep.decision != Decision.RELEASE


def test_removed_lesion_blocked(ct):
    ds, phys, sigma, _ = ct
    src = sim.with_pixels(ds, sim.add_lesion(phys, 0, 64, 64, 6, 100.0), new_uids=False)
    rep = verify("DENOISE", [src], [sim.with_pixels(ds, sim.denoise(phys, sigma, 0.8))])
    assert rep.decision != Decision.RELEASE


def test_hu_shift_flagged(ct):
    ds, phys, sigma, _ = ct
    out = sim.shift_values(sim.denoise(phys, sigma, 0.8), 15.0)
    rep = verify("DENOISE", [ds], [sim.with_pixels(ds, out)])
    assert rep.decision != Decision.RELEASE
    assert rep.layer("L5_quant").status == "fail"


def test_deid_clean_and_leaks():
    ds = _phi_ds()
    m = np.zeros((1, ds.Rows, ds.Columns), bool)
    m[0, :6, :20] = True
    assert verify("DEID", [ds], [sim.deidentify(ds, m)], allowed_mask=[m]).decision == Decision.RELEASE

    leak = sim.deidentify(ds, m)
    leak.PatientName = ds.PatientName
    assert verify("DEID", [ds], [leak], allowed_mask=[m]).decision == Decision.BLOCK

    hidden = sim.deidentify(ds, m)
    hidden.ImageComments = f"orig id {ds.PatientID}"
    assert verify("DEID", [ds], [hidden], allowed_mask=[m]).decision == Decision.BLOCK


def test_deid_out_of_scope_pixel_edit():
    ds = _phi_ds()
    m = np.zeros((1, ds.Rows, ds.Columns), bool)
    m[0, :6, :20] = True
    bad = sim.deidentify(ds, m)
    px = bad.pixel_array.copy()
    px[40, 40] += 1
    bad.PixelData = px.tobytes()
    assert verify("DEID", [ds], [bad], allowed_mask=[m]).decision == Decision.BLOCK


def test_crop_geometry():
    ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    good = sim.crop(ds, 10, 20, 64, 64)
    assert verify("CROP", [ds], [good]).decision == Decision.RELEASE
    bad = sim.crop(ds, 10, 20, 64, 64)
    bad.ImagePositionPatient = [float(x) + 1.0 for x in bad.ImagePositionPatient]
    assert verify("CROP", [ds], [bad]).decision == Decision.BLOCK


@pytest.mark.parametrize("mm,ok", [(0.2, True), (2.0, False)])
def test_series_geometry(ct, mm, ok):
    ds, phys, sigma, _ = ct
    src = sim.make_series(ds, 4, 5.0)
    out = [sim.with_pixels(s, sim.denoise(phys, sigma, 0.8)) for s in src]
    out = sim.perturb_ipp(out, 2, mm)
    rep = verify("DENOISE", src, out)
    assert (rep.decision == Decision.RELEASE) == ok


def test_certificate_roundtrip(ct):
    ds, _, _, clean = ct
    rep = verify("DENOISE", [ds], [clean])
    derived = mark_derived(clean, ds, rep, "nlmeans-standin", "1.0")
    assert "DERIVED" in derived.ImageType
    cert = certificate(rep, "nlmeans-standin", "1.0", KEY)
    assert verify_certificate(cert, KEY)
    cert["decision"] = "RELEASE" if cert["decision"] != "RELEASE" else "BLOCK"
    assert not verify_certificate(cert, KEY)
