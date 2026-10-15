"""The synthetic phantoms give stable, known values."""
import pytest

from microct3d.validation.phantoms import run_axis_phantom, run_criterion_phantom


def test_canal_criterion_and_calibre():
    rows = {r["structure"]: r for r in run_criterion_phantom()}
    for name, cal in [("canal 20 um", 22.4), ("canal 30 um", 31.6),
                      ("canal 50 um", 51.0), ("canal 100 um", 100.5)]:
        assert rows[name]["calibre_um"] == pytest.approx(cal, abs=0.05)
        assert rows[name]["kept_pct"] > 80
    for name in ("thin cylinder", "flat slit", "lacunae 12 um"):
        assert rows[name]["kept_pct"] == 0


def test_axis_length_and_area_equivalent_diameter():
    rows = {r["phantom"]: r for r in run_axis_phantom()}
    for d in (30, 50, 80):
        r = rows[f"45deg {d} um"]
        assert r["Deq_um"] == pytest.approx(d, rel=0.04)
        assert r["length_voxel_count_um"] < 0.75 * r["length_mst_um"]
    e = rows["elliptic 36 um"]
    assert e["Deq_um"] == pytest.approx(e["true_Deq_um"], rel=0.02)
    assert e["calibre_um"] < e["Deq_um"]
