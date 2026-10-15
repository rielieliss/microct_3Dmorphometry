"""Synthetic volumes with known geometry.

In real samples there is no ground truth: the true diameter of a canal is unknown.
The same code is therefore run on volumes where the answer is known by construction.

* :func:`criterion_phantom` - a bone plate with cylinders of 20, 30, 50 and 100 µm and,
  on purpose, three structures that are not canals (a thin cylinder, a flat slit and
  lacuna-sized spheres). Checks the calibre and the 20 µm canal criterion.
* :func:`axis_phantom` - straight and 45-degree cylinders and an elliptical one.
  Checks the axis length and the area-equivalent diameter.
"""
from __future__ import annotations

import numpy as np

from .._blocks import edt
from ..canals import openings
from ..skeleton import axis_segments

STRUCTURES = [("canal 20 um", "cylinder", 20.0), ("canal 30 um", "cylinder", 30.0),
              ("canal 50 um", "cylinder", 50.0), ("canal 100 um", "cylinder", 100.0),
              ("thin cylinder", "cylinder", 10.0), ("flat slit", "slit", 10.0),
              ("lacunae 12 um", "spheres", 12.0)]


def criterion_phantom(nz=120, ny=160, nx=220, plate=120, voxel_um=5.0):
    """``(pore, labels)``: pore space of the plate and one label per structure."""
    y0 = (ny - plate) // 2
    tissue = np.zeros((nz, ny, nx), bool)
    tissue[:, y0:y0 + plate, :] = True
    pore = np.zeros_like(tissue)
    lab = np.zeros(tissue.shape, np.uint8)
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    step = nx // (len(STRUCTURES) + 1)
    cy = y0 + plate // 2
    for i, (_, kind, d) in enumerate(STRUCTURES, 1):
        cx = i * step
        r = d / 2.0 / voxel_um
        if kind == "cylinder":
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        elif kind == "slit":
            m = (np.abs(yy - cy) <= max(0.5, r)) & (np.abs(xx - cx) <= 12)
        else:
            m = np.zeros_like(tissue)
            for k, zc in enumerate(range(15, nz - 10, 25)):
                m |= ((zz - zc) ** 2 + (yy - cy - (k % 3 - 1) * 12) ** 2
                      + (xx - cx) ** 2) <= r ** 2
        m &= tissue
        pore |= m
        lab[m] = i
    return pore, lab


def run_criterion_phantom(voxel_um=5.0, canal_um=20.0) -> list[dict]:
    from skimage.morphology import skeletonize
    pore, lab = criterion_phantom(voxel_um=voxel_um)
    r = canal_um / 2.0 / voxel_um
    _, canal = openings(pore, [r], r)
    diam = edt(pore) * 2 * voxel_um
    rows = []
    for i, (name, kind, d) in enumerate(STRUCTURES, 1):
        m = lab == i
        row = {"structure": name, "true_um": d,
               "kept_pct": 100.0 * (canal & m).sum() / m.sum()}
        if kind == "cylinder":
            row["calibre_um"] = float(np.median(diam[m & skeletonize(m)]))
        rows.append(row)
    return rows


def axis_phantom(kind: str, d_um: float, voxel_um=5.0, n=160):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    c = n // 2
    if kind == "straight":
        r = d_um / 2 / voxel_um
        return (yy - c) ** 2 + (xx - c) ** 2 <= r ** 2
    if kind == "45deg":
        r = d_um / 2 / voxel_um
        m = (yy - c) ** 2 + ((xx - c - (zz - c)) / np.sqrt(2)) ** 2 <= r ** 2
        m[:20] = m[-20:] = False
        m[:, :, :10] = m[:, :, -10:] = False
        return m
    if kind == "elliptic":   # d_um = minor axis; major = d_um / 0.6
        a, b = d_um / 0.6 / 2 / voxel_um, d_um / 2 / voxel_um
        return ((yy - c) / b) ** 2 + ((xx - c) / a) ** 2 <= 1
    raise ValueError(kind)


def run_axis_phantom(voxel_um=5.0) -> list[dict]:
    rows = []
    for kind, d in [("straight", 30), ("45deg", 30), ("straight", 50), ("45deg", 50),
                    ("straight", 80), ("45deg", 80), ("elliptic", 36)]:
        m = axis_phantom(kind, d, voxel_um)
        vol, canal = openings(m, [2.0], 2.0)
        seg = axis_segments(canal, m, voxel_um)
        L = seg["length_um"]
        deq = 2 * np.sqrt(vol[2.0] * voxel_um ** 3 / seg["axis_length_um"] / np.pi)
        true_deq = 2 * np.sqrt((d / 0.6 / 2) * (d / 2)) if kind == "elliptic" else d
        rows.append({"phantom": f"{kind} {d} um",
                     "calibre_um": float((seg["calibre_um"] * L).sum() / L.sum()),
                     "Deq_um": float(deq), "true_Deq_um": float(true_deq),
                     "length_mst_um": float(L.sum()),
                     "length_voxel_count_um": float(seg["length_naive_um"].sum())})
    return rows


def run_all() -> list[str]:
    out = ["criterion phantom (voxel 5 um)"]
    for r in run_criterion_phantom():
        cal = f"  calibre {r['calibre_um']:.1f} um" if "calibre_um" in r else ""
        out.append(f"  {r['structure']:<15} kept {r['kept_pct']:5.1f} %{cal}")
    out.append("axis phantom")
    for r in run_axis_phantom():
        out.append(f"  {r['phantom']:<16} calibre {r['calibre_um']:5.1f}  D_eq "
                   f"{r['Deq_um']:5.1f} (true {r['true_Deq_um']:5.1f})  length MST "
                   f"{r['length_mst_um']:5.0f} / voxel count {r['length_voxel_count_um']:5.0f} um")
    return out
