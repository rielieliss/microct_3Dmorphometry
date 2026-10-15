"""Block A: standard 3D morphometry with CTAn's definitions.

Three definitions follow CTAn:
Obj.N uses 6-connectivity while closed pores use 26; Conn is
Obj.N + Po.N(cl) - Eu.N (not 1 - Eu.N); and Po(cl) is divided by BV + Po.V(cl),
not by TV.

Two caveats that belong in any report:

* Conn mixes connectivities (bone 6 for Obj.N, background 26 for Po.N(cl), bone 26
  for Eu.N). It is CTAn's convention, not a strictly consistent Betti number; the
  difference comes from bone fragments touching only at vertices and matters only on
  small VOIs with loose fragments.
* BS includes the external surface of the fragment and the cut faces of the VOI, so
  BS, BS/BV and BS/TV also depend on the geometry of the cut.
"""
from __future__ import annotations

import numpy as np

from ._blocks import count_components, euler_number
from .closed_pores import Pore
from .geometry import box_counts, fractal_dimension, surface_area


def bone_morphometry(bone: np.ndarray, pores: list[Pore], voxel_mm: float,
                     object_connectivity: int = 6,
                     fd_sizes=(12, 16, 24, 32, 48, 64, 100), fd_fit_factor: int = 3,
                     max_voxels: float = 2.0e8) -> dict:
    """The parameters that need only the bone mask (and the closed pores).

    Split from :func:`tissue_morphometry` so that the pipeline never holds the bone,
    the envelope and the closed-pore labels at the same time.
    """
    vox3 = voxel_mm ** 3
    obj_n = count_components(bone, object_connectivity, max_voxels)
    eu = euler_number(bone, connectivity=3)
    counts = box_counts(bone, sorted(set(fd_sizes)))
    fd, fd_r2, used = fractal_dimension(counts, fd_sizes, bone.shape[0], fd_fit_factor)
    po_v = float(sum(p.volume_mm3 for p in pores))
    BV = float(bone.sum()) * vox3
    BS = surface_area(bone, voxel_mm)
    return {
        "BV_mm3": BV, "BS_mm2": BS, "BS_BV_per_mm": BS / BV, "Obj_N": obj_n,
        "PoN_cl": len(pores), "PoV_cl_mm3": po_v,
        "PoS_cl_mm2": float(np.nansum([p.surface_mm2 for p in pores])),
        "Po_cl_pct": 100.0 * po_v / (BV + po_v),
        "EuN": eu, "Conn": obj_n + len(pores) - eu,
        "FD": fd, "FD_R2": fd_r2,
        "FD_box_range_vox": f"{used[0]}-{used[-1]}" if used else "",
    }


def tissue_morphometry(bone_stats: dict, tissue: np.ndarray, voxel_mm: float) -> dict:
    """The parameters that need the envelope (TV without guard, as in CTAn)."""
    TV = float(tissue.sum()) * voxel_mm ** 3
    BV, BS = bone_stats["BV_mm3"], bone_stats["BS_mm2"]
    po_tot = TV - BV
    return {
        "TV_mm3": TV, "BV_TV_pct": 100.0 * BV / TV, "BS_TV_per_mm": BS / TV,
        "EuN_TV_per_mm3": bone_stats["EuN"] / TV,
        "ConnDn_per_mm3": bone_stats["Conn"] / TV,
        "TS_mm2": surface_area(tissue, voxel_mm),
        # by subtraction: kept for comparison with CTAn only. With BV ~97 % of TV the
        # relative error of TV is amplified about 20x on the remaining 3 %.
        "Po_tot_subtraction_pct": 100.0 * po_tot / TV,
        "Po_op_subtraction_pct": 100.0 * (po_tot - bone_stats["PoV_cl_mm3"]) / TV,
    }


def ctan_morphometry(bone: np.ndarray, tissue: np.ndarray, pores: list[Pore],
                     voxel_mm: float, object_connectivity: int = 6,
                     fd_sizes=(12, 16, 24, 32, 48, 64, 100), fd_fit_factor: int = 3,
                     max_voxels: float = 2.0e8) -> dict:
    """All CTAn-equivalent parameters of one VOI (convenience wrapper).

    ``tissue`` is the shrink-wrap envelope WITHOUT the guard zone, as in CTAn.
    ``pores`` comes from :func:`microct3d.closed_pores.measure_closed_pores`.
    """
    out = bone_morphometry(bone, pores, voxel_mm, object_connectivity, fd_sizes,
                           fd_fit_factor, max_voxels)
    out.update(tissue_morphometry(out, tissue, voxel_mm))
    return out
