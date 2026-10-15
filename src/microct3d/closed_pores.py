"""Closed pores: voids entirely surrounded by bone, measured one by one.

Definition (CTAn's): background components, in 26-connectivity, that touch none of
the faces of the volume. No canal-size criterion is applied here, so the numbers stay
comparable with CTAn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._blocks import closed_pore_labels
from .geometry import mesh, mesh_area, mesh_volume


@dataclass
class Pore:
    label: int
    voxels: int
    volume_mm3: float          # enclosed by the mesh
    surface_mm2: float
    esd_um: float              # diameter of the sphere of equal volume
    sphericity: float          # pi^(1/3) (6V)^(2/3) / S; ceiling ~0.91 with this mesh
    elongation: float          # sqrt(l2/l1): ~0 = tube
    flatness: float            # sqrt(l3/l2): ~0 = slit


def _principal_ratios(mask: np.ndarray) -> tuple[float, float]:
    c = np.argwhere(mask).astype(float)
    c -= c.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]
    elong = float(np.sqrt(max(ev[1], 0) / ev[0])) if ev[0] > 0 else float("nan")
    flat = float(np.sqrt(max(ev[2], 0) / ev[1])) if ev[1] > 0 else float("nan")
    return elong, flat


def measure_closed_pores(bone: np.ndarray, voxel_mm: float, connectivity: int = 26,
                         shape_min_vox: int = 100, max_voxels: float = 2.0e8
                         ) -> tuple[list[Pore], np.ndarray]:
    """Label and measure every closed pore.

    Volume and surface come from each pore's marching-cubes mesh. Elongation and
    flatness come from the principal axes of the voxel cloud and are computed only for
    pores with at least ``shape_min_vox`` voxels (ESD ~29 µm at 5 µm): below that the
    descriptor measures the voxel grid, not the pore.

    Returns the list of pores (largest first) and the label volume.
    """
    from scipy import ndimage as ndi

    labels, n = closed_pore_labels(bone, connectivity, max_voxels)
    vox3 = voxel_mm ** 3
    pores: list[Pore] = []
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        sub = labels[sl] == i
        nvox = int(sub.sum())
        try:
            v, f = mesh(sub, voxel_mm)
            S, V = mesh_area(v, f), mesh_volume(v, f)
        except (RuntimeError, ValueError):
            S, V = float("nan"), nvox * vox3
        sph = (np.pi ** (1 / 3)) * ((6 * V) ** (2 / 3)) / S if S and S > 0 else float("nan")
        elong = flat = float("nan")
        if nvox >= shape_min_vox:
            elong, flat = _principal_ratios(sub)
        pores.append(Pore(i, nvox, V, S, (6 * V / np.pi) ** (1 / 3) * 1000.0, sph,
                          elong, flat))
    pores.sort(key=lambda p: p.volume_mm3, reverse=True)
    return pores, labels


def summarize(pores: list[Pore], tissue_volume_mm3: float) -> dict:
    """Per-VOI summary of the closed pores (Block B)."""
    if not pores:
        return {"PoN_cl_per_mm3": 0.0}
    esd = np.array([p.esd_um for p in pores])
    sph = np.array([p.sphericity for p in pores])
    el = np.array([p.elongation for p in pores])
    fl = np.array([p.flatness for p in pores])
    return {
        "PoN_cl_per_mm3": len(pores) / tissue_volume_mm3,
        "ESD_cl_mean_um": float(esd.mean()),
        "ESD_cl_sd_um": float(esd.std(ddof=1)) if len(esd) > 1 else float("nan"),
        "sphericity_cl_mean": float(np.nanmean(sph)),
        "elongation_cl_mean": float(np.nanmean(el)) if np.isfinite(el).any() else float("nan"),
        "flatness_cl_mean": float(np.nanmean(fl)) if np.isfinite(fl).any() else float("nan"),
    }
