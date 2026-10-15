"""One pass over one VOI: every parameter of the method, from the binary stack.

    binary VOI (bone = True)
      |- cortical thickness (2D, on the full stack) ............ Ct.Th
      |- crop to bone + margin
      |- ENVELOPE (3D shrink-wrap) ............................. TV
      |- closed pores (CTAn definition) ........................ Block B
      |- CTAn-equivalent morphometry ........................... Block A
      |- GUARD ZONE -> pore space -> LAMELLA FILTER
      |    |- CANAL CRITERION (a 20 µm sphere fits) ............ Block C
      |    |    |- open / closed canal objects ................. Ca.V/TV, Ca.N/TV, Po.V/TV
      |    |    |- axis of the canal space ..................... calibre, axis fractions, D_eq
      |    |- local thickness bands, wall re-entrance .......... V(20-70)/TV, ...
"""
from __future__ import annotations

import time

import numpy as np

from . import canals, closed_pores, envelope, morphometry, skeleton, thickness
from .config import Params


def analyze_voi(bone_full: np.ndarray, p: Params = Params(), log=print,
                with_orientation: bool = False) -> dict:
    """Analyse one binary VOI and return a flat dict of parameters.

    ``bone_full``: boolean ``[z, y, x]`` array, True = bone, background outside the ROI.
    It is not modified.

    Memory: at most three full-size boolean arrays are alive at any time (plus the
    closed-pore labels while the bone is the only other array), about 3 bytes per voxel.
    """
    t0 = time.time()
    vmm = p.voxel_mm
    vox3 = vmm ** 3
    res: dict = {"n_slices": int(bone_full.shape[0])}

    # ---- cortical thickness: 2D, ten slices of the full stack
    res.update(thickness.cortical_thickness(bone_full, p.voxel_um, p.ctth_n_slices,
                                            p.ctth_line_spacing, p.ctth_pca_radius))

    # ---- crop (own copy: the arrays below are modified in place)
    margin = max(2 * p.shrinkwrap_diameter_vox, 20)
    bone, _ = envelope.crop_to_bone(bone_full, margin)
    if bone is bone_full:
        bone = bone_full.copy()
    log(f"  volume {bone.shape} ({bone.size / 1e9:.2f} G voxels)")

    # ---- block B and the bone-only part of block A (bone + pore labels)
    pores, labels = closed_pores.measure_closed_pores(
        bone, vmm, p.closed_pore_connectivity, p.shape_min_vox)
    del labels
    stats = morphometry.bone_morphometry(bone, pores, vmm, p.object_connectivity,
                                         p.fd_box_sizes_vox, p.fd_box_fit_factor)

    # ---- envelope: the rest of block A (bone + tissue)
    tissue = envelope.shrinkwrap(bone, p.shrinkwrap_diameter_vox, p.max_block_voxels)
    res.update(stats)
    res.update(morphometry.tissue_morphometry(stats, tissue, vmm))
    res.update(closed_pores.summarize(pores, res["TV_mm3"]))
    log(f"  block A/B done ({time.time() - t0:.0f} s)")

    # ---- block C: everything below is divided by the tissue volume WITH guard
    tissue_g = envelope.guard(tissue, p.guard_vox, p.max_block_voxels)
    del tissue
    TVg = float(tissue_g.sum()) * vox3
    res["TV_guard_mm3"] = TVg
    np.logical_not(bone, out=bone)
    np.logical_and(bone, tissue_g, out=bone)            # bone := pore space with guard
    pore, n_before, n_after = canals.remove_lamellae(bone, p.lamella_min_extent_vox)
    del bone
    res["lamella_objects_removed"] = n_before - n_after
    raw = int(pore.sum())

    radii = {d: p.radius_vox(d) for d in p.sensitivity_diameters_um}
    r_main = p.radius_vox(p.canal_diameter_um)
    radii.setdefault(p.canal_diameter_um, r_main)
    volumes, canal = canals.openings(pore, sorted(set(radii.values())), r_main,
                                     p.max_block_voxels)
    for d, r in sorted(radii.items()):
        res[f"V_sphere{d:g}um_TVg_pct"] = 100.0 * volumes[r] * vox3 / TVg
    open_v, closed_v = canals.open_and_closed(canal, tissue_g)
    if not with_orientation:
        del tissue_g
        tissue_g = None
    res.update({
        "CaV_TV_pct": 100.0 * sum(open_v) * vox3 / TVg,
        "CaN_per_mm3": len(open_v) / TVg,
        "PoV_cl_criterion_TVg_pct": 100.0 * sum(closed_v) * vox3 / TVg,
        "PoV_TV_pct": 100.0 * (sum(open_v) + sum(closed_v)) * vox3 / TVg,
        "PoN_per_mm3": (len(open_v) + len(closed_v)) / TVg,
        "excess_removed_x": raw / max(1, sum(open_v) + sum(closed_v)),
    })
    log(f"  canal criterion done ({time.time() - t0:.0f} s)")

    bands_d = tuple(p.size_bands_um)
    sb = canals.size_bands(pore, bands_d, p.cavity_diameter_um, p.reentrance_bands_vox,
                           p.voxel_um)
    V = sb["V"]
    for lo, hi in zip(bands_d[:-1], bands_d[1:]):
        res[f"V_{lo:g}_{hi:g}um_TV_pct"] = 100.0 * (V[lo] - V[hi]) * vox3 / TVg
    res[f"V_above_{bands_d[-1]:g}um_TV_pct"] = 100.0 * V[bands_d[-1]] * vox3 / TVg
    res[f"V_above_{bands_d[-1]:g}um_of_pore_pct"] = (100.0 * V[bands_d[-1]] / V[bands_d[0]]
                                                     if V[bands_d[0]] else float("nan"))
    if sb["area_smooth"]:
        s = p.reentrance_main_vox
        res["wall_reentrance_depth_um"] = (sb["vol_reentrant"][s] / sb["area_smooth"]
                                           * p.voxel_um)

    seg = skeleton.axis_segments(canal, pore, p.voxel_um, p.min_segment_vox, tissue_g,
                                 p.orientation_cutoff_deg)
    res.update(skeleton.summarize_axis(seg, V[bands_d[0]], V[bands_d[-1]], p.voxel_um,
                                       bands_d))
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  done ({res['seconds']:.0f} s)")
    return res
