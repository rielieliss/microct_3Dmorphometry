"""Tissue envelope (TV) and the guard zone.

Every porosity is a ratio between void and tissue, so "inside the bone" has to be
defined. The envelope wraps the fragment like a membrane: it bridges the mouths of
canals that open on the surface and follows the larger recesses. It reproduces CTAn's
"ROI shrink-wrap (3D), stretch over holes with diameter D".
"""
from __future__ import annotations

import warnings

import numpy as np

from ._blocks import border_components, edt_greater


def crop_to_bone(volume: np.ndarray, margin: int) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Crop to the bounding box of the bone plus ``margin`` voxels of background.

    The shrink-wrap needs free space around the fragment; cropping changes no result,
    only memory.
    """
    idx = [np.flatnonzero(volume.any(axis=ax)) for ax in ((1, 2), (0, 2), (0, 1))]
    if any(len(i) == 0 for i in idx):
        raise ValueError("the volume contains no bone")
    box = tuple(slice(max(0, i[0] - margin), min(n, i[-1] + margin + 1))
                for i, n in zip(idx, volume.shape))
    if volume[box].size == volume.size:
        return volume, box
    return np.ascontiguousarray(volume[box]), box


def shrinkwrap(bone: np.ndarray, diameter_vox: int = 30,
               max_voxels: float = 1.5e8) -> np.ndarray:
    """3D shrink-wrap envelope of a binary bone volume.

    The film wraps the structure from outside and does not enter openings narrower
    than ``diameter_vox``: it is the morphological closing of the bone by a ball of
    radius ``r = D/2``, computed from the outside (background connected to the array
    border, eroded by ``r``, then dilated by ``r``).

    The z axis is treated as OPEN: the VOI is a cut and the bone continues above and
    below it, so the first and last planes are not an external surface. The volume is
    padded with ``ceil(r) + 1`` copies of nothing on z only, which keeps the erosion
    from eating the ends. Checked against the envelope exported by CTAn: TV -0.18 %,
    Jaccard 0.995 (closing the ends instead gives +2.70 %).

    Peak memory is two boolean arrays.
    """
    r = diameter_vox / 2.0
    if bone.shape[0] < diameter_vox:
        warnings.warn(f"the VOI has {bone.shape[0]} slices, fewer than D={diameter_vox}: "
                      "the envelope cannot close on the structure in z and TV will be "
                      "inflated (+23.6 % measured on a 26-slice VOI)")
    pad = int(np.ceil(r)) + 1
    p = np.zeros((bone.shape[0] + 2 * pad,) + bone.shape[1:], bool)
    p[pad:-pad] = bone
    np.logical_not(p, out=p)                               # background
    outside = border_components(p, 6, max_voxels / 2)    # background reaching the border
    del p
    eroded = edt_greater(outside, r, max_voxels)           # outside eroded by r
    del outside
    eroded = border_components(eroded, 6, max_voxels / 2)
    np.logical_not(eroded, out=eroded)
    tissue = edt_greater(eroded, r, max_voxels)            # = complement of the dilation
    del eroded
    return np.ascontiguousarray(tissue[pad:-pad])


def guard(tissue: np.ndarray, guard_vox: int, max_voxels: float = 1.5e8) -> np.ndarray:
    """Tissue eroded by ``guard_vox`` voxels.

    The envelope never touches the bone surface exactly; the thin shell between them
    would be counted as pore and would dominate the skeleton of the pore space. The
    eroded tissue is the common denominator of all porosities (TV with guard).
    """
    if guard_vox <= 0:
        return tissue
    return edt_greater(tissue, guard_vox, max_voxels)
