"""Block C: the canal network.

What is a canal? Not every void is. The criterion is LOCAL and of SIZE: a point of the
pore space is canal if a sphere of 20 µm diameter fits there (morphological opening by
a ball). A slit, a residual lamella or the gap left by the envelope cannot hold the
sphere and are removed; a canal can.

Why not a shape criterion? Requiring a tube shape was tried on two units and works on
neither. On the skeleton segment between branch points it discards 72-78 % of the
volume (in a branched network every segment is short by construction). On the
connected object it discards 91 % at once (Haversian and Volkmann canals connect, the
whole network is one object, and its shape is the shape of the VOI). "Is this a
canal?" only makes sense point by point, which is what the sphere does.

All porosities share ONE denominator, the tissue volume with guard, so total porosity
is the SUM of canal and closed porosity, never TV - BV.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ._blocks import ComponentScan, _label, edt, filter_min_extent, slabs


def remove_lamellae(pore: np.ndarray, min_extent_vox: int = 3,
                    connectivity: int = 26) -> tuple[np.ndarray, int, int]:
    """Discard pore objects thinner than ``min_extent_vox`` in their smallest
    bounding-box side: segmentation lamellae, not pores. Returns ``(kept, n_before, n_after)``."""
    return filter_min_extent(pore, min_extent_vox, connectivity)


def openings(mask: np.ndarray, radii: list[float], main_radius: float,
             max_voxels: float = 1.5e8) -> tuple[dict[float, int], np.ndarray]:
    """Morphological openings of ``mask`` by balls of several radii, in one pass.

    Opening by a ball of radius r keeps the voxels reached by a ball that fits
    entirely inside the mask: core = ``edt > r``, dilated by r, intersected with the
    mask. Dilation is done by distance (``edt(~core) <= r``), identical on a voxel
    lattice and much faster than a structuring element.

    Returns ``({radius: volume in voxels}, opening mask for main_radius)``.
    """
    rmax = max(radii)
    halo = 2 * int(np.ceil(rmax)) + 2
    nz = mask.shape[0]
    volumes = {r: 0 for r in radii}
    out = np.zeros_like(mask)
    for z0, z1 in slabs(nz, mask.size, max_voxels):
        a, b = max(0, z0 - halo), min(nz, z1 + halo)
        d = edt(mask[a:b])
        for r in radii:
            core = d > r
            if not core.any():
                continue
            reached = edt(~core) <= r
            del core
            sub = reached[z0 - a:z1 - a] & mask[z0:z1]
            del reached
            volumes[r] += int(sub.sum())
            if r == main_radius:
                out[z0:z1] = sub
        del d
    return volumes, out


def _outside_near(tissue: np.ndarray, z0: int, z1: int) -> np.ndarray:
    """Voxels of slab [z0, z1) that are outside the tissue or 6-adjacent to it.

    Computed from the tissue mask slab by slab, so the complement of the tissue is
    never materialised for the whole volume.
    """
    f = ~tissue[z0:z1]
    v = f.copy()
    v[:, 1:, :] |= f[:, :-1, :]; v[:, :-1, :] |= f[:, 1:, :]
    v[:, :, 1:] |= f[:, :, :-1]; v[:, :, :-1] |= f[:, :, 1:]
    v[1:] |= f[:-1]; v[:-1] |= f[1:]
    if z0 > 0:
        v[0] |= ~tissue[z0 - 1]
    if z1 < tissue.shape[0]:
        v[-1] |= ~tissue[z1]
    return v


def open_and_closed(canal: np.ndarray, tissue: np.ndarray, connectivity: int = 26,
                    max_voxels: float = 2.0e8) -> tuple[list[int], list[int]]:
    """Split the canal space into OPEN objects (touching the tissue boundary or a face
    of the volume) and CLOSED ones (entirely surrounded by bone).

    ``tissue`` is the tissue mask with guard. In a biopsy almost the whole network is
    open, because the canals emerge on the cut. Contact is tested with full
    6-adjacency, including along z inside a slab (see CHANGELOG 1.0.1). Returns the voxel counts of the open and of the
    closed objects.
    """
    scan = ComponentScan(canal, connectivity, max_voxels)
    volume: dict[int, int] = defaultdict(int)
    touching: set[int] = set()
    for z0, z1, base, n in scan._slabs:
        lbl, _ = _label(canal[z0:z1], connectivity)
        cnt = np.bincount(lbl.ravel(), minlength=n + 1)
        hit = np.unique(lbl[_outside_near(tissue, z0, z1) & (lbl > 0)])
        for x in np.flatnonzero(cnt[1:]) + 1:
            volume[scan.uf.find(base + int(x))] += int(cnt[x])
        for x in hit.tolist():
            if x:
                touching.add(scan.uf.find(base + int(x)))
        del lbl
    is_open = {r: (r in touching) or scan.touches_border[r] for r in volume}
    return ([v for r, v in volume.items() if is_open[r]],
            [v for r, v in volume.items() if not is_open[r]])


# --------------------------------------------------------- size bands and wall
def _faces(a, z_lo, z_hi, region=None) -> int:
    """Voxel faces between True and False in ``a``, counted for z rows in
    ``[z_lo, z_hi)``; with ``region``, only faces with one voxel inside it."""
    n = 0
    zt = min(z_hi, a.shape[0] - 1)
    if zt > z_lo:
        d = a[z_lo:zt] != a[z_lo + 1:zt + 1]
        if region is not None:
            d &= region[z_lo:zt] | region[z_lo + 1:zt + 1]
        n += int(d.sum())
    s = a[z_lo:z_hi]
    r = region[z_lo:z_hi] if region is not None else None
    for ax in (1, 2):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[ax] = slice(0, -1); s1[ax] = slice(1, None)
        d = s[tuple(s0)] != s[tuple(s1)]
        if r is not None:
            d &= r[tuple(s0)] | r[tuple(s1)]
        n += int(d.sum())
    return n


def size_bands(pore: np.ndarray, diameters_um, cavity_um: float, shell_vox,
               voxel_um: float, max_voxels: float = 1.5e8) -> dict:
    """Pore volume by local thickness, and the wall of the enlarged spaces.

    Each point of pore space gets its local thickness (Hildebrand & Rüegsegger): the
    diameter of the largest sphere inside the pore that contains it. "Volume where a
    sphere of D fits" is the opening by a ball of diameter D, so a band is a
    difference of openings, e.g. V(20-70) = V(20) - V(70).

    Wall re-entrance (exploratory): filling every space wider than ``cavity_um`` with
    balls of that diameter gives a "smooth cavity" with no re-entrance smaller than
    the ball. The pore volume left outside it, within ``s`` voxels of its surface,
    divided by its surface area, is the mean depth of the re-entrances. A smooth wall
    gives ~0; a scalloped wall more. There is no external reference that validates it.

    Returns voxel counts ``V[D]``, the smooth-cavity area (faces), the real wall area
    and the re-entrant volume for each shell width.
    """
    radii = {d: (d / 2.0) / voxel_um for d in diameters_um}
    r_cav = (cavity_um / 2.0) / voxel_um
    halo = 2 * int(np.ceil(r_cav)) + max(shell_vox) + 4
    nz = pore.shape[0]
    V = {d: 0 for d in diameters_um}
    area_smooth = 0
    area_real = {s: 0 for s in shell_vox}
    vol_reent = {s: 0 for s in shell_vox}
    for z0, z1 in slabs(nz, pore.size, max_voxels):
        a, b = max(0, z0 - halo), min(nz, z1 + halo)
        m = np.ascontiguousarray(pore[a:b])
        if not m.any():
            continue
        i0, i1 = z0 - a, z1 - a
        dist = edt(m)
        cavity = None
        for d_um in diameters_um:
            r = radii[d_um]
            core = dist > r
            if not core.any():
                continue
            reached = edt(~core) <= r
            reached &= m
            V[d_um] += int(reached[i0:i1].sum())
            if d_um == cavity_um:
                cavity = reached
        del dist
        if cavity is None or not cavity[i0:i1].any():
            continue
        area_smooth += _faces(cavity, i0, i1)
        dc = edt(~cavity)
        for s in shell_vox:
            near = dc <= s
            area_real[s] += _faces(m, i0, i1, region=near)
            vol_reent[s] += int((near[i0:i1] & m[i0:i1] & ~cavity[i0:i1]).sum())
    return {"V": V, "area_smooth": area_smooth, "area_real": area_real,
            "vol_reentrant": vol_reent}
