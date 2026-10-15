"""The axis of the canal network: segments, true lengths, calibre and orientation.

The axis is the 3D skeleton of the space that passed the canal criterion (not of the
whole pore space: narrow necks and short stubs at branch points would otherwise enter
the measurement). It is cut at branch points; every segment with at least
``min_segment_vox`` voxels gets

* its calibre: median diameter of the inscribed sphere along it (smallest width of the
  lumen);
* its length: the minimum spanning tree of its voxels with steps of 1, sqrt2 or sqrt3
  voxel, plus one voxel. Counting voxels underestimates an oblique segment by up to
  42 % (32 % measured on a 45-degree phantom);
* optionally its orientation to the local surface normal.

The area-equivalent diameter (the 3D counterpart of a contour-area diameter) uses the
stereological relation mean cross-section = volume / axis length:
``D_eq = 2 sqrt(V / (pi L))``. Inscribed sphere and D_eq agree on circular sections;
D_eq is larger on any other. Because junction volume is included, D_eq tends to
overestimate the true area and should be read as an upper bound.
"""
from __future__ import annotations

import numpy as np

from ._blocks import edt, slabs


def _arc_points(skel: np.ndarray, max_voxels: float = 2.0e8) -> np.ndarray:
    """Coordinates of skeleton voxels with at most two neighbours (arcs)."""
    from scipy import ndimage as ndi
    nz = skel.shape[0]
    k = np.ones((3, 3, 3), np.uint8)
    parts = []
    for z0, z1 in slabs(nz, skel.size, max_voxels):
        a, b = max(0, z0 - 1), min(nz, z1 + 1)
        sub = np.ascontiguousarray(skel[a:b])
        cnt = ndi.convolve(sub.astype(np.uint8), k, mode="constant")
        arc = sub[z0 - a:z1 - a] & (cnt[z0 - a:z1 - a] <= 3)   # self + <= 2 neighbours
        p = np.argwhere(arc).astype(np.int32)
        if len(p):
            p[:, 0] += z0
            parts.append(p)
    return np.concatenate(parts) if parts else np.zeros((0, 3), np.int32)


def _graph(points: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of 26-adjacent points, and the MST length of each."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    n = len(points)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    nz, ny, nx = shape
    lin = (points[:, 0].astype(np.int64) * ny + points[:, 1]) * nx + points[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_sorted = lin[order]
    rows, cols, w = [], [], []
    for dz in (0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                az = points[:, 0].astype(np.int64) + dz
                ay = points[:, 1].astype(np.int64) + dy
                ax = points[:, 2].astype(np.int64) + dx
                ok = np.flatnonzero((az < nz) & (ay >= 0) & (ay < ny) & (ax >= 0) & (ax < nx))
                if ok.size == 0:
                    continue
                target = (az[ok] * ny + ay[ok]) * nx + ax[ok]
                pos = np.searchsorted(lin_sorted, target)
                valid = pos < n
                pos = np.where(valid, pos, 0)
                hit = valid & (lin_sorted[pos] == target)
                if hit.any():
                    rows.append(ok[hit])
                    cols.append(order[pos[hit]])
                    w.append(np.full(int(hit.sum()), np.sqrt(dz * dz + dy * dy + dx * dx)))
    if rows:
        r, c, ww = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    else:
        r = c = np.zeros(0, np.int64); ww = np.zeros(0)
    g = coo_matrix((ww, (r, c)), shape=(n, n)).tocsr()
    nc, lab = connected_components(g, directed=False)
    mst = minimum_spanning_tree(g).tocoo()
    length = np.bincount(lab[mst.row], weights=mst.data, minlength=nc)
    return lab, length


def _values_at(field_source: np.ndarray, points: np.ndarray, func, halo: int,
               max_voxels: float = 2.0e8) -> np.ndarray:
    """Evaluate ``func(field_source[slab])`` at ``points``, slab by slab."""
    nz = field_source.shape[0]
    out = None
    for z0, z1 in slabs(nz, field_source.size, max_voxels):
        sel = np.flatnonzero((points[:, 0] >= z0) & (points[:, 0] < z1))
        if sel.size == 0:
            continue
        a, b = max(0, z0 - halo), min(nz, z1 + halo)
        f = func(np.ascontiguousarray(field_source[a:b]))
        p = points[sel]
        vals = f[..., p[:, 0] - a, p[:, 1], p[:, 2]]
        if out is None:
            out = np.zeros(vals.shape[:-1] + (len(points),), np.float32)
        out[..., sel] = vals
        del f
    return out if out is not None else np.zeros(len(points), np.float32)


def _surface_normals(tissue: np.ndarray, points: np.ndarray, sigma: float = 4.0
                     ) -> np.ndarray:
    from scipy import ndimage as ndi

    def grad(block):
        d = ndi.gaussian_filter(edt(block).astype(np.float32), sigma, mode="nearest")
        return np.stack(np.gradient(d))
    return _values_at(tissue, points, grad, int(np.ceil(3 * sigma)) + 2).T


def axis_segments(canal: np.ndarray, pore: np.ndarray, voxel_um: float,
                  min_segment_vox: int = 4, tissue: np.ndarray | None = None,
                  orientation_cutoff_deg: float = 45.0) -> dict:
    """Skeletonize the canal space and measure its segments.

    ``canal``: space that passed the canal criterion. ``pore``: pore space after the
    lamella filter, where the inscribed sphere is measured. ``tissue``: if given,
    segment orientation to the local surface normal is computed.

    Returns arrays ``calibre_um``, ``length_um``, ``length_naive_um`` (voxel count)
    and, with tissue, ``angle_deg``; plus ``axis_length_um`` of the WHOLE skeleton
    (arcs and branch voxels), used for D_eq.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(canal)
    whole = np.argwhere(skel).astype(np.int32)
    _, whole_len = _graph(whole, skel.shape)
    axis_length_vox = float(whole_len.sum() + len(whole_len))
    pts = _arc_points(skel)
    shape = skel.shape
    del skel, whole
    lab, seg_len = _graph(pts, shape)
    n = len(seg_len)
    radius = _values_at(pore, pts, edt, halo=100)
    nvox = np.bincount(lab, minlength=n)
    order = np.argsort(lab, kind="stable")
    r_sorted = radius[order]
    start = np.searchsorted(lab[order], np.arange(n), side="left")
    stop = np.searchsorted(lab[order], np.arange(n), side="right")
    keep = nvox >= min_segment_vox
    idx = np.flatnonzero(keep)
    calibre = np.array([np.median(r_sorted[start[i]:stop[i]]) * 2 * voxel_um for i in idx])
    out = {"calibre_um": calibre,
           "length_um": (seg_len[idx] + 1.0) * voxel_um,
           "length_naive_um": nvox[idx] * voxel_um,
           "axis_length_um": axis_length_vox * voxel_um}
    if tissue is not None and len(idx):
        p_sorted = pts[order]
        centroids, directions = [], []
        for i in idx:
            p = p_sorted[start[i]:stop[i]].astype(np.float64)
            centroids.append(np.clip(p.mean(0).round().astype(int), 0,
                                     np.array(shape) - 1))
            q = p - p.mean(0)
            directions.append(np.linalg.svd(q, full_matrices=False)[2][0])
        normals = _surface_normals(tissue, np.array(centroids))
        nn = np.linalg.norm(normals, axis=1)
        cos = np.abs(np.einsum("ij,ij->i", np.array(directions),
                               normals / np.where(nn == 0, 1, nn)[:, None]))
        angle = np.degrees(np.arccos(np.clip(cos, 0, 1)))
        angle[nn == 0] = np.nan
        out["angle_deg"] = angle
        out["longitudinal"] = angle > orientation_cutoff_deg
    return out


def summarize_axis(seg: dict, canal_volume_vox: int, cavity_volume_vox: int,
                   voxel_um: float, bands_um=(20.0, 70.0, 100.0)) -> dict:
    """Length fractions by calibre band, length-weighted calibre and D_eq."""
    cal, L = seg["calibre_um"], seg["length_um"]
    tot = float(L.sum())
    out: dict = {"n_segments": int(len(L)), "axis_length_mm": seg["axis_length_um"] / 1000.0}
    if tot <= 0:
        return out
    edges = [0.0] + list(bands_um) + [np.inf]
    names = ["below_%g" % bands_um[0]] + ["%g_%g" % (a, b) for a, b in
                                           zip(bands_um[:-1], bands_um[1:])] + \
            ["above_%g" % bands_um[-1]]
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        out["axis_%s_pct" % name] = 100.0 * float(L[(cal >= lo) & (cal < hi)].sum()) / tot
    out["CaDm_length_weighted_um"] = float((cal * L).sum() / tot)
    out["CaDm_unweighted_um"] = float(cal.mean())
    vox3 = voxel_um ** 3
    Lw = seg["axis_length_um"]
    out["Deq_network_um"] = 2 * np.sqrt(canal_volume_vox * vox3 / Lw / np.pi)
    small = cal < bands_um[-1]
    Lc = Lw * float(L[small].sum()) / tot
    if Lc > 0:
        out["Deq_canals_um"] = 2 * np.sqrt((canal_volume_vox - cavity_volume_vox) * vox3
                                           / Lc / np.pi)
        out["CaDm_canals_length_weighted_um"] = float((cal[small] * L[small]).sum()
                                                      / L[small].sum())
    if "longitudinal" in seg:
        out["longitudinal_length_pct"] = 100.0 * float(L[seg["longitudinal"]].sum()) / tot
    return out
