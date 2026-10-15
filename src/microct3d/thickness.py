"""Cortical thickness (Ct.Th) by perpendicular lines in 2D.

1. Ten equally spaced slices along z (5 % margins left out: that is where the VOI
   cuts the bone).
2. In each slice, the PLATE: bone with pores filled and the outline smoothed, which is
   what a caliper would measure. Without filling, every pore breaks the medial axis.
3. The medial axis of the plate, with short branches pruned.
4. At points along the axis, a line PERPENDICULAR to the local axis direction (PCA of
   neighbouring axis points) walks both ways while inside the plate.
5. Ct.Th = mean of the lines in the THICKER HALF OF EACH SLICE. The plate thins at the
   two ends, where the VOI cuts it obliquely, and there the perpendicular measures a
   wedge. By construction this is larger than the mean thickness.

Lines that reach the image border are discarded. As an internal check, twice the
distance transform at the same point (the inscribed circle) should match the line
length on a plate; their ratio is reported.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def plate(mask: np.ndarray, min_area: int = 500, closing_radius: int = 4) -> np.ndarray:
    from skimage import morphology
    m = ndi.binary_fill_holes(mask)
    if closing_radius:
        m = ndi.binary_closing(m, morphology.disk(closing_radius))
        m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n > 1:
        cnt = np.bincount(lbl.ravel())
        cnt[0] = 0
        m = np.isin(lbl, np.flatnonzero(cnt >= min_area))
    return m


def medial_axis(mask: np.ndarray, min_branch: int = 12) -> np.ndarray:
    """2D skeleton with branches shorter than ``min_branch`` pixels pruned."""
    from skimage import morphology
    sk = morphology.skeletonize(mask)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(min_branch):
        nb = ndi.convolve(sk.astype(np.uint8), kernel, mode="constant") - 1
        tips = sk & (nb == 1)
        if not tips.any():
            break
        sk = sk & ~tips
    return sk


def _direction(points: np.ndarray, i: int, radius: int):
    d = points - points[i]
    near = points[(np.abs(d) <= radius).all(axis=1)]
    if len(near) < 5:
        return None
    c = near - near.mean(axis=0)
    _, v = np.linalg.eigh(c.T @ c)
    return v[:, -1]


def _walk(mask, y, x, dy, dx, step=0.5, limit=400):
    H, W = mask.shape
    t = 0.0
    while t < limit:
        t += step
        yy, xx = int(round(y + dy * t)), int(round(x + dx * t))
        if yy < 0 or xx < 0 or yy >= H or xx >= W:
            return t, True
        if not mask[yy, xx]:
            return t - step / 2, False
    return t, True


def measure_slice(mask: np.ndarray, spacing: int = 8, pca_radius: int = 25,
                  min_area: int = 500) -> list[tuple]:
    """``[(y, x, forward_px, backward_px, ny, nx, edt_px)]`` for one slice."""
    m = plate(mask, min_area)
    if not m.any():
        return []
    sk = medial_axis(m, min_branch=30)
    if not sk.any():
        return []
    pts = np.argwhere(sk)
    dist = ndi.distance_transform_edt(m)
    out = []
    for i in range(0, len(pts), spacing):
        d = _direction(pts, i, pca_radius)
        if d is None:
            continue
        ny, nx = -d[1], d[0]
        y, x = float(pts[i][0]), float(pts[i][1])
        a, hit_a = _walk(m, y, x, ny, nx)
        b, hit_b = _walk(m, y, x, -ny, -nx)
        if hit_a or hit_b:
            continue
        out.append((pts[i][0], pts[i][1], a, b, ny, nx, dist[pts[i][0], pts[i][1]]))
    return out


def cortical_thickness(volume: np.ndarray, voxel_um: float, n_slices: int = 10,
                       spacing: int = 8, pca_radius: int = 25) -> dict:
    """Ct.Th of a VOI (``volume`` = the full binary stack, True = bone), in mm."""
    nz = volume.shape[0]
    margin = max(1, int(0.05 * nz))
    idx = np.unique(np.linspace(margin, nz - 1 - margin, n_slices).round().astype(int))
    rows = []
    for k in idx:
        for t in measure_slice(np.ascontiguousarray(volume[k]), spacing, pca_radius):
            rows.append(t + (int(k),))
    if not rows:
        return {"CtTh_mm": float("nan")}
    th = np.array([t[2] + t[3] for t in rows]) * voxel_um / 1000.0
    inscribed = np.array([t[6] for t in rows]) * 2 * voxel_um / 1000.0
    sl = np.array([t[7] for t in rows])
    body = np.zeros(len(th), bool)
    for k in np.unique(sl):
        s = sl == k
        body[s] = th[s] >= np.median(th[s])
    return {
        "CtTh_mm": float(th[body].mean()),
        "CtTh_sd_mm": float(th[body].std(ddof=1)) if body.sum() > 1 else float("nan"),
        "CtTh_all_lines_mm": float(th.mean()),
        "CtTh_n_lines": int(len(th)), "CtTh_n_lines_used": int(body.sum()),
        "CtTh_line_to_circle_ratio": float((th[body] / inscribed[body]).mean()),
    }
