"""Surface meshes and box counting, block-wise where volumes are large."""
from __future__ import annotations

import numpy as np

from ._blocks import slabs


def mesh(mask: np.ndarray, spacing: float):
    """Marching-cubes mesh of a binary mask padded by one voxel (closed surface).

    Returns ``(verts, faces)``; an empty mask gives an empty mesh.
    """
    from skimage import measure
    if not mask.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    v, f, _, _ = measure.marching_cubes(np.pad(mask, 1).astype(np.uint8), 0.5,
                                        spacing=(spacing,) * 3)
    return v, f


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    a = verts[faces[:, 0]]
    n = np.cross(verts[faces[:, 1]] - a, verts[faces[:, 2]] - a)
    return float(np.linalg.norm(n, axis=1).sum()) / 2.0


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by a closed mesh (divergence theorem).

    This is how CTAn measures pore volumes: counting voxels overestimates them,
    because marching cubes cuts the voxel corners (0.86x on a 27-voxel cube).
    """
    t = verts[faces]
    return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def _slab_faces(verts, faces, spacing, a, z0, z1, last):
    """Faces of a slab mesh whose centroid lies in the slab's own z range.

    Marching cubes is local, so with a halo the slab mesh reproduces the faces of the
    full mesh in that range; selecting by centroid counts every face exactly once.
    """
    zc = verts[faces].mean(axis=1)[:, 0] / spacing - 1.0 + a
    ok = zc >= z0 - 0.5 - 1e-6
    if not last:
        ok &= zc < z1 - 0.5 - 1e-6
    return faces[ok]


def surface_area(mask: np.ndarray, spacing: float, max_voxels: float = 1.5e8) -> float:
    """Marching-cubes surface area of a large mask, accumulated slab by slab."""
    total = 0.0
    nz = mask.shape[0]
    limits = list(slabs(nz, mask.size, max_voxels))
    for k, (z0, z1) in enumerate(limits):
        a, b = max(0, z0 - 2), min(nz, z1 + 2)
        v, f = mesh(np.ascontiguousarray(mask[a:b]), spacing)
        if len(f):
            total += mesh_area(v, _slab_faces(v, f, spacing, a, z0, z1,
                                              k == len(limits) - 1))
    return total


# ---------------------------------------------------------------- box counting
def _surface_voxels(block: np.ndarray) -> np.ndarray:
    """Bone voxels with at least one of the six neighbours outside the bone."""
    a = block
    d = a.copy()
    d[1:] &= a[:-1]; d[0] = False
    d[:-1] &= a[1:]; d[-1] = False
    d[:, 1:] &= a[:, :-1]; d[:, 0] = False
    d[:, :-1] &= a[:, 1:]; d[:, -1] = False
    d[:, :, 1:] &= a[:, :, :-1]; d[:, :, 0] = False
    d[:, :, :-1] &= a[:, :, 1:]; d[:, :, -1] = False
    return a & ~d


def box_counts(bone: np.ndarray, sizes, max_voxels: float = 1.5e8) -> dict[int, int]:
    """N(s): number of cubes of side ``s`` voxels that contain bone surface."""
    ny, nx = bone.shape[1], bone.shape[2]
    acc = {s: [] for s in sizes if s > 1}
    n1 = 0
    nz = bone.shape[0]
    for z0, z1 in slabs(nz, bone.size, max_voxels):
        a, b = max(0, z0 - 1), min(nz, z1 + 1)
        surf = _surface_voxels(np.ascontiguousarray(bone[a:b]))[z0 - a:z1 - a]
        p = np.argwhere(surf)
        if not len(p):
            continue
        p[:, 0] += z0
        n1 += len(p)
        for s in acc:
            q = p // s
            acc[s].append(np.unique((q[:, 0] * (ny // s + 2) + q[:, 1])
                                    * (nx // s + 2) + q[:, 2]))
    counts = {1: n1}
    for s, lst in acc.items():
        counts[s] = int(len(np.unique(np.concatenate(lst)))) if lst else 0
    return counts


def fractal_dimension(counts: dict[int, int], sizes, height_vox: int | None = None,
                      fit_factor: int = 3) -> tuple[float, float, list[int]]:
    """Slope of log N(s) against log s, sign changed, with its R².

    A box size is used only if it fits ``fit_factor`` times in the VOI height: a box
    that does not fit three times in the specimen samples no structure, and the count
    saturates.
    Returns ``(fd, r2, sizes_used)``.
    """
    use = [s for s in sizes if (not height_vox or s * fit_factor <= height_vox)
           and counts.get(s, 0) > 0]
    if len(use) < 3:
        return float("nan"), float("nan"), use
    x = np.log(np.array(use, float))
    y = np.log(np.array([counts[s] for s in use], float))
    A = np.vstack([x, np.ones_like(x)]).T
    (m, q), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - (m * x + q)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return float(-m), float(r2), use
