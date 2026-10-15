"""Memory-bounded building blocks: distance transforms and connected components
computed in slabs along z.

A full-volume label image in uint32 or a float32 distance map costs 4 bytes per voxel
(23 GB for a 5.8-gigavoxel VOI). Everything here works slab by slab and gives exactly the same result as the
single-pass computation; the tests check that for 1, 2, 4 and 8 slabs.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

try:  # exact multithreaded EDT, much lighter on memory than scipy's
    import edt as _edt_lib
except ImportError:  # pragma: no cover
    _edt_lib = None

try:  # connected components with uint32 output
    import cc3d as _cc3d
except ImportError:  # pragma: no cover
    _cc3d = None


# --------------------------------------------------------------------- slabs
def slabs(nz: int, n_voxels: int, max_voxels: float, minimum: int = 1
          ) -> Iterator[tuple[int, int]]:
    """Yield ``(z0, z1)`` slab limits covering ``range(nz)``."""
    nb = max(minimum, int(np.ceil(n_voxels / max_voxels)))
    cuts = np.linspace(0, nz, nb + 1).astype(int)
    for k in range(nb):
        z0, z1 = int(cuts[k]), int(cuts[k + 1])
        if z1 > z0:
            yield z0, z1


# ------------------------------------------------------------------- distance
def edt(mask: np.ndarray, threads: int | None = None) -> np.ndarray:
    """Exact Euclidean distance transform (distance of each True voxel to the
    nearest False voxel, in voxels)."""
    mask = np.ascontiguousarray(mask)
    if _edt_lib is not None:
        if threads:
            return _edt_lib.edt(mask, parallel=threads)
        return _edt_lib.edt(mask)
    from scipy import ndimage as ndi
    return ndi.distance_transform_edt(mask)


def edt_greater(mask: np.ndarray, r: float, max_voxels: float = 1.5e8) -> np.ndarray:
    """``edt(mask) > r`` computed slab by slab.

    Exact: a voxel's distance is decided by features at most ``r`` away, so a halo of
    ``ceil(r) + 1`` slices makes the thresholded slab identical to the full result.
    Used for erosion by a ball (envelope, guard zone).
    """
    nz = mask.shape[0]
    halo = int(np.ceil(r)) + 1
    out = np.empty(mask.shape, bool)
    for z0, z1 in slabs(nz, mask.size, max_voxels):
        a, b = max(0, z0 - halo), min(nz, z1 + halo)
        d = edt(mask[a:b])
        np.greater(d[z0 - a:z1 - a], r, out=out[z0:z1])
        del d
    return out


# --------------------------------------------------------- connected components
def _label(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    mask = np.ascontiguousarray(mask)
    if _cc3d is not None:
        lbl = _cc3d.connected_components(mask, connectivity=connectivity,
                                         out_dtype=np.uint32)
        return lbl, int(lbl.max())
    from scipy import ndimage as ndi
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    lbl, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, rank))
    return lbl.astype(np.uint32), int(n)


def _interface_shifts(connectivity: int) -> list[tuple[int, int]]:
    """In-plane offsets (dy, dx) that connect two consecutive z planes."""
    if connectivity == 6:
        return [(0, 0)]
    if connectivity == 18:
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, x: int) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:  # path compression
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _bboxes(lbl: np.ndarray, z_offset: int) -> dict[int, list[int]]:
    from scipy import ndimage as ndi
    out = {}
    for i, sl in enumerate(ndi.find_objects(lbl), 1):
        if sl is not None:
            out[i] = [sl[0].start + z_offset, sl[0].stop + z_offset,
                      sl[1].start, sl[1].stop, sl[2].start, sl[2].stop]
    return out


class ComponentScan:
    """Connected components of a large mask, found slab by slab with union-find.

    After construction, each component is identified by its union-find root, and
    ``touches_border[root]`` / ``bbox[root]`` describe it. :meth:`relabel` writes any
    per-root lookup back to a full-size array, again slab by slab.
    """

    def __init__(self, mask: np.ndarray, connectivity: int = 26,
                 max_voxels: float = 2.0e8, need_bbox: bool = False,
                 invert: bool = False) -> None:
        self.mask = mask
        self.invert = invert          # label ~mask, slab by slab, never the whole complement
        self.connectivity = connectivity
        self.max_voxels = max_voxels
        self.uf = _UnionFind()
        self._slabs: list[tuple[int, int, int, int]] = []   # z0, z1, base, n
        border: set[int] = set()
        boxes: dict[int, list[int]] = {}
        nz = mask.shape[0]
        limits = list(slabs(nz, mask.size, max_voxels))
        prev_plane = None
        base = 0
        for k, (z0, z1) in enumerate(limits):
            lbl, n = _label(self._slab(z0, z1), connectivity)
            if need_bbox:
                for i, box in _bboxes(lbl, z0).items():
                    boxes[base + i] = box
            ids = np.where(lbl > 0, lbl + np.uint32(base), np.uint32(0))
            del lbl
            for x in np.unique(ids).tolist():
                if x:
                    self.uf.add(x)
            faces = [ids[:, 0], ids[:, -1], ids[:, :, 0], ids[:, :, -1]]
            if k == 0:
                faces.append(ids[0])
            if k == len(limits) - 1:
                faces.append(ids[-1])
            for f in faces:
                border.update(int(v) for v in np.unique(f) if v)
            if prev_plane is not None:
                self._join(prev_plane, ids[0])
            prev_plane = ids[-1].copy()
            self._slabs.append((z0, z1, base, n))
            base += n
            del ids
        self.roots = {self.uf.find(x) for x in self.uf.parent}
        self.touches_border = {r: False for r in self.roots}
        for x in border:
            self.touches_border[self.uf.find(x)] = True
        self.bbox: dict[int, list[int]] = {}
        for lab, b in boxes.items():
            r = self.uf.find(lab)
            f = self.bbox.get(r)
            if f is None:
                self.bbox[r] = list(b)
            else:
                f[0], f[1] = min(f[0], b[0]), max(f[1], b[1])
                f[2], f[3] = min(f[2], b[2]), max(f[3], b[3])
                f[4], f[5] = min(f[4], b[4]), max(f[5], b[5])

    def _slab(self, z0: int, z1: int) -> np.ndarray:
        s = self.mask[z0:z1]
        return ~s if self.invert else s

    def _join(self, prev: np.ndarray, cur: np.ndarray) -> None:
        H, W = prev.shape
        pairs: set[tuple[int, int]] = set()
        for dy, dx in _interface_shifts(self.connectivity):
            a = prev[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
            b = cur[max(0, -dy):H + min(0, -dy), max(0, -dx):W + min(0, -dx)]
            m = (a > 0) & (b > 0)
            if m.any():
                pairs |= set(zip(a[m].tolist(), b[m].tolist()))
        for u, v in pairs:
            self.uf.union(int(u), int(v))

    @property
    def n_components(self) -> int:
        return len(self.roots)

    def relabel(self, value_of_root: Callable[[int], int], dtype=np.uint32
                ) -> np.ndarray:
        """Full-size array with ``value_of_root(root)`` written on every voxel of
        each component (0 on background)."""
        out = np.zeros(self.mask.shape, dtype)
        for z0, z1, base, n in self._slabs:
            lbl, _ = _label(self._slab(z0, z1), self.connectivity)
            lut = np.zeros(n + 1, dtype)
            for x in range(1, n + 1):
                lut[x] = value_of_root(self.uf.find(base + x))
            np.take(lut, lbl, out=out[z0:z1])
            del lbl
        return out


def count_components(mask: np.ndarray, connectivity: int,
                     max_voxels: float = 2.0e8) -> int:
    """Number of connected components of ``mask``."""
    return ComponentScan(mask, connectivity, max_voxels).n_components


def border_components(mask: np.ndarray, connectivity: int = 6,
                      max_voxels: float = 2.0e8) -> np.ndarray:
    """Boolean mask of the components of ``mask`` that touch any face of the array."""
    scan = ComponentScan(mask, connectivity, max_voxels)
    return scan.relabel(lambda r: scan.touches_border[r], dtype=bool)


def filter_min_extent(mask: np.ndarray, min_extent: int, connectivity: int = 26,
                      max_voxels: float = 2.0e8) -> tuple[np.ndarray, int, int]:
    """Keep the components whose smallest bounding-box side is ``>= min_extent``.

    Returns ``(kept_mask, n_components, n_kept)``. The bounding box of a component
    that crosses slab borders is its total extent, not the extent of each piece.
    """
    scan = ComponentScan(mask, connectivity, max_voxels, need_bbox=True)
    ok = {r for r, b in scan.bbox.items()
          if min(b[1] - b[0], b[3] - b[2], b[5] - b[4]) >= min_extent}
    kept = scan.relabel(lambda r: r in ok, dtype=bool)
    return kept, scan.n_components, len(ok)


def closed_pore_labels(bone: np.ndarray, connectivity: int = 26,
                       max_voxels: float = 2.0e8) -> tuple[np.ndarray, int]:
    """Label the closed pores of a binary bone volume.

    A closed pore is a background component that touches none of the six faces:
    void entirely surrounded by bone. Labelling the background once and discarding
    the border components isolates every closed pore at once. With 26-connectivity
    this reproduces CTAn's Po.N(cl) exactly.
    """
    scan = ComponentScan(bone, connectivity, max_voxels, invert=True)
    closed = sorted(r for r in scan.roots if not scan.touches_border[r])
    new = {r: i for i, r in enumerate(closed, 1)}
    dtype = np.uint16 if len(closed) <= np.iinfo(np.uint16).max else np.uint32
    return scan.relabel(lambda r: new.get(r, 0), dtype=dtype), len(closed)


# ---------------------------------------------------------------------- Euler
def euler_number(volume: np.ndarray, connectivity: int = 3,
                 max_voxels: float = 1.5e8) -> int:
    """3D Euler characteristic, identical to ``skimage.measure.euler_number``.

    skimage casts the volume to int64 and pads it (8 bytes/voxel). The algorithm is
    a histogram of 2x2x2 configurations, and a histogram is additive, so it is
    accumulated slab by slab in uint8. ``connectivity=3`` treats the foreground as
    26-connected (background 6-connected), as CTAn does.
    """
    from scipy import ndimage as ndi
    from skimage.measure._regionprops_utils import EULER_COEFS3D_26

    if volume.ndim != 3:
        raise ValueError("euler_number expects a 3D volume")
    if connectivity not in (1, 3):
        raise ValueError("connectivity must be 1 or 3")
    config = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                       [[0, 0, 0], [0, 16, 64], [0, 32, 128]]], dtype=np.uint8)
    coefs = EULER_COEFS3D_26[::-1] if connectivity == 1 else EULER_COEFS3D_26
    nz, ny, nx = volume.shape
    NZ = nz + 2
    hist = np.zeros(256, np.int64)

    def plane(k: int) -> np.ndarray:
        p = np.zeros((ny + 2, nx + 2), np.uint8)
        if 1 <= k <= nz:
            p[1:-1, 1:-1] = volume[k - 1] > 0
        return p

    for Z0, Z1 in slabs(NZ, volume.size, max_voxels):
        a, b = max(0, Z0 - 1), min(NZ, Z1 + 1)
        block = np.stack([plane(k) for k in range(a, b)])
        conv = ndi.convolve(block, config, mode="constant", cval=0)
        hist += np.bincount(conv[Z0 - a:Z1 - a].ravel(), minlength=256)
        del block, conv
    return int(0.125 * (coefs @ hist))
