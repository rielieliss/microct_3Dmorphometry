"""Reading image stacks and Scanco ISQ files, writing meshes."""
from __future__ import annotations

import glob
import io as _io
import os
import re
import struct
import zipfile
from dataclasses import dataclass

import numpy as np

SLICE_EXTENSIONS = (".bmp", ".tif", ".tiff", ".png")
# Files written next to the slices by earlier analyses; never read them as slices.
_NOT_A_SLICE = re.compile(r"(_poros|_preview|_labels|_fig[AB]_|_morfometria3d)", re.I)


def natural_key(s: str) -> list:
    """Sort key that orders ``slice2`` before ``slice10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def slice_index(path: str) -> int | None:
    """z index from a CTAn-style file name ending in four digits (``voi_0127.tif``)."""
    m = re.search(r"(\d{4})\.[^.]+$", os.path.basename(path))
    return int(m.group(1)) if m else None


def list_slices(folder: str) -> list[str]:
    """Slice files of a VOI folder, in natural order."""
    files = [f for f in glob.glob(os.path.join(folder, "*"))
             if f.lower().endswith(SLICE_EXTENSIONS)
             and not _NOT_A_SLICE.search(os.path.basename(f))]
    return sorted(files, key=natural_key)


def _stack(readers) -> np.ndarray:
    from PIL import Image
    first = np.asarray(Image.open(readers[0]()))
    vol = np.empty((len(readers),) + first.shape, dtype=bool)
    vol[0] = first > 0
    for k, read in enumerate(readers[1:], 1):
        a = np.asarray(Image.open(read()))
        if a.shape != first.shape:
            raise ValueError(f"slice {k} has shape {a.shape}, expected {first.shape}")
        vol[k] = a > 0
    return vol


def read_binary_stack(folder: str, z_range_mm: tuple[float, float] | None = None,
                      voxel_um: float = 5.0) -> np.ndarray:
    """Read a folder of binary slices as a boolean volume ``[z, y, x]`` (True = bone).

    ``z_range_mm`` keeps only the slices between two heights, as CTAn reports them
    ("Lower/Upper vertical position"), so a VOI can be reproduced exactly when the
    exported stack is taller than the analysed region.
    """
    files = list_slices(folder)
    if z_range_mm is not None:
        lo, hi = (int(round(v / (voxel_um / 1000.0))) for v in z_range_mm)
        missing = [f for f in files if slice_index(f) is None]
        if missing:
            raise ValueError(f"cannot read the z index of {os.path.basename(missing[0])}")
        files = [f for f in files if lo <= slice_index(f) <= hi]
    if not files:
        raise FileNotFoundError(f"no slices in {folder}")
    return _stack([lambda f=f: f for f in files])


def read_binary_stack_from_zip(zip_path: str, member_folder: str) -> np.ndarray:
    """Same as :func:`read_binary_stack`, for a folder stored inside a .zip archive
    (``member_folder`` like ``"SAMPLE/VOI"``), without extracting anything."""
    zf = zipfile.ZipFile(zip_path)
    prefix = member_folder.rstrip("/") + "/"
    names = [n for n in zf.namelist()
             if n.startswith(prefix) and "/" not in n[len(prefix):]
             and n.lower().endswith(SLICE_EXTENSIONS)
             and not _NOT_A_SLICE.search(os.path.basename(n))]
    if not names:
        raise FileNotFoundError(f"no slices under {member_folder} in {zip_path}")
    names.sort(key=natural_key)
    return _stack([lambda n=n: _io.BytesIO(zf.read(n)) for n in names])


# ------------------------------------------------------------------ Scanco ISQ
@dataclass
class IsqHeader:
    nx: int
    ny: int
    nz: int
    nz_header: int
    data_offset: int
    voxel_um_header: float
    mu_scaling: int
    name: str


def read_isq_header(path: str) -> IsqHeader:
    """Header of a Scanco ``.ISQ`` file.

    The number of slices is taken from the FILE SIZE, not from the header: one scan in
    the study declared 2175 slices but contained data for 1377.
    """
    with open(path, "rb") as f:
        h = f.read(512)

    def i32(offset: int) -> int:
        return struct.unpack_from("<i", h, offset)[0]

    nx, ny, nz = i32(44), i32(48), i32(52)
    offset = (i32(508) + 1) * 512
    nz_real = (os.path.getsize(path) - offset) // (nx * ny * 2)
    return IsqHeader(nx=nx, ny=ny, nz=int(nz_real), nz_header=nz, data_offset=offset,
                     voxel_um_header=i32(56) / nx, mu_scaling=i32(88),
                     name=h[128:168].split(b"\0")[0].decode(errors="replace").strip())


def open_isq(path: str) -> tuple[np.memmap, IsqHeader]:
    """Memory-map the ISQ volume as signed 16-bit ``[z, y, x]`` without loading it."""
    hd = read_isq_header(path)
    vol = np.memmap(path, dtype="<i2", mode="r", offset=hd.data_offset,
                    shape=(hd.nz, hd.ny, hd.nx))
    return vol, hd


def isq_slice_uint16(vol: np.ndarray, k: int) -> np.ndarray:
    """Slice ``k`` as unsigned 16-bit (value + 32768): exact and reversible."""
    return (vol[k].astype(np.int32) + 32768).astype(np.uint16)


# ---------------------------------------------------------------------- meshes
def write_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Binary STL from vertices and triangular faces."""
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)),
                                               ("a", "<u2")]))
    rec["n"] = n / ln
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        f.write(rec.tobytes())
