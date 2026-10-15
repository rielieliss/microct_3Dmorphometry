"""From reconstructed 16-bit slices to the binary bone mask.

Three steps, identical for every sample:

1. Perona-Malik anisotropic diffusion, slice by slice, on images normalised to
   [0, 1]: 15 iterations, conductance kappa = 0.03, step lambda = 0.20 (stable below
   0.25), exponential conductance. Runs on the GPU through PyTorch when available.
2. Global Otsu threshold per sample, from the histogram of the filtered volume
   ("upper": bone = values >= threshold). One value per volume, no manual tuning.
3. 2D despeckle: remove white objects smaller than a disc of 20 µm radius and fill
   black holes smaller than a disc of 10 µm radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BinarizationParams:
    iterations: int = 15
    kappa: float = 0.03
    step: float = 0.20
    conductance: str = "exp"          # "exp": exp(-(g/k)^2); "quad": 1/(1+(g/k)^2)
    max_value: float = 65535.0
    despeckle_white_um: float = 20.0
    despeckle_black_um: float = 10.0
    histogram_slice_step: int = 10
    voxel_um: float = 5.0

    def disc_area(self, radius_um: float) -> int:
        r = radius_um / self.voxel_um
        return max(2, int(round(np.pi * r * r)))


def _conductance(d, kappa, kind, xp):
    if kind == "exp":
        return xp.exp(-(d / kappa) ** 2)
    return 1.0 / (1.0 + (d / kappa) ** 2)


def anisotropic_diffusion(slices: np.ndarray, p: BinarizationParams = BinarizationParams(),
                          device: str | None = None) -> np.ndarray:
    """Perona-Malik diffusion of a batch of 2D slices ``[b, y, x]`` (uint16).

    Uses PyTorch (CUDA if available) and falls back to NumPy. Returns uint16.
    """
    try:
        import torch
    except (ImportError, OSError):   # not installed, or installed with broken CUDA DLLs
        torch = None
    x = slices.astype(np.float32) / p.max_value
    if torch is not None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        with torch.no_grad():
            for _ in range(p.iterations):
                dn = torch.zeros_like(t); ds = torch.zeros_like(t)
                de = torch.zeros_like(t); dw = torch.zeros_like(t)
                dn[:, 1:, :] = t[:, :-1, :] - t[:, 1:, :]
                ds[:, :-1, :] = t[:, 1:, :] - t[:, :-1, :]
                de[:, :, :-1] = t[:, :, 1:] - t[:, :, :-1]
                dw[:, :, 1:] = t[:, :, :-1] - t[:, :, 1:]
                t = t + p.step * sum(_conductance(d, p.kappa, p.conductance, torch) * d
                                     for d in (dn, ds, de, dw))
        out = (t.clamp(0, 1) * p.max_value).round().cpu().numpy()
    else:
        u = x.copy()
        for _ in range(p.iterations):
            dn = np.zeros_like(u); ds = np.zeros_like(u)
            de = np.zeros_like(u); dw = np.zeros_like(u)
            dn[:, 1:, :] = u[:, :-1, :] - u[:, 1:, :]
            ds[:, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
            de[:, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
            dw[:, :, 1:] = u[:, :, :-1] - u[:, :, 1:]
            u = u + p.step * sum(_conductance(d, p.kappa, p.conductance, np) * d
                                 for d in (dn, ds, de, dw))
        out = np.round(np.clip(u, 0, 1) * p.max_value)
    return out.astype(slices.dtype)


def otsu_from_histogram(counts: np.ndarray) -> int:
    """Otsu threshold (index of the histogram bin) maximising between-class variance."""
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0
    centers = np.arange(counts.size, dtype=np.float64)
    w = np.cumsum(counts)
    mu = np.cumsum(counts * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_b = mu / w
        m_f = (mu[-1] - mu) / (total - w)
        sigma = w * (total - w) * (m_b - m_f) ** 2
    sigma[~np.isfinite(sigma)] = 0
    return int(np.argmax(sigma))


def global_otsu(slices: Iterable[np.ndarray], n_levels: int = 65536) -> int:
    """One Otsu threshold for a whole volume, from the summed slice histograms."""
    hist = np.zeros(n_levels, dtype=np.int64)
    for a in slices:
        hist += np.bincount(np.asarray(a).ravel(), minlength=n_levels)
    return otsu_from_histogram(hist)


def _remove_small(mask: np.ndarray, size: int, holes: bool) -> np.ndarray:
    from skimage import morphology as mo
    f = mo.remove_small_holes if holes else mo.remove_small_objects
    try:  # scikit-image >= 0.26: remove objects of `max_size` pixels or fewer
        return f(mask, max_size=size)
    except TypeError:  # older versions: remove objects smaller than `min_size`
        kw = "area_threshold" if holes else "min_size"
        return f(mask, **{kw: size + 1})


def binarize_slice(filtered: np.ndarray, threshold: int,
                   p: BinarizationParams = BinarizationParams()) -> np.ndarray:
    """Threshold one filtered slice and despeckle it (bone = True)."""
    bw = filtered >= threshold
    bw = _remove_small(bw, p.disc_area(p.despeckle_white_um), holes=False)
    bw = _remove_small(bw, p.disc_area(p.despeckle_black_um), holes=True)
    return bw


def binarize_files(in_files: Sequence[str], out_dir: str,
                   p: BinarizationParams = BinarizationParams()) -> int:
    """Binarize a folder of FILTERED 16-bit TIFF slices into 0/255 TIFFs.

    Returns the Otsu threshold that was used.
    """
    import os

    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    thr = global_otsu(tifffile.imread(f) for f in in_files[::p.histogram_slice_step])
    for f in in_files:
        bw = binarize_slice(tifffile.imread(f), thr, p)
        tifffile.imwrite(os.path.join(out_dir, os.path.basename(f)),
                         bw.astype(np.uint8) * 255)
    return thr
