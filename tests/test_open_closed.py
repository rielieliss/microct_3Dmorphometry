"""Open/closed classification of canal objects (Ca.V/TV, Ca.N/TV).

An object is open when it touches the outside of the tissue (6-adjacency) or a face
of the volume, including contact along z inside one slab.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from microct3d import canals


def _brute_force(canal, tissue):
    lbl, n = ndi.label(canal, ndi.generate_binary_structure(3, 3))
    near = ndi.binary_dilation(~tissue, ndi.generate_binary_structure(3, 1))
    faces = np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(),
                            lbl[:, -1].ravel(), lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()])
    hit = set(np.unique(lbl[near]).tolist()) | set(np.unique(faces).tolist())
    vol = np.bincount(lbl.ravel(), minlength=n + 1)
    return (sorted(int(vol[i]) for i in range(1, n + 1) if i in hit),
            sorted(int(vol[i]) for i in range(1, n + 1) if i not in hit))


def test_contact_along_z_inside_a_slab_is_open():
    shape = (20, 20, 20)
    tissue = np.ones(shape, bool)
    tissue[0:5, 8:13, 8:13] = False          # outside, only ABOVE the first object
    canal = np.zeros(shape, bool)
    canal[5:9, 9:12, 9:12] = True            # touches the outside only along z
    canal[12:16, 2:5, 2:5] = True            # surrounded by bone
    opened, closed = canals.open_and_closed(canal, tissue, max_voxels=1e9)
    assert opened == [36] and closed == [36]


@pytest.mark.parametrize("max_voxels", [1e9, 12000, 5000, 2500])
def test_matches_brute_force(max_voxels):
    rng = np.random.default_rng(7)
    shape = (40, 30, 30)
    tissue = ~ndi.binary_dilation(rng.random(shape) < 0.002, iterations=2)
    canal = ndi.binary_opening(rng.random(shape) < 0.35) & tissue
    got = canals.open_and_closed(canal, tissue, max_voxels=max_voxels)
    assert (sorted(got[0]), sorted(got[1])) == _brute_force(canal, tissue)
