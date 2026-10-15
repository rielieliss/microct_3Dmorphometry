"""Block-wise computations must equal the single-pass ones exactly."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.measure import euler_number as sk_euler

from microct3d import _blocks as B


def _random_volume(seed=0, shape=(40, 30, 30), p=0.45):
    rng = np.random.default_rng(seed)
    v = rng.random(shape) < p
    return ndi.binary_opening(v)


@pytest.mark.parametrize("conn", [6, 26])
@pytest.mark.parametrize("max_voxels", [1e9, 9000, 4000, 2000])
def test_count_components(conn, max_voxels):
    v = _random_volume(1)
    rank = 1 if conn == 6 else 3
    _, n = ndi.label(v, ndi.generate_binary_structure(3, rank))
    assert B.count_components(v, conn, max_voxels) == n


@pytest.mark.parametrize("max_voxels", [1e9, 9000, 3000])
def test_border_components(max_voxels):
    v = _random_volume(2)
    lbl, _ = ndi.label(v)
    faces = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(),
                                      lbl[:, -1].ravel(), lbl[:, :, 0].ravel(),
                                      lbl[:, :, -1].ravel()]))
    expected = np.isin(lbl, faces[faces > 0])
    assert np.array_equal(B.border_components(v, 6, max_voxels), expected)


@pytest.mark.parametrize("max_voxels", [1e9, 5000])
def test_filter_min_extent(max_voxels):
    v = _random_volume(3)
    kept, n, n_ok = B.filter_min_extent(v, 3, 26, max_voxels)
    lbl, n0 = ndi.label(v, np.ones((3, 3, 3)))
    assert n == n0
    ok = [i for i, sl in enumerate(ndi.find_objects(lbl), 1)
          if min(s.stop - s.start for s in sl) >= 3]
    assert n_ok == len(ok)
    assert np.array_equal(kept, np.isin(lbl, ok))


@pytest.mark.parametrize("max_voxels", [1e9, 6000, 2500])
def test_closed_pores(max_voxels):
    bone = np.ones((30, 30, 30), bool)
    bone[5:8, 5:8, 5:8] = False          # closed pore 1
    bone[15:18, 20:23, 10:13] = False    # closed pore 2
    bone[22, 3, 3] = False               # closed pore 3 (one voxel)
    bone[:, 14, 14] = False              # open canal through the whole height
    lbl, n = B.closed_pore_labels(bone, 26, max_voxels)
    assert n == 3
    assert (lbl > 0).sum() == 27 + 27 + 1


@pytest.mark.parametrize("max_voxels", [1e9, 8000, 3000, 1500])
def test_euler_number(max_voxels):
    v = _random_volume(4)
    assert B.euler_number(v, 3, max_voxels) == sk_euler(v, connectivity=3)


@pytest.mark.parametrize("max_voxels", [1e9, 7000])
def test_edt_greater(max_voxels):
    v = _random_volume(5, p=0.7)
    assert np.array_equal(B.edt_greater(v, 1.5, max_voxels),
                          ndi.distance_transform_edt(v) > 1.5)
