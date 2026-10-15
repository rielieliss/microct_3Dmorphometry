"""Every computation checked against an INDEPENDENT reference: the textbook definition
implemented naively, or another library (scikit-image, SciPy)."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from microct3d import _blocks as B
from microct3d import canals, closed_pores, envelope, geometry, preprocessing
from microct3d.skeleton import axis_segments
from microct3d.validation.agreement import bland_altman, lin_ccc


def _ball(r):
    k = int(np.ceil(r))
    g = np.indices((2 * k + 1,) * 3) - k
    return (g ** 2).sum(0) <= r * r


def _blobs(seed=0, shape=(36, 40, 40)):
    rng = np.random.default_rng(seed)
    v = ndi.gaussian_filter(rng.random(shape), 2.0) > 0.52
    return v


# --------------------------------------------------------------- binarization
def test_otsu_matches_scikit_image():
    from skimage.filters import threshold_otsu
    rng = np.random.default_rng(1)
    img = np.concatenate([rng.normal(9000, 800, 5000), rng.normal(30000, 2500, 3000)])
    img = np.clip(img, 0, 65535).astype(np.uint16)
    ours = preprocessing.otsu_from_histogram(np.bincount(img, minlength=65536))
    ref = threshold_otsu(hist=np.bincount(img, minlength=65536))
    assert abs(ours - ref) <= 1


def test_diffusion_preserves_constant_image_and_range():
    img = np.full((2, 20, 20), 12345, np.uint16)
    out = preprocessing.anisotropic_diffusion(img)
    assert np.array_equal(out, img)          # zero-flux boundaries: nothing moves
    noisy = (np.random.default_rng(2).random((1, 30, 30)) * 65535).astype(np.uint16)
    out = preprocessing.anisotropic_diffusion(noisy)
    assert out.dtype == np.uint16 and out.std() < noisy.std()


# ------------------------------------------------------------ morphology
@pytest.mark.parametrize("r", [1.5, 2.0, 3.0])
def test_opening_equals_textbook_opening_by_a_ball(r):
    m = _blobs(3)
    vols, op = canals.openings(m, [r], r)
    ref = ndi.binary_dilation(ndi.binary_erosion(m, _ball(r), border_value=1), _ball(r))
    # erosion by a ball of radius r == centres whose EDT exceeds r (ball fits strictly)
    core = ndi.distance_transform_edt(m) > r
    ref2 = ndi.binary_dilation(core, _ball(r)) & m
    assert np.array_equal(op, ref2)
    assert vols[r] == int(ref2.sum())
    assert (op & ~m).sum() == 0 and (ref & ~ref2 & m).sum() >= 0


def test_shrinkwrap_equals_closing_from_outside():
    bone = np.zeros((30, 50, 50), bool)
    bone[:, 10:40, 10:40] = True
    bone[:, 23:27, 10:22] = False            # groove narrower than D
    bone[:, 30:38, 30:40] = False            # notch wider than D (follows it)
    D = 6
    tissue = envelope.shrinkwrap(bone, D)
    # reference: pad z with empty planes, closing of the bone seen from outside
    r, pad = D / 2.0, int(np.ceil(D / 2.0)) + 1
    p = np.zeros((30 + 2 * pad, 50, 50), bool)
    p[pad:-pad] = bone
    lbl, _ = ndi.label(~p)
    border = np.unique(np.concatenate([lbl[0].ravel(), lbl[-1].ravel(), lbl[:, 0].ravel(),
                                       lbl[:, -1].ravel(), lbl[:, :, 0].ravel(),
                                       lbl[:, :, -1].ravel()]))
    outside = np.isin(lbl, border[border > 0])
    er = ndi.distance_transform_edt(outside) > r
    lbl2, _ = ndi.label(er)
    b2 = np.unique(np.concatenate([lbl2[0].ravel(), lbl2[-1].ravel(), lbl2[:, 0].ravel(),
                                   lbl2[:, -1].ravel(), lbl2[:, :, 0].ravel(),
                                   lbl2[:, :, -1].ravel()]))
    er = np.isin(lbl2, b2[b2 > 0])
    ref = ndi.distance_transform_edt(~er) > r
    assert np.array_equal(tissue, ref[pad:-pad])
    assert tissue[10, 25, 15]                # groove bridged
    assert not tissue[10, 34, 35]            # wide notch followed


# ---------------------------------------------------------- closed pores, mesh
def test_closed_pores_equal_scipy_reference():
    bone = ~_blobs(4, (30, 30, 30))
    lbl, n = B.closed_pore_labels(bone, 26)
    ref, nr = ndi.label(~bone, np.ones((3, 3, 3)))
    faces = np.unique(np.concatenate([ref[0].ravel(), ref[-1].ravel(), ref[:, 0].ravel(),
                                      ref[:, -1].ravel(), ref[:, :, 0].ravel(),
                                      ref[:, :, -1].ravel()]))
    closed = [i for i in range(1, nr + 1) if i not in set(faces)]
    assert n == len(closed)
    assert (lbl > 0).sum() == np.isin(ref, closed).sum()


def test_mesh_measures_match_scikit_image_and_geometry():
    from skimage import measure
    z, y, x = np.indices((50, 50, 50)) - 25
    sphere = x * x + y * y + z * z <= 18 ** 2
    v, f = geometry.mesh(sphere, 1.0)
    assert geometry.mesh_area(v, f) == pytest.approx(measure.mesh_surface_area(v, f))
    assert geometry.mesh_volume(v, f) == pytest.approx(4 / 3 * np.pi * 18 ** 3, rel=0.01)
    # slab-wise area == whole-mesh area (up to float32 rounding of the vertices)
    assert geometry.surface_area(sphere, 1.0, max_voxels=20000) == pytest.approx(
        geometry.mesh_area(v, f), rel=1e-6)


def test_pore_descriptors_on_a_sphere_and_a_tube():
    bone = np.ones((60, 60, 60), bool)
    z, y, x = np.indices(bone.shape)
    bone[(x - 20) ** 2 + (y - 20) ** 2 + (z - 30) ** 2 <= 8 ** 2] = False
    bone[(x - 45) ** 2 + (y - 45) ** 2 <= 2 ** 2] = False
    bone[:5] = bone[-5:] = True             # close the tube: a closed elongated pore
    pores, _ = closed_pores.measure_closed_pores(bone, 0.005, shape_min_vox=20)
    sph = max(pores, key=lambda p: p.sphericity)
    tube = min(pores, key=lambda p: p.elongation)
    assert sph.esd_um == pytest.approx(16 * 5, rel=0.03)      # ESD of a sphere = 2r
    assert 0.85 < sph.sphericity < 0.95                        # mesh ceiling ~0.9
    assert sph.elongation == pytest.approx(1, abs=0.05)
    assert tube.elongation < 0.15


# ----------------------------------------------------------------- box counting
def test_box_counts_equal_naive_definition():
    v = _blobs(5, (40, 44, 48))
    counts = geometry.box_counts(v, [2, 3, 5, 8], max_voxels=20000)
    er = ndi.binary_erosion(v, ndi.generate_binary_structure(3, 1), border_value=0)
    surf = v & ~er
    for s in (2, 3, 5, 8):
        p = np.argwhere(surf) // s
        assert counts[s] == len({tuple(q) for q in p})
    fd, r2, _ = geometry.fractal_dimension({1: 1000, 2: 250, 4: 62.5, 8: 15.625},
                                           [1, 2, 4, 8])
    assert fd == pytest.approx(2.0) and r2 == pytest.approx(1.0)


# --------------------------------------------------------------------- skeleton
def test_axis_length_and_calibre_on_a_straight_cylinder():
    z, y, x = np.indices((80, 40, 40))
    cyl = (y - 20) ** 2 + (x - 20) ** 2 <= 5 ** 2
    seg = axis_segments(cyl, cyl, voxel_um=5.0)
    # thinning shortens a skeleton by about one radius at each free end (here the
    # cylinder is cut by the top and bottom faces): 80 slices -> ~70 voxels of axis
    assert 80 * 5 - 2 * 5 * 5 - 10 <= seg["axis_length_um"] <= 80 * 5
    assert np.median(seg["calibre_um"]) == pytest.approx(2 * 5 * 5, rel=0.12)


# ------------------------------------------------------------------- statistics
def test_lin_ccc_and_bland_altman_by_definition():
    rng = np.random.default_rng(6)
    a = rng.normal(10, 2, 40)
    b = a * 1.02 + rng.normal(0, 0.1, 40)
    s = np.cov(a, b, ddof=0)
    ref = 2 * s[0, 1] / (s[0, 0] + s[1, 1] + (a.mean() - b.mean()) ** 2)
    assert lin_ccc(a, b) == pytest.approx(ref)
    assert lin_ccc(a, a) == pytest.approx(1.0)
    ba = bland_altman(a, b)
    d = (b - a) / ((a + b) / 2) * 100
    assert ba["bias_pct"] == pytest.approx(d.mean())
    assert ba["loa_high_pct"] - ba["loa_low_pct"] == pytest.approx(2 * 1.96 * d.std(ddof=1))
