"""Known topology and geometry."""
import numpy as np

from microct3d.closed_pores import measure_closed_pores
from microct3d.envelope import shrinkwrap
from microct3d.morphometry import ctan_morphometry


def test_cube_with_cavity():
    bone = np.zeros((40, 40, 40), bool)
    bone[5:35, 5:35, 5:35] = True
    bone[15:20, 15:20, 15:20] = False                 # one closed cavity
    tissue = np.zeros_like(bone)
    tissue[5:35, 5:35, 5:35] = True
    pores, _ = measure_closed_pores(bone, 0.005)
    r = ctan_morphometry(bone, tissue, pores, 0.005, fd_sizes=(2, 3, 4))
    assert r["Obj_N"] == 1
    assert r["PoN_cl"] == 1
    assert r["EuN"] == 2          # one object + one cavity, no tunnel
    assert r["Conn"] == 0
    assert abs(r["BV_TV_pct"] - 100 * (30 ** 3 - 125) / 30 ** 3) < 1e-9


def test_tunnel_counts_as_connectivity():
    bone = np.zeros((30, 30, 30), bool)
    bone[5:25, 5:25, 5:25] = True
    bone[5:25, 14:16, 14:16] = False                  # a canal right through
    pores, _ = measure_closed_pores(bone, 0.005)
    r = ctan_morphometry(bone, bone, pores, 0.005, fd_sizes=(2, 3, 4))
    assert r["EuN"] == 0 and r["Conn"] == 1


def test_shrinkwrap_bridges_canal_mouths_and_keeps_z_open():
    bone = np.zeros((60, 80, 80), bool)
    bone[:, 20:60, 20:60] = True
    bone[:, 38:42, 20:30] = False      # a 4-voxel-wide groove open on one side
    tissue = shrinkwrap(bone, diameter_vox=10)
    assert tissue[5:-5, 39, 25].all()  # the groove is inside the envelope (away from the z ends)
    assert tissue[0, 30, 30] and tissue[-1, 30, 30]   # z ends are not eroded
    assert not tissue[:, 10, 10].any()
