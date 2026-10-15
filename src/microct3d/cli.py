"""Command line: ``microct3d analyze``, ``microct3d binarize``, ``microct3d phantom``."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def _write_rows(path: str, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_analyze(a: argparse.Namespace) -> int:
    from .config import Params
    from .io import read_binary_stack, read_binary_stack_from_zip
    from .pipeline import analyze_voi

    p = Params(voxel_um=a.voxel_um, canal_diameter_um=a.canal_um, guard_vox=a.guard,
               shrinkwrap_diameter_vox=a.shrinkwrap)
    targets = []
    for t in a.voi:
        if t.lower().endswith(".zip"):
            if not a.member:
                print("a .zip needs --member SAMPLE/VOI", file=sys.stderr)
                return 2
            targets.append((t, a.member))
        else:
            targets.append((t, None))
    rows = []
    for path, member in targets:
        name = member or os.path.relpath(path)
        print(f"=== {name}")
        bone = (read_binary_stack_from_zip(path, member) if member
                else read_binary_stack(path, voxel_um=a.voxel_um))
        r = {"voi": name}
        r.update(analyze_voi(bone, p, with_orientation=a.orientation))
        rows.append(r)
    _write_rows(a.out, rows)
    print(f"-> {a.out}")
    return 0


def cmd_binarize(a: argparse.Namespace) -> int:
    from .preprocessing import BinarizationParams, binarize_files
    files = sorted(glob.glob(os.path.join(a.filtered_dir, "*.tif*")))
    if not files:
        print("no TIFF slices found", file=sys.stderr)
        return 2
    thr = binarize_files(files, a.out_dir, BinarizationParams(voxel_um=a.voxel_um))
    print(f"Otsu threshold {thr}; {len(files)} slices -> {a.out_dir}")
    return 0


def cmd_phantom(a: argparse.Namespace) -> int:
    from .validation.phantoms import run_all
    for line in run_all():
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="microct3d", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("analyze", help="analyse binary VOIs (folders of slices or .zip)")
    s.add_argument("voi", nargs="+")
    s.add_argument("--member", help="SAMPLE/VOI folder inside a .zip")
    s.add_argument("--out", default="results.csv")
    s.add_argument("--voxel-um", type=float, default=5.0)
    s.add_argument("--canal-um", type=float, default=20.0)
    s.add_argument("--guard", type=int, default=3)
    s.add_argument("--shrinkwrap", type=int, default=30)
    s.add_argument("--orientation", action="store_true",
                   help="also classify axis segments as longitudinal / transverse")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("binarize", help="Otsu + despeckle on filtered 16-bit slices")
    s.add_argument("filtered_dir")
    s.add_argument("out_dir")
    s.add_argument("--voxel-um", type=float, default=5.0)
    s.set_defaults(func=cmd_binarize)

    s = sub.add_parser("phantom", help="run the synthetic validation phantoms")
    s.set_defaults(func=cmd_phantom)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
