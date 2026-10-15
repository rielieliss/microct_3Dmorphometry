"""Analyse every VOI under a root folder and write one CSV row per VOI.

    python examples/analyze_folder.py ROOT results.csv

A VOI is any sub-folder whose name starts with "VOI" and contains binary slices.
"""
import csv
import os
import sys

from microct3d import Params, analyze_voi
from microct3d.io import list_slices, read_binary_stack


def main(root: str, out: str) -> None:
    rows = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in sorted(dirnames):
            folder = os.path.join(dirpath, d)
            if not d.upper().startswith("VOI") or not list_slices(folder):
                continue
            print(f"=== {os.path.relpath(folder, root)}")
            r = {"voi": os.path.relpath(folder, root)}
            r.update(analyze_voi(read_binary_stack(folder), Params()))
            rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} VOIs -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
