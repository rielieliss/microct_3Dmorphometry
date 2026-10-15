"""Read CTAn (Bruker CT-Analyser) "3D analysis" text reports for comparison."""
from __future__ import annotations

import re

# CTAn row label -> key used by microct3d.morphometry
CTAN_KEYS = {
    "Tissue volume,TV": "TV_mm3",
    "Bone volume,BV": "BV_mm3",
    "Percent bone volume,BV/TV": "BV_TV_pct",
    "Bone surface,BS": "BS_mm2",
    "Bone surface / volume ratio,BS/BV": "BS_BV_per_mm",
    "Bone surface density,BS/TV": "BS_TV_per_mm",
    "Number of objects,Obj.N": "Obj_N",
    "Number of closed pores,Po.N(cl)": "PoN_cl",
    "Volume of closed pores,Po.V(cl)": "PoV_cl_mm3",
    "Surface of closed pores,Po.S(cl)": "PoS_cl_mm2",
    "Closed porosity (percent),Po(cl)": "Po_cl_pct",
    "Open porosity (percent),Po(op)": "Po_op_subtraction_pct",
    "Total porosity (percent),Po(tot)": "Po_tot_subtraction_pct",
    "Euler number,Eu.N": "EuN",
    "Connectivity,Conn": "Conn",
    "Connectivity density,Conn.Dn": "ConnDn_per_mm3",
    "Tissue surface,TS": "TS_mm2",
    "Fractal dimension,FD": "FD",
}


def read_report(path: str) -> list[dict]:
    """All analysis blocks of a CTAn report, each as ``{key: value}`` plus
    ``n_layers``. A report can hold several tasks; each is returned separately."""
    text = open(path, encoding="latin-1").read()
    blocks = []
    for chunk in re.split(r"<<<<< Begin of task", text):
        m = re.search(r"^Number of layers,,(\d+)", chunk, re.M)
        if not m:
            continue
        vals = {"n_layers": int(m.group(1))}
        for label, key in CTAN_KEYS.items():
            mm = re.search("^" + re.escape(label) + r",(-?[\d.]+)", chunk, re.M)
            if mm:
                vals[key] = float(mm.group(1))
        blocks.append(vals)
    return blocks


def matching_block(blocks: list[dict], n_layers: int, bone_volume_mm3: float,
                   tolerance_pct: float = 0.5) -> dict | None:
    """The report block that describes the SAME VOI: same number of slices and BV
    within ``tolerance_pct``. The slice range alone does not guarantee the same ROI
    (one VOI with the right range had BV +92.9 % because it had been redrawn)."""
    ok = [b for b in blocks if b.get("n_layers") == n_layers and "BV_mm3" in b
          and abs(b["BV_mm3"] - bone_volume_mm3) / b["BV_mm3"] * 100 <= tolerance_pct]
    return ok[-1] if ok else None
