# microct3d — 3D morphometry of cortical bone porosity

[![tests](https://github.com/rielieliss/microct_3Dmorphometry/actions/workflows/tests.yml/badge.svg)](https://github.com/rielieliss/microct_3Dmorphometry/actions/workflows/tests.yml)

`microct3d` measures the porosity of cortical bone in binary micro-CT volumes. It
does two things:

1. **Reproduces CTAn's 3D analysis** (Bruker CT-Analyser), so the classical
   parameters stay comparable with the software and with the literature.
2. **Adds what the commercial software does not measure**: it separates the canal
   network from closed pores with a local size criterion, and it measures canal
   calibre (inscribed sphere and area-equivalent diameter), pore space by size
   band, the canal axis with true segment lengths, the wall of enlarged spaces, and
   cortical thickness.

The defaults assume 5 µm isotropic voxels; every value is a parameter. Everything is
block-wise along z, so VOIs of several billion voxels run on a 16 GB machine.

> *Resumo em português* — ver [abaixo](#resumo-em-português).

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The method in one page](#the-method-in-one-page)
- [Output parameters](#output-parameters)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Citing](#citing)

## Installation

```bash
git clone https://github.com/rielieliss/microct_3Dmorphometry.git
cd microct_3Dmorphometry
pip install -e .            # add ".[gpu]" for GPU diffusion filtering, ".[dev]" for tests
```

Python ≥ 3.10. Depends on NumPy, SciPy, scikit-image, `connected-components-3d`,
`edt`, `tifffile` and Pillow. PyTorch is only needed to run the diffusion filter on
a GPU (a missing or broken installation falls back to NumPy).

**Memory.** Peak use is about 3 bytes per voxel of the cropped VOI (at most three
boolean volumes, or the bone plus 16-bit closed-pore labels).

## Quick start

From the command line:

```bash
# a folder of binary slices (bone = white), one VOI
microct3d analyze "path/to/SAMPLE/VOI" --out results.csv

# a VOI stored inside a .zip archive, read without extracting
microct3d analyze SAMPLE.zip --member "SAMPLE/VOI" --out results.csv

# binarize filtered 16-bit slices (global Otsu + despeckle)
microct3d binarize path/to/filtered_slices path/to/binary_slices

# run the synthetic phantoms
microct3d phantom
```

From Python:

```python
from microct3d import Params, analyze_voi
from microct3d.io import read_binary_stack

bone = read_binary_stack("path/to/SAMPLE/VOI")          # bool [z, y, x], True = bone
results = analyze_voi(bone, Params(voxel_um=5.0))
print(results["PoV_TV_pct"], results["CaN_per_mm3"], results["CtTh_mm"])
```

`Params` holds every number that defines the method
([`config.py`](src/microct3d/config.py)).

## The method in one page

```
binary VOI (bone = True)
  ├─ cortical thickness (2D, perpendicular lines) ................ Ct.Th
  ├─ ENVELOPE: 3D shrink-wrap, D = 30 voxels, z axis open ........ TV
  ├─ closed pores (CTAn definition, 26-connectivity) ............. Block B
  ├─ CTAn-equivalent morphometry ................................. Block A
  └─ GUARD ZONE (3 voxels) → pore space → LAMELLA FILTER (3 voxels)
       ├─ CANAL CRITERION: a 20 µm sphere fits ................... Block C
       │    ├─ open / closed canal objects ....................... Ca.V/TV, Ca.N/TV, Po.V/TV
       │    └─ 3D axis of the canal space ........................ calibre, axis fractions, D_eq
       └─ local-thickness bands, wall re-entrance ................ V(20–70)/TV, V(70–100)/TV, V(>100)/TV
```

Key choices:

- **Envelope.** CTAn's "shrink-wrap (3D), stretch over holes" with a 30-voxel
  (150 µm) sphere. The z axis is treated as open, because the VOI is a cut and the
  bone continues above and below it.
- **Guard zone.** The 3 voxels just inside the envelope are excluded, so the gap
  between envelope and bone is not counted as pore.
- **Canal criterion.** A point of pore space is canal if a sphere of 20 µm diameter
  fits there (morphological opening); 20 µm is 4 voxels at 5 µm. The diameter is a
  parameter, and extra diameters can be recorded for sensitivity analysis.
- **One denominator.** All porosities are divided by the tissue volume with guard,
  so total porosity is the **sum** of canal and closed porosity, never TV − BV.
- **Calibre.** Inscribed sphere (Hildebrand & Rüegsegger), i.e. the smallest width
  of the lumen, plus the area-equivalent diameter `D_eq = 2·sqrt(V/(π·L))` for
  comparison with studies that measure contour area.

## Output parameters

| key | parameter | notes |
|---|---|---|
| `TV_mm3`, `BV_mm3`, `BV_TV_pct` | tissue volume, bone volume, BV/TV | CTAn definitions (TV without guard) |
| `BS_mm2`, `BS_BV_per_mm`, `BS_TV_per_mm` | bone surface and ratios | include the external and cut surfaces |
| `Obj_N` | number of bone objects | 6-connectivity |
| `PoN_cl`, `PoV_cl_mm3`, `PoS_cl_mm2`, `Po_cl_pct` | closed pores | Po(cl) = Po.V(cl) / (BV + Po.V(cl)) |
| `EuN`, `EuN_TV_per_mm3` | Euler number and its density | the density is size-independent |
| `Conn`, `ConnDn_per_mm3` | connectivity and its density | CTAn convention |
| `TS_mm2`, `FD` | tissue surface, fractal dimension | FD fitted on 12–100-voxel boxes |
| `CtTh_mm` | cortical thickness | mean of the thicker half of each slice |
| `ESD_cl_mean_um`, `sphericity_cl_mean`, `elongation_cl_mean`, `flatness_cl_mean` | closed-pore size and shape | shape only for pores ≥ 100 voxels |
| `CaV_TV_pct`, `CaN_per_mm3` | canal volume fraction, canal objects per volume | not CTAn's open porosity |
| `PoV_TV_pct`, `PoN_per_mm3` | total porosity, total porous objects | canal + closed, same denominator |
| `V_20_70um_TV_pct`, `V_70_100um_TV_pct`, `V_above_100um_TV_pct` | pore volume by local thickness | the three sum to `PoV_TV_pct` |
| `wall_reentrance_depth_um` | mean re-entrance depth of spaces > 100 µm | exploratory |
| `axis_20_70_pct`, `axis_70_100_pct`, `axis_above_100_pct` | canal-axis length by calibre | denominator = canal axis only |
| `CaDm_length_weighted_um` | canal calibre (inscribed sphere) | length-weighted |
| `Deq_canals_um` | area-equivalent diameter of 20–100 µm canals | upper bound |
| `Po_tot_subtraction_pct`, `Po_op_subtraction_pct` | CTAn's Po(tot), Po(op) | by subtraction; for comparison only |

The full dictionary is in [docs/parameters.md](docs/parameters.md).

## Tests

**Against independent implementations** (`tests/test_reference.py`): every
computation is re-done a second way and must agree.

| computation | reference |
|---|---|
| Otsu threshold | `skimage.filters.threshold_otsu` |
| Euler number (block-wise) | `skimage.measure.euler_number` |
| opening by a ball (canal criterion) | textbook definition with an explicit ball element |
| shrink-wrap envelope | closing from the outside, written directly with SciPy |
| closed pores | `scipy.ndimage.label`, border components removed by hand |
| mesh area / volume | `skimage.measure.mesh_surface_area`; analytic sphere volume |
| ESD, sphericity, elongation | digital sphere and tube of known size |
| box counting | naive enumeration of occupied boxes |
| Lin's CCC, Bland–Altman | formulas written from the definition |

The full suite (44 tests) also checks that every block-wise computation equals the
single-pass one, known topologies (cavity, tunnel), the open/closed classification
of canal objects against a brute-force reference, and the synthetic phantoms
(`microct3d phantom`: cylinders of known calibre, a slit, lacuna-sized spheres and
oblique and elliptic canals). `microct3d.validation` also reads CTAn 3D-analysis
reports and computes Bland–Altman and Lin's CCC, to compare with CTAn on your own
data.

## Repository layout

```
src/microct3d/
  config.py          every parameter, with its rationale
  io.py              binary stacks (folders or .zip), Scanco ISQ, STL
  preprocessing.py   anisotropic diffusion, global Otsu, despeckle
  envelope.py        crop, 3D shrink-wrap, guard zone
  _blocks.py         block-wise EDT, connected components, Euler number
  geometry.py        meshes, surface area, box counting
  closed_pores.py    Block B
  morphometry.py     Block A (CTAn-equivalent)
  canals.py          Block C: lamella filter, canal criterion, size bands
  skeleton.py        canal axis: segments, MST length, calibre, D_eq, orientation
  thickness.py       cortical thickness
  pipeline.py        one pass over one VOI
  cli.py             command line
  validation/        phantoms, CTAn report reader, Bland–Altman / Lin's CCC
tests/               pytest suite
docs/                parameter dictionary
```

This repository contains code only: no images, data or results.

## Citing

See [CITATION.cff](CITATION.cff). Please also cite the libraries the method rests on:
NumPy (Harris et al., *Nature* 2020), SciPy (Virtanen et al., *Nat Methods* 2020) and
scikit-image (van der Walt et al., *PeerJ* 2014).

## License

MIT — see [LICENSE](LICENSE).

---

## Resumo em português

`microct3d` mede a porosidade do osso cortical em volumes binários de microCT. Ele
reproduz as definições da análise 3D do CTAn e acrescenta o que o CTAn não mede: a separação entre rede de canais e poros fechados
por um critério local de tamanho (cabe uma esfera de 20 µm), o calibre dos canais
(esfera inscrita e diâmetro equivalente pela área), o espaço poroso por faixa de
calibre, o eixo dos canais com o comprimento real dos trechos e a espessura cortical.
Os valores padrão supõem voxel isotrópico de 5 µm. O repositório contém apenas o
código: não contém imagens, dados ou resultados.
