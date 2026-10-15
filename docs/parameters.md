# Output parameters

Keys returned by `microct3d.analyze_voi`. "TVg" = tissue volume with guard.

## Block A — CTAn-equivalent (TV without guard)

| key | symbol | unit | definition |
|---|---|---|---|
| `TV_mm3` | TV | mm³ | envelope volume (voxel count). Describes the VOI, not the bone |
| `BV_mm3` | BV | mm³ | bone volume (voxel count) |
| `BV_TV_pct` | BV/TV | % | BV / TV |
| `BS_mm2` | BS | mm² | marching-cubes surface of the bone, cut faces included |
| `BS_BV_per_mm` | BS/BV | 1/mm | specific surface |
| `BS_TV_per_mm` | BS/TV | 1/mm | surface density |
| `Obj_N` | Obj.N | n | bone objects, 6-connectivity |
| `PoN_cl` | Po.N(cl) | n | closed pores, background 26-connectivity |
| `PoV_cl_mm3`, `PoS_cl_mm2` | Po.V(cl), Po.S(cl) | mm³, mm² | from each pore's mesh |
| `Po_cl_pct` | Po(cl) | % | Po.V(cl) / (BV + Po.V(cl)) |
| `EuN` | Eu.N | n | Euler number (bone 26-connected). Extensive |
| `EuN_TV_per_mm3` | Eu.N/TV | 1/mm³ | Euler density (size-independent) |
| `Conn`, `ConnDn_per_mm3` | Conn, Conn.Dn | n, 1/mm³ | Obj.N + Po.N(cl) − Eu.N (CTAn convention); / TV |
| `TS_mm2` | TS | mm² | envelope surface |
| `FD`, `FD_R2`, `FD_box_range_vox` | FD | — | box-counting slope on 12–100-voxel boxes |
| `Po_tot_subtraction_pct`, `Po_op_subtraction_pct` | Po(tot), Po(op) | % | by subtraction, comparison only |

## Cortical thickness

| key | unit | definition |
|---|---|---|
| `CtTh_mm`, `CtTh_sd_mm` | mm | mean (SD) of the perpendicular lines in the thicker half of each slice |
| `CtTh_all_lines_mm` | mm | mean of all lines |
| `CtTh_line_to_circle_ratio` | — | line / (2 × EDT); ~1 when the axis is perpendicular to the plate |

## Block B — Closed pores (CTAn definition, no canal criterion)

| key | unit | definition |
|---|---|---|
| `PoN_cl_per_mm3` | 1/mm³ | Po.N(cl) / TV |
| `ESD_cl_mean_um`, `ESD_cl_sd_um` | µm | equivalent spherical diameter, mean and SD over pores |
| `sphericity_cl_mean` | — | mean sphericity (ceiling ≈ 0.91) |
| `elongation_cl_mean` | — | mean √(λ₂/λ₁), pores ≥ 100 voxels |
| `flatness_cl_mean` | — | mean √(λ₃/λ₂), pores ≥ 100 voxels |

## Block C — Canal network (denominator TVg)

| key | unit | definition |
|---|---|---|
| `TV_guard_mm3` | mm³ | tissue volume with guard |
| `V_sphere{D}um_TVg_pct` | % | pore volume where a sphere of D µm fits (sensitivity) |
| `CaV_TV_pct` | % | volume of OPEN canal-space objects / TVg |
| `CaN_per_mm3` | 1/mm³ | number of open canal-space objects / TVg |
| `PoV_cl_criterion_TVg_pct` | % | closed objects under the canal criterion / TVg |
| `PoV_TV_pct` | % | total porosity = open + closed |
| `PoN_per_mm3` | 1/mm³ | open + closed objects / TVg |
| `excess_removed_x` | × | raw pore (after lamella filter) / pore that passed the criterion |
| `lamella_objects_removed` | n | objects discarded by the lamella filter |
| `V_20_70um_TV_pct`, `V_70_100um_TV_pct`, `V_above_100um_TV_pct` | % | pore volume by local thickness; sum = `PoV_TV_pct` |
| `V_above_100um_of_pore_pct` | % | share of pore space above 100 µm |
| `wall_reentrance_depth_um` | µm | mean re-entrance depth of spaces > 100 µm (exploratory) |
| `n_segments`, `axis_length_mm` | n, mm | canal-axis segments and total axis length |
| `axis_below_20_pct`, `axis_20_70_pct`, `axis_70_100_pct`, `axis_above_100_pct` | % | canal-axis length by segment calibre |
| `CaDm_length_weighted_um` | µm | inscribed-sphere calibre, length-weighted |
| `CaDm_unweighted_um` | µm | same, simple mean over segments |
| `CaDm_canals_length_weighted_um` | µm | calibre of segments < 100 µm |
| `Deq_network_um` | µm | area-equivalent diameter of the whole canal space |
| `Deq_canals_um` | µm | area-equivalent diameter of 20–100 µm canals (upper bound) |
| `longitudinal_length_pct` | % | with `--orientation`: axis length at > 45° to the surface normal |
