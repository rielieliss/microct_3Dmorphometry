# Changelog

## 1.0.2 — 2026-10-14

No change in any computed value.

- New `tests/test_open_closed.py`: an object that touches the outside only along z,
  inside one slab, must be open; and the block-wise classification must equal a
  brute-force single-pass reference for several slab sizes. 44 tests in total.
- `canals.open_and_closed` accepts `max_voxels` (slab size), as the other block-wise
  functions do.

## 1.0.1 — 2026-10-14

Review release.

- **Fix:** the open/closed classification of canal objects now tests 6-adjacency
  along z inside each slab as well (earlier scripts tested it only across slab
  boundaries, so a few open objects were counted as closed). Total porosity is
  unaffected.

- **Memory:** the pipeline now works in place and keeps at most three full-size
  boolean volumes alive; closed pores are labelled on the complement slab by slab.
  1.0.0 could exceed 16 GB on multi-gigavoxel VOIs.
- **Fix:** a PyTorch installation with broken CUDA libraries raised `OSError` on
  import; the diffusion filter now falls back to NumPy.
- `morphometry.bone_morphometry` and `morphometry.tissue_morphometry` split the
  CTAn-equivalent parameters by what they need (`ctan_morphometry` still does both).
- `canals.open_and_closed` takes the tissue mask and builds its complement per slab.
- New `tests/test_reference.py`: every computation checked against an independent
  implementation (scikit-image, SciPy, textbook definitions). 39 tests in total.
- Documented limitations: skeleton end-shortening (D_eq is an upper bound) and the
  absence of edge correction in Conn.

## 1.0.0 — 2026-10-13

First public release: a single pipeline, a command line, tests and documentation.

Canal-axis analysis:

- the axis is the skeleton of the space that passed the 20 µm canal criterion, not of
  the whole pore space (the old denominator included stubs and necks below 20 µm);
- segment length from the minimum spanning tree (diagonal steps counted as √2 and
  √3), not voxel count;
- canal calibre weighted by segment length;
- new area-equivalent diameter (`Deq_canals_um`);
- Euler number also reported per tissue volume (`EuN_TV_per_mm3`).
