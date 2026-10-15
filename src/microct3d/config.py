"""Analysis parameters.

Every number that defines the method lives here. The defaults assume 5 µm isotropic
voxels.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Params:
    """Parameters of the cortical-porosity pipeline.

    Attributes
    ----------
    voxel_um
        Isotropic voxel size in micrometres.
    shrinkwrap_diameter_vox
        Diameter (voxels) of the sphere used by the 3D shrink-wrap envelope. Openings
        narrower than this are bridged over, as in CTAn's "ROI shrink-wrap (3D),
        stretch over holes with diameter D". 30 voxels = 150 µm.
    guard_vox
        Thickness (voxels) of the layer just inside the envelope that is excluded from
        porosity measurements. The envelope never sits exactly on the bone surface; the
        thin gap between them would otherwise be counted as pore space.
    lamella_min_extent_vox
        Pore-space objects whose smallest bounding-box side is below this value are
        segmentation lamellae, not pores, and are discarded.
    canal_diameter_um
        A point of pore space is "canal" if a sphere of this diameter fits there
        (morphological opening). 20 µm is 4 voxels at 5 µm.
    sensitivity_diameters_um
        Extra opening diameters whose volumes are recorded, so the canal criterion can
        be revised without reprocessing.
    size_bands_um
        Upper limits of the size bands used to split pore space by local thickness:
        20-70 µm, 70-100 µm and > 100 µm.
    cavity_diameter_um
        Sphere used to build the "smooth cavity" for the wall re-entrance index.
    reentrance_bands_vox
        Widths (voxels) of the shell around the smooth cavity where re-entrances are
        measured; ``reentrance_main_vox`` is the one reported.
    min_segment_vox
        Skeleton segments shorter than this are ignored (calibre undefined).
    orientation_cutoff_deg
        Angle to the local surface normal above which a segment is longitudinal
        (Haversian) rather than transverse (Volkmann), after Pratt & Cooper.
    fd_box_sizes_vox
        Box sizes used to fit the fractal dimension (12-100 voxels). Very small boxes
        measure the voxel staircase, not the roughness of the surface.
    fd_box_fit_factor
        A box size is used only if it fits this many times in the VOI height.
    closed_pore_connectivity
        Background connectivity for closed pores. 26 reproduces CTAn's Po.N(cl).
    object_connectivity
        Bone connectivity for Obj.N. 6 reproduces CTAn's Obj.N.
    shape_min_vox
        Closed pores smaller than this get no elongation/flatness (too few points).
    ctth_n_slices, ctth_line_spacing, ctth_pca_radius
        Cortical thickness: number of slices, one line every N skeleton points, and
        the neighbourhood (pixels) used to estimate the local axis direction.
    max_block_voxels
        Memory budget for block-wise processing along z.
    """

    voxel_um: float = 5.0
    shrinkwrap_diameter_vox: int = 30
    guard_vox: int = 3
    lamella_min_extent_vox: int = 3
    canal_diameter_um: float = 20.0
    sensitivity_diameters_um: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0)
    size_bands_um: tuple[float, ...] = (20.0, 70.0, 100.0)
    cavity_diameter_um: float = 100.0
    reentrance_bands_vox: tuple[int, ...] = (3, 5, 10)
    reentrance_main_vox: int = 5
    min_segment_vox: int = 4
    orientation_cutoff_deg: float = 45.0
    fd_box_sizes_vox: tuple[int, ...] = (12, 16, 24, 32, 48, 64, 100)
    fd_box_fit_factor: int = 3
    closed_pore_connectivity: int = 26
    object_connectivity: int = 6
    shape_min_vox: int = 100
    ctth_n_slices: int = 10
    ctth_line_spacing: int = 8
    ctth_pca_radius: int = 25
    max_block_voxels: float = 1.5e8
    extra: dict = field(default_factory=dict)

    @property
    def voxel_mm(self) -> float:
        return self.voxel_um / 1000.0

    def radius_vox(self, diameter_um: float) -> float:
        """Sphere radius in voxels for a diameter given in micrometres."""
        return diameter_um / 2.0 / self.voxel_um


DEFAULT = Params()
"""The default parameter set."""
