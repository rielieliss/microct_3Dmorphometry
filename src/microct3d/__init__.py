"""microct3d: 3D morphometry of cortical bone porosity from micro-CT.

Reproduces CTAn's 3D analysis (validated against it) and adds what the commercial
software does not provide: separation of the canal network from closed pores by a
local size criterion, canal calibre, pore-space size bands, the canal axis with true
segment lengths, and cortical thickness.
"""
from .config import DEFAULT, Params
from .pipeline import analyze_voi

__all__ = ["Params", "DEFAULT", "analyze_voi"]
__version__ = "1.0.2"
