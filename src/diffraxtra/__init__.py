"""Extras for `diffrax`."""

__all__ = [
    "AbstractDiffEqSolver",
    "DiffEqSolver",
    "AbstractVectorizedDenseInterpolation",
    "VectorizedDenseInterpolation",
]

from ._src import (
    AbstractDiffEqSolver,
    AbstractVectorizedDenseInterpolation,
    DiffEqSolver,
    VectorizedDenseInterpolation,
)
from ._version import (  # noqa: F401
    version as __version__,
    version_tuple as __version_tuple__,
)
