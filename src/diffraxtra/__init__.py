"""Extras for `diffrax`."""

__all__ = [
    "DiffEqSolver",
    "AbstractVectorizedDenseInterpolation",
    "VectorizedDenseInterpolation",
]

from ._version import (  # noqa: F401
    version as __version__,
    version_tuple as __version_tuple__,
)
from .diffeq import DiffEqSolver
from .interp import AbstractVectorizedDenseInterpolation, VectorizedDenseInterpolation
