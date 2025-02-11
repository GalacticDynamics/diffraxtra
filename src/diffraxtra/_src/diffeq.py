"""General wrapper around `diffrax.diffeqsolve`.

This module is private. See `diffraxtra` for the public API.

"""

__all__ = [
    "DiffEqSolver",  # exported to Public API
    # ---
    "default_stepsize_controller",
    "default_max_steps",
    "default_adjoint",
]

from dataclasses import KW_ONLY
from typing import Any, TypeAlias, final

import diffrax as dfx
import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Real

from .diffeq_abc import AbstractDiffEqSolver, params

RealSz0Like: TypeAlias = Real[int | float | Array | np.ndarray[Any, Any], ""]
BoolSz0Like: TypeAlias = Bool[ArrayLike, ""]


default_stepsize_controller = params["stepsize_controller"].default
default_max_steps = params["max_steps"].default
default_adjoint = params["adjoint"].default


@final
class DiffEqSolver(AbstractDiffEqSolver, strict=True):
    """Class-based interface for solving differential equations.

    This is a convenience wrapper around `diffrax.diffeqsolve`, allowing for
    pre-configuration of a `diffrax.AbstractSolver`,
    `diffrax.AbstractStepSizeController`, `diffrax.AbstractAdjoint`, and
    ``max_steps``. Pre-configuring these objects can be useful when you want to:

    - repeatedly solve similar differential equations and can reuse the same
       solver and associated settings.
    - pass the differential equation solver as an argument to a function.

    Note that for some `diffrax.SaveAt` options, `max_steps=None` can be
    incompatible. In such cases, you can override the `max_steps` argument when
    calling the `DiffEqSolver` object.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from diffraxtra import DiffEqSolver

    Construct a solver object.

    >>> solver = DiffEqSolver(dfx.Dopri5(),
    ...                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

    And a differential equation to solve.

    >>> term = dfx.ODETerm(lambda t, y, args: -y)

    Then solve the differential equation.

    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=f64[1], ... )

    The solution can be saved at specific times.

    >>> saveat = dfx.SaveAt(ts=[0., 1., 2., 3.])
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[4],
              ys=f64[4], ... )

    The solution can be densely interpolated.

    >>> saveat = dfx.SaveAt(t1=True, dense=True)
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln
    Solution( t0=f64[], t1=f64[], ts=f64[1],
              ys=f64[1], ... )
    >>> soln.evaluate(0.5)
    Array(0.60653213, dtype=float64)

    Using the `VectorizedDenseInterpolation` class, the interpolation can be
    vectorized, enabling evaluation of batched solutions over batches of times.

    >>> from diffraxtra import VectorizedDenseInterpolation
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
    >>> soln = VectorizedDenseInterpolation.apply_to_solution(soln)
    >>> soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
    Array([[0.90483742, 0.81872516],
           [0.74080871, 0.67031456]], dtype=float64)

    This can be more conveniently done using `vectorize_interpolation`.
    >>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
    ...               vectorize_interpolation=True)
    >>> soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
    Array([[0.90483742, 0.81872516],
           [0.74080871, 0.67031456]], dtype=float64)

    """

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: dfx.AbstractSolver[Any]

    _: KW_ONLY

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: dfx.AbstractStepSizeController[Any, Any] = eqx.field(
        default=default_stepsize_controller
    )

    #: How to differentiate `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: dfx.AbstractAdjoint = eqx.field(default=default_adjoint)

    #: Event. Can override the `event` argument when calling `DiffEqSolver`
    event: dfx.Event | None = None

    #: The maximum number of steps to take before quitting.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: int | None = eqx.field(default=default_max_steps, static=True)
