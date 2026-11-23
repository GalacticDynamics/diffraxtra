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
from typing import Any, Literal, final

import diffrax as dfx
import equinox as eqx

from .diffeq_abc import AbstractDiffEqSolver, default_max_steps, params

default_stepsize_controller = params["stepsize_controller"].default
default_adjoint = params["adjoint"].default


@final
class DiffEqSolver(AbstractDiffEqSolver):  # pylint: disable=R0903,W0223
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

    With a `diffrax.StepTo` controller, ``max_steps`` defaults to the number of
    steps on the grid, ``len(ts) - 1``. Saving every step shows the bound,
    because diffrax pads the saved steps out to ``max_steps``.

    >>> grid = jnp.linspace(0, 1, 11)
    >>> stepto = DiffEqSolver(dfx.Euler(), stepsize_controller=dfx.StepTo(grid))
    >>> every_step = dfx.SaveAt(steps=True)
    >>> stepto(term, t0=0, t1=1, y0=1, saveat=every_step).ys.shape
    (10,)

    A ``max_steps`` given to the solver or to the call is used as is.

    >>> stepto(term, t0=0, t1=1, y0=1, saveat=every_step, max_steps=20).ys.shape
    (20,)
    >>> stepto30 = DiffEqSolver(dfx.Euler(), stepsize_controller=dfx.StepTo(grid),
    ...                         max_steps=30)
    >>> stepto30(term, t0=0, t1=1, y0=1, saveat=every_step).ys.shape
    (30,)

    Passing ``max_steps="auto"`` to the call asks for the grid's step count
    even when the solver holds an integer.

    >>> stepto30(term, t0=0, t1=1, y0=1, saveat=every_step, max_steps="auto").ys.shape
    (10,)

    ``max_steps=None`` stays unbounded, so diffrax rejects it when saving
    every step.

    >>> stepto(term, t0=0, t1=1, y0=1, saveat=every_step, max_steps=None)
    Traceback (most recent call last):
    ...
    ValueError: `max_steps=None` is incompatible with saving at `steps=...

    Other controllers use diffrax's default.

    >>> solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=every_step).ys.shape
    (4096,)

    `diffraxtra.fixed_steps` builds a `diffrax.StepTo` grid from ``t0``,
    ``dt`` and a number of steps.

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
    #: The default, ``"auto"``, means ``len(ts) - 1`` with a `diffrax.StepTo`
    #: controller, and `diffrax.diffeqsolve`'s default otherwise.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: int | None | Literal["auto"] = eqx.field(default="auto", static=True)
