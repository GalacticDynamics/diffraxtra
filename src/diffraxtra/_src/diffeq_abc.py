"""General wrapper around `diffrax.diffeqsolve`.

This module is private. See `diffraxtra` for the public API.

"""

__all__ = [
    "AbstractDiffEqSolver",  # exported to Public API
    "params",
]

import functools as ft
import inspect
from collections.abc import Mapping
from dataclasses import _MISSING_TYPE, KW_ONLY, MISSING
from typing import Any, TypeAlias

import diffrax as dfx
import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, PyTree, Real
from plum import dispatch

from .interp import VectorizedDenseInterpolation

RealSz0Like: TypeAlias = Real[int | float | Array | np.ndarray[Any, Any], ""]
BoolSz0Like: TypeAlias = Bool[ArrayLike, ""]


# Get the signature of `dfx.diffeqsolve`, first unwrapping the
# `equinox.filter_jit`
params = inspect.signature(dfx.diffeqsolve.__wrapped__).parameters  # type: ignore[attr-defined]
default_saveat = params["saveat"].default
default_progress_meter = params["progress_meter"].default
default_event = params["event"].default
default_throw = params["throw"].default


class AbstractDiffEqSolver(eqx.Module):
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

    If you are here for the first bullet -- repeatedly solving similar
    equations -- reuse the **terms** as well as the solver, and pass whatever
    varies through ``args``. Rebuilding the terms on each call recompiles the
    integrator every time; see `AbstractDiffEqSolver.__call__` for why and for
    the numbers.

    """

    #: The solver for the differential equation.
    #: See the diffrax guide on how to choose a solver.
    solver: eqx.AbstractVar[dfx.AbstractSolver[Any]]

    _: KW_ONLY

    #: How to change the step size as the integration progresses.
    #: See diffrax's list of stepsize controllers.
    stepsize_controller: eqx.AbstractVar[dfx.AbstractStepSizeController[Any, Any]]

    #: How to differentiate `diffeqsolve`.
    #: See `diffrax` for options.
    adjoint: eqx.AbstractVar[dfx.AbstractAdjoint]

    #: Event. Can override the `event` argument when calling `DiffEqSolver`
    event: eqx.AbstractVar[dfx.Event | None]

    #: The maximum number of steps to take before quitting.
    #: Some `diffrax.SaveAt` options can be incompatible with `max_steps=None`,
    #: so you can override the `max_steps` argument when calling `DiffEqSolver`
    max_steps: eqx.AbstractVar[int | None]

    # -------------------------------------------

    # @ft.partial(quax.quaxify)  # TODO: so don't need to strip units
    @dispatch
    @ft.partial(eqx.filter_jit)
    def __call__(  # pylint: disable=R0913,R0914,R0917
        self: "AbstractDiffEqSolver",
        terms: PyTree[dfx.AbstractTerm],
        /,
        t0: RealSz0Like,
        t1: RealSz0Like,
        dt0: RealSz0Like | None,
        y0: PyTree[ArrayLike],
        args: PyTree[Any] = None,
        *,
        # Diffrax options
        saveat: dfx.SaveAt = default_saveat,
        event: dfx.Event | None | _MISSING_TYPE = MISSING,
        max_steps: int | None | _MISSING_TYPE = MISSING,
        throw: bool = default_throw,
        progress_meter: dfx.AbstractProgressMeter[Any] = default_progress_meter,
        solver_state: PyTree[ArrayLike] | None = None,
        controller_state: PyTree[ArrayLike] | None = None,
        made_jump: BoolSz0Like | None = None,
        # Extra options
        vectorize_interpolation: bool = False,
    ) -> dfx.Solution:
        """Solve a differential equation.

        For all arguments, see `diffrax.diffeqsolve`.

        Args:
            terms : the terms of the differential equation.
            t0: the start of the region of integration.
            t1: the end of the region of integration.
            dt0: the step size to use for the first step.
            y0: the initial value. This can be any PyTree of JAX arrays.
            args: any additional arguments to pass to the vector field.
            saveat: what times to save the solution of the differential equation.
            adjoint: how to differentiate diffeqsolve.
            event: an event at which to terminate the solve early.
            max_steps: the maximum number of steps to take before quitting.
            throw: whether to raise an exception if the integration fails.
            progress_meter: a progress meter.
            solver_state: some initial state for the solver.
            controller_state: some initial state for the step size controller.
            made_jump: whether a jump has just been made at t0.

            vectorize_interpolation: whether to vectorize the interpolation
                using `VectorizedDenseInterpolation`.

        Build ``terms`` once, outside the call:
            This method is wrapped in `equinox.filter_jit`, as
            `diffrax.diffeqsolve` is itself. A `diffrax.AbstractTerm` holds a
            plain Python function, which is not a JAX array, so it lands in the
            *static* half of the JIT cache key -- the half compared with ``==``
            rather than traced. Python functions compare by identity, so a term
            constructed inside the calling function is a new cache key on every
            call, and the whole integrator is recompiled each time.

            The cost is roughly three orders of magnitude. On a scalar
            exponential decay, per call::

                                        jax 0.7.1     jax 0.10
                fresh term each call      244 ms       415 ms
                shared term + `args`      0.41 ms      0.31 ms

            The exact figures move with JAX version and machine; the ratio
            does not. Bare `diffrax.diffeqsolve` behaves the same way, so this
            is a property of the filter-JIT boundary, not of this wrapper.

            So define the term once and route per-call data through ``args``::

                # Recompiles every call: `k` is baked into a new closure.
                def solve(k, y0):
                    term = dfx.ODETerm(lambda t, y, args: -k * y)
                    return solver(term, 0.0, 1.0, 0.01, y0)

                # Compiles once: `k` is traced, the term is shared.
                TERM = dfx.ODETerm(lambda t, y, args: -args[0] * y)

                def solve(k, y0):
                    return solver(TERM, 0.0, 1.0, 0.01, y0, (k,))

            This cannot be fixed by hashing the term more cleverly. Every
            closure built from one ``def`` shares a ``__code__`` object, so
            keying on that would make the two ``k`` values above collide and
            silently reuse the wrong compiled function. Identity is what keeps
            the cache correct; ``args`` is the supported way to vary data
            without changing it.

            `jax.config.jax_explain_cache_misses` reports which key changed if
            you suspect a solve is recompiling.

        """
        # Parse `max_steps`, allowing for it to be overridden.
        max_steps = self.max_steps if max_steps is MISSING else max_steps
        # Parse `event`, allowing for it to be overridden.
        event = self.event if event is MISSING else event

        # Solve with `diffrax.diffeqsolve`, using the `DiffEqSolver`'s `solver`,
        # `stepsize_controller` and `adjoint`.
        soln: dfx.Solution = dfx.diffeqsolve(
            terms,
            self.solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            event=event,
            max_steps=max_steps,
            throw=throw,
            progress_meter=progress_meter,
            solver_state=solver_state,
            controller_state=controller_state,
            made_jump=made_jump,
        )
        # Optionally vectorize the interpolation.
        if vectorize_interpolation and soln.interpolation is not None:
            soln = VectorizedDenseInterpolation.apply_to_solution(soln)

        return soln

    # -------------------------------------------

    # TODO: a contextmanager for producing a temporary DiffEqSolver with
    # different field values.

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractDiffEqSolver]", *args: Any, **kwargs: Any
    ) -> "AbstractDiffEqSolver":
        """Construct an `AbstractDiffEqSolver` from arguments."""
        raise NotImplementedError  # pragma: no cover


# ==========================================================


@AbstractDiffEqSolver.__call__.dispatch  # type: ignore[attr-defined,untyped-decorator]
@ft.partial(eqx.filter_jit)
def call(self: "AbstractDiffEqSolver", terms: Any, /, **kwargs: Any) -> dfx.Solution:
    """Solve a differential equation, with keyword arguments."""
    t0 = kwargs.pop("t0")
    t1 = kwargs.pop("t1")
    dt0 = kwargs.pop("dt0", None)
    y0 = kwargs.pop("y0")
    args = kwargs.pop("args", None)
    out: dfx.Solution = self(terms, t0, t1, dt0, y0, args, **kwargs)  # type: ignore[assignment, call-arg]
    return out


# ==========================================================


@AbstractDiffEqSolver.from_.dispatch
def from_(
    cls: type[AbstractDiffEqSolver], obj: AbstractDiffEqSolver, /
) -> AbstractDiffEqSolver:
    """Construct a `DiffEqSolver` from another `DiffEqSolver`.

    The class types must match exactly.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from diffraxtra import DiffEqSolver

    >>> solver = DiffEqSolver(dfx.Dopri5())
    >>> DiffEqSolver.from_(solver) is solver
    True

    """
    if type(obj) is not cls:  # pylint: disable=unidiomatic-typecheck
        msg = f"Cannot convert {type(obj)} to {cls}"
        raise TypeError(msg)
    return obj


@AbstractDiffEqSolver.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[AbstractDiffEqSolver],
    scheme: dfx.AbstractSolver,  # type: ignore[type-arg]
    /,
    **kwargs: Any,
) -> AbstractDiffEqSolver:
    """Construct a `DiffEqSolver` from a `diffrax.AbstractSolver`.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from diffraxtra import DiffEqSolver

    >>> solver = DiffEqSolver.from_(dfx.Dopri5())

    .. skip: next if(DIFFRAX_LT_070, reason="diffrax < 0.7 has different repr")

    >>> solver
    DiffEqSolver(solver=Dopri5())

    """
    return cls(scheme, **kwargs)


@AbstractDiffEqSolver.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[AbstractDiffEqSolver], obj: Mapping[str, Any], /
) -> AbstractDiffEqSolver:
    """Construct a `DiffEqSolver` from a mapping.

    Examples
    --------
    >>> import diffrax as dfx
    >>> from diffraxtra import DiffEqSolver

    >>> solver = DiffEqSolver.from_({"solver": dfx.Dopri5(),
    ...       "stepsize_controller": dfx.PIDController(rtol=1e-5, atol=1e-5)})

    .. skip: next if(DIFFRAX_LT_070, reason="diffrax < 0.7 has different repr")

    >>> solver
    DiffEqSolver(
      solver=Dopri5(), stepsize_controller=PIDController(rtol=1e-05, atol=1e-05)
    )

    """
    return cls(**obj)


# TODO: fix `equinox.Partial` mypy type-arg error below
@AbstractDiffEqSolver.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[AbstractDiffEqSolver],
    obj: eqx.Partial,  # type: ignore[type-arg]
    /,
) -> AbstractDiffEqSolver:
    """Construct a `DiffEqSolver` from an `equinox.Partial`.

    Examples
    --------
    >>> import equinox as eqx
    >>> import diffrax as dfx
    >>> from diffraxtra import DiffEqSolver

    >>> partial = eqx.Partial(dfx.diffeqsolve, solver=dfx.Dopri5())

    >>> solver = DiffEqSolver.from_(partial)

    .. skip: next if(DIFFRAX_LT_070, reason="diffrax < 0.7 has different repr")

    >>> solver
    DiffEqSolver(solver=Dopri5())

    """
    obj = eqx.error_if(
        obj, obj.func is not dfx.diffeqsolve, "must be a partial of diffeqsolve"
    )
    return cls(**obj.keywords)  # TODO: what about obj.args?
