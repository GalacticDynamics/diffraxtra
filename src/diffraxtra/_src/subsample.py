"""Reduce the storage of a dense solution by cubic Hermite resampling."""

__all__ = ["subsample"]

import functools as ft
import operator
from collections.abc import Callable
from typing import Any, cast
from typing_extensions import override

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .interp import VectorizedDenseInterpolation

# Keep typing compatible with JAX releases where ravel_pytree is untyped.
_ravel: Callable[[Any], tuple[jax.Array, Callable[[jax.Array], Any]]] = ravel_pytree


class _LocalInterpolation(dfx.AbstractLocalInterpolation):
    """Select an original solver piece or a resampled Hermite piece."""

    t0: Any  # type: ignore[misc]
    t1: Any  # type: ignore[misc]
    original: dfx.AbstractLocalInterpolation
    hermite: dfx.ThirdOrderHermitePolynomialInterpolation
    exact: jax.Array

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        t0: Any,
        t1: Any,
        data: jax.Array,
        exact: jax.Array,
        original_cls: Callable[..., dfx.AbstractLocalInterpolation],
        original_unravel: Callable[[jax.Array], Any],
        hermite_unravel: Callable[[jax.Array], Any],
        original_size: int,
        hermite_size: int,
    ) -> None:
        self.t0, self.t1 = t0, t1
        self.original = original_cls(
            t0=t0, t1=t1, **original_unravel(data[:original_size])
        )
        self.hermite = dfx.ThirdOrderHermitePolynomialInterpolation(
            t0=t0, t1=t1, **hermite_unravel(data[:hermite_size])
        )
        self.exact = exact

    @override
    def evaluate(self, t0: Any, t1: Any = None, left: bool = True) -> Any:
        return jax.lax.cond(
            self.exact,
            lambda: self.original.evaluate(t0, t1, left=left),
            lambda: self.hermite.evaluate(t0, t1, left=left),
        )


def _resample(interp: dfx.DenseInterpolation, max_steps: int) -> dfx.DenseInterpolation:
    # The two interpolation layouts each require shape and reconstruction metadata.
    # pylint: disable=too-many-locals
    if interp.ts.ndim != 1:
        msg = (
            "Use jax.vmap(subsample) or VectorizedDenseInterpolation "
            "for batched solutions."
        )
        raise ValueError(msg)
    if max_steps >= interp.ts.shape[0] - 1:
        return interp

    def resize(x: jax.Array, size: int) -> jax.Array:
        x = x[:size]
        return jnp.pad(x, ((0, size - x.shape[0]),) + ((0, 0),) * (x.ndim - 1))

    # Both paths have the same static layout, so mixed batches work under JIT.
    exact = interp.ts_size <= max_steps + 1
    original_example = jax.tree.map(
        lambda x: jnp.zeros(x.shape[1:], x.dtype), interp.infos
    )
    y_example = jax.tree.map(jnp.zeros_like, interp.y0_if_trivial)
    hermite_example = dict.fromkeys(("y0", "y1", "k0", "k1"), y_example)
    flat_original, original_unravel = _ravel(original_example)
    flat_hermite, hermite_unravel = _ravel(hermite_example)
    width = max(flat_original.size, flat_hermite.size)
    dtype = jnp.result_type(flat_original, flat_hermite)

    def pack(infos: Any) -> jax.Array:
        data = jax.vmap(lambda info: _ravel(info)[0])(infos)
        return jnp.pad(data.astype(dtype), ((0, 0), (0, width - data.shape[1])))

    def resample() -> tuple[jax.Array, Any]:
        # Diffrax stores increasing internal times even for backward integration.
        ts = jnp.linspace(interp.ts[0], interp.ts[interp.ts_size - 1], max_steps + 1)
        times = ts * interp.direction
        ys = jax.vmap(interp.evaluate)(times)
        dys = jax.vmap(interp.derivative)(times)

        def increments(dy: jax.Array) -> jax.Array:
            dt = jnp.diff(ts).reshape((max_steps,) + (1,) * (dy.ndim - 1))
            return dy * dt * interp.direction

        infos = {
            "y0": jax.tree.map(lambda y: y[:-1], ys),
            "y1": jax.tree.map(lambda y: y[1:], ys),
            "k0": jax.tree.map(lambda dy: increments(dy[:-1]), dys),
            "k1": jax.tree.map(lambda dy: increments(dy[1:]), dys),
        }
        return ts, pack(infos)

    def trim() -> tuple[jax.Array, Any]:
        original = jax.tree.map(lambda x: resize(x, max_steps), interp.infos)
        ts = resize(interp.ts, max_steps + 1)
        ts = jnp.where(jnp.arange(max_steps + 1) < interp.ts_size, ts, jnp.inf)
        return ts, pack(original)

    ts, data = jax.lax.cond(exact, trim, resample)
    return dfx.DenseInterpolation(
        ts=ts,
        ts_size=jnp.where(exact, interp.ts_size, max_steps + 1),
        infos={"data": data, "exact": jnp.full((max_steps,), exact)},
        interpolation_cls=ft.partial(
            _LocalInterpolation,
            original_cls=interp.interpolation_cls,
            original_unravel=original_unravel,
            hermite_unravel=hermite_unravel,
            original_size=flat_original.size,
            hermite_size=flat_hermite.size,
        ),
        direction=interp.direction,
        t0_if_trivial=interp.t0_if_trivial,
        y0_if_trivial=interp.y0_if_trivial,
    )


def subsample(soln: dfx.Solution, /, *, max_steps: int) -> dfx.Solution:
    """Return a dense solution with storage for ``max_steps`` intervals.

    If the accepted steps fit, only unused padding is removed and the original
    interpolation is preserved exactly. Otherwise, values and time derivatives
    of the original dense curve are sampled at
    ``max_steps + 1`` uniformly spaced times over its actual integration interval.
    These define cubic Hermite pieces; simply slicing the solver's original
    coefficients would give incorrect interpolation across skipped steps.

    A budget at or above the original capacity leaves the interpolation unchanged.
    When resampling, the new curve agrees at the sampling knots (both endpoints).
    Between knots it is an approximation, with no inherited solver tolerance
    guarantee. Increase ``max_steps`` and check convergence for your application.
    Intended for smooth ODE solutions; jumps and stochastic paths are not preserved.

    Only the interpolation is replaced, using ``equinox.tree_at``. Saved ``ts``,
    ``ys``, statistics, and solver state retain their original meaning and storage.
    The input is unchanged. The returned interpolation stores no reference to the
    original dense buffers. This does not change the integration's step budget.

    Supports PyTree states, backward integration, and
    :class:`~diffraxtra.VectorizedDenseInterpolation`. For ordinary batched dense
    solutions, apply with ``jax.vmap``. Under JIT, ``max_steps`` must be static.

    Args:
        soln: A solution saved with ``diffrax.SaveAt(dense=True)``.
        max_steps: Positive integer number of intervals in the new interpolation.

    Returns:
        A new ``diffrax.Solution`` containing the smaller dense interpolation.

    Examples:
    >>> import diffrax as dfx
    >>> from diffraxtra import subsample
    >>> sol = dfx.diffeqsolve(
    ...     dfx.ODETerm(lambda t, y, args: -y), dfx.Dopri5(),
    ...     t0=0., t1=1., dt0=0.01, y0=1.,
    ...     saveat=dfx.SaveAt(dense=True), max_steps=256)
    >>> compact = subsample(sol, max_steps=16)
    >>> compact.interpolation.ts.shape
    (17,)
    >>> bool(abs(compact.evaluate(0.5) - sol.evaluate(0.5)) < 1e-6)
    True
    >>> trimmed = subsample(sol, max_steps=128)
    >>> bool(trimmed.evaluate(0.5) == sol.evaluate(0.5))
    True

    """
    if isinstance(max_steps, bool):
        msg = "max_steps must be a positive integer."
        raise TypeError(msg)
    try:
        max_steps = operator.index(max_steps)
    except TypeError as error:
        msg = "max_steps must be a static positive integer."
        raise TypeError(msg) from error
    if max_steps < 1:
        msg = "max_steps must be positive."
        raise ValueError(msg)

    interp = cast("Any", soln.interpolation)
    if isinstance(interp, VectorizedDenseInterpolation):
        scalar = jax.vmap(lambda x: _resample(x, max_steps))(
            interp.scalar_interpolation
        )
        new_interp = eqx.tree_at(lambda x: x.scalar_interpolation, interp, scalar)
    elif isinstance(interp, dfx.DenseInterpolation):
        new_interp = _resample(interp, max_steps)
    elif interp is None:
        msg = "subsample requires a dense solution; use SaveAt(dense=True)."
        raise ValueError(msg)
    else:
        msg = "subsample requires a diffrax.DenseInterpolation."
        raise TypeError(msg)
    result: dfx.Solution = eqx.tree_at(lambda x: x.interpolation, soln, new_interp)
    return result
