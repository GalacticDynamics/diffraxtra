"""Fixed-step grids for `diffrax.StepTo`.

This module is private. See `diffraxtra` for the public API.

"""

__all__ = ["fixed_steps"]  # exported to Public API

import operator

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Real


# jax < 0.5.3 has no JVP rule for `optimization_barrier`. The barrier is the
# identity, so its tangent passes straight through.
# TODO: call `jax.lax.optimization_barrier` directly once the jax floor is 0.5.3.
@jax.custom_jvp
def _barrier(x: Array) -> Array:
    return jax.lax.optimization_barrier(x)  # type: ignore[no-untyped-call,no-any-return]


@_barrier.defjvp
def _barrier_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    return _barrier(*primals), tangents[0]


def fixed_steps(
    t0: Real[ArrayLike, ""], dt: Real[ArrayLike, ""], n_steps: int, /
) -> tuple[Real[Array, ""], Real[Array, ""], dfx.StepTo, int]:
    """Make the grid for ``n_steps`` fixed steps of size ``dt`` from ``t0``.

    The grid is ``t0 + dt * arange(n_steps + 1)``. This returns
    ``(t0, t1, stepsize_controller, max_steps)``, ready to pass to
    `diffrax.diffeqsolve` with ``dt0=None``. ``t0`` and ``t1`` are the ends of
    the grid, the controller is a `diffrax.StepTo` on the grid, and
    ``max_steps`` is ``n_steps``. `DiffEqSolver` takes ``max_steps`` from a
    `diffrax.StepTo` by itself, so with it you can drop the fourth value.

    The grid is built behind `jax.lax.optimization_barrier`, which stops
    XLA's simplifier from rewriting ``t1`` and the grid separately with
    different roundings. `diffrax.StepTo` rejects a ``t1`` that is even one
    rounding off ``ts[-1]``.

    ``t0`` and ``dt`` can be traced, so this works under `jax.jit`,
    `jax.vmap` and `jax.grad`. ``n_steps`` sets the shape of the grid, so it
    must be a concrete integer. A negative ``dt`` runs backward in time.
    ``t0`` and ``dt`` must be scalars. To run several grids, `jax.vmap` over
    them.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import diffrax as dfx
    >>> from diffraxtra import DiffEqSolver, fixed_steps

    >>> t0, t1, stepto, max_steps = fixed_steps(0.0, 0.25, 4)
    >>> t0, t1, max_steps
    (Array(0., dtype=float64...), Array(1., dtype=float64...), 4)
    >>> stepto.ts
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float64...)

    Solve ``dy/dt = t`` with `diffrax.diffeqsolve`. Heun's method is exact on
    it, so ``y(t1) = y0 + (t1**2 - t0**2) / 2``.

    >>> term = dfx.ODETerm(lambda t, y, args: t)
    >>> soln = dfx.diffeqsolve(term, dfx.Heun(), t0, t1, None, 1.0,
    ...                        stepsize_controller=stepto, max_steps=max_steps)
    >>> soln.ys
    Array([1.5], dtype=float64)
    >>> soln.stats["num_steps"]
    Array(4, dtype=...)

    A negative ``dt`` runs backward, here from 1 to 0.

    >>> t0, t1, stepto, _ = fixed_steps(1.0, -0.25, 4)
    >>> solver = DiffEqSolver(dfx.Heun(), stepsize_controller=stepto)
    >>> soln = solver(term, t0, t1, None, 1.5)
    >>> soln.ts, soln.ys
    (Array([0.], dtype=float64), Array([1.], dtype=float64))

    ``t0`` and ``dt`` can be traced. Here ``t1 = t0 + 100 dt``, so the
    derivatives of ``y(t1)`` are ``t1 - t0 = 1`` for ``t0`` and
    ``100 t1 = 200`` for ``dt``.

    >>> @jax.jit
    ... def y1(t0, dt):
    ...     t0, t1, stepto, _ = fixed_steps(t0, dt, 100)
    ...     solver = DiffEqSolver(dfx.Heun(), stepsize_controller=stepto)
    ...     return solver(term, t0, t1, None, 1.0).ys[-1]
    >>> y1(1.0, 0.01)
    Array(2.5, dtype=float64)
    >>> jnp.round(jnp.stack(jax.grad(y1, argnums=(0, 1))(1.0, 0.01)), 8)
    Array([  1., 200.], dtype=float64)
    >>> jnp.round(jax.vmap(y1, in_axes=(0, None))(jnp.array([0.0, 1.0]), 0.01), 8)
    Array([1.5, 2.5], dtype=float64)

    ``n_steps`` must be at least 1.

    >>> fixed_steps(0.0, 0.1, 0)
    Traceback (most recent call last):
    ...
    ValueError: n_steps must be at least 1, got 0

    >>> fixed_steps(jnp.array([0.0, 1.0, 2.0]), 0.1, 2)
    Traceback (most recent call last):
    ...
    ValueError: t0 and dt must be scalars, got shapes (3,) and ()
    >>> fixed_steps(0.0, jnp.array([0.1, 0.2, 0.3]), 2)
    Traceback (most recent call last):
    ...
    ValueError: t0 and dt must be scalars, got shapes () and (3,)

    ``n_steps`` must be an integer.

    >>> fixed_steps(0.0, 0.1, 2.5)
    Traceback (most recent call last):
    ...
    TypeError: 'float' object cannot be interpreted as an integer

    """
    n_steps = operator.index(n_steps)
    if n_steps < 1:
        msg = f"n_steps must be at least 1, got {n_steps}"
        raise ValueError(msg)
    if jnp.ndim(t0) or jnp.ndim(dt):
        msg = (
            f"t0 and dt must be scalars, got shapes {jnp.shape(t0)} and {jnp.shape(dt)}"
        )
        raise ValueError(msg)
    grid = _barrier(t0 + dt * jnp.arange(n_steps + 1))
    return grid[0], grid[-1], dfx.StepTo(ts=grid), n_steps
