---
name: diffraxtra
description:
  Use when writing, reviewing, or debugging code that imports diffraxtra —
  `DiffEqSolver`, `VectorizedDenseInterpolation`, or their `Abstract*` bases.
  Also use when evaluating a `diffrax` dense interpolation at an array of times,
  when a batched `diffrax.diffeqsolve` result needs `.evaluate`, when the output
  of a batched solve has an unexpected shape, when `vectorize_interpolation` or
  `apply_to_solution` appears, when a per-call `max_steps` or `event` override
  is needed, when `max_steps=None` conflicts with a `diffrax.SaveAt`, or when a
  `DiffEqSolver` recompiles on every call.
---

# Using diffraxtra Effectively

`diffraxtra` is two extras for [diffrax](https://docs.kidger.site/diffrax/), and
nothing else:

- **`DiffEqSolver`** — an `equinox.Module` holding a `solver`,
  `stepsize_controller`, `adjoint`, `event` and `max_steps`, callable with the
  rest of `diffrax.diffeqsolve`'s arguments. Use it to pre-configure a solve
  once and pass it around, or to reuse one configuration across many solves.
- **`VectorizedDenseInterpolation`** — a wrapper around
  `diffrax.DenseInterpolation` that evaluates a **batched** solution at an
  **arbitrarily-shaped** array of times, without you writing the `jax.vmap`.

Each comes as an `Abstract*`/`@final` pair of `equinox.Module`s, and each has a
`plum`-dispatched `from_` constructor. That is the whole library.

Checked against diffraxtra 1.5.3, diffrax 0.7.0, equinox 0.13.0, plum-dispatch
2.5.7, jax 0.7.1, Python >=3.11. The supported floor is `diffrax>=0.6`.

## Quick start

```python
import diffrax as dfx
import jax.numpy as jnp

from diffraxtra import DiffEqSolver

solver = DiffEqSolver(
    dfx.Dopri5(), stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8)
)
term = dfx.ODETerm(lambda t, y, args: -y)
saveat = dfx.SaveAt(t1=True, dense=True)

soln = solver(
    term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=saveat, vectorize_interpolation=True
)

# A plain `diffrax` solution only evaluates one time at a time; this one takes
# any shape.
assert soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2)).shape == (2, 2)
```

`solver(...)` forwards everything to `diffrax.diffeqsolve` — the signature is
diffrax's, minus the three arguments the object already holds
(`solver`/`stepsize_controller`/`adjoint`), plus `vectorize_interpolation`.

The examples below run in sequence and reuse `term`, `saveat`, `solver` and
`pid` from here, except where a block re-imports for itself.

## The shape contract

This is the thing to get right. `evaluate` returns

```text
(*interp.batch_shape, *jnp.shape(t), *jnp.shape(y0))
```

— batch dimensions first, then the shape of the times you passed, then the shape
of `y0`. Every one of the three parts can be empty.

```python
import jax

from diffraxtra import VectorizedDenseInterpolation

pid = dfx.PIDController(rtol=1e-8, atol=1e-8)


@jax.vmap
def solve(y0):
    return dfx.diffeqsolve(
        term,
        dfx.Dopri5(),
        t0=0.0,
        t1=3.0,
        dt0=0.1,
        y0=y0,
        saveat=saveat,
        stepsize_controller=pid,
    )


sol = solve(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))  # 3 solves of a 2-vector
interp = VectorizedDenseInterpolation(sol.interpolation)

assert interp.batch_shape == (3,)

ts = jnp.linspace(0.0, 3.0, 4)
assert interp.evaluate(1.0).shape == (3, 2)  # (*batch, *y0)
assert interp.evaluate(ts).shape == (3, 4, 2)  # (*batch, *t, *y0)
assert interp.evaluate(ts.reshape(2, 2)).shape == (3, 2, 2, 2)  # (*batch, 2, 2, *y0)
```

The `y0` shape is not stored; it falls out of the interpolation's own leaves.
`batch_shape` is inferred from `jnp.shape(scalar_interpolation.t0_if_trivial)` —
i.e. from the solve's own batching, **not** from the shape of `y0`. An unbatched
solve of a vector `y0` has `batch_shape == ()`.

Batch dimensions can be nested arbitrarily deep, and are **stored flattened**:

```python
sol2 = jax.vmap(solve)(jnp.ones((2, 3, 2)))  # 2x3 solves of a 2-vector
interp2 = VectorizedDenseInterpolation(sol2.interpolation)

assert interp2.batch_shape == (2, 3)
assert interp2.scalar_interpolation.t0_if_trivial.shape == (6,)  # flattened to 2*3
assert interp2.evaluate(ts).shape == (2, 3, 4, 2)
```

`scalar_interpolation` is the original `diffrax.DenseInterpolation` with every
batch dimension collapsed into one leading axis; `batch_shape` is what restores
it. If you construct the wrapper by hand, either let both be inferred or make
sure the `batch_shape` you pass has the right **ndim**, not just the right
product — the mismatch surfaces as an opaque `vmap got inconsistent sizes`
error, from a later `evaluate`, not from the constructor.

## Three ways to get a vectorized interpolation

| Way                                                    | Use when                                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------ |
| `solver(..., vectorize_interpolation=True)`            | you own the solve and are already using `DiffEqSolver`                   |
| `VectorizedDenseInterpolation.apply_to_solution(soln)` | you have a `diffrax.Solution` from anywhere and want it wrapped in place |
| `VectorizedDenseInterpolation(soln.interpolation)`     | you want the interpolation object itself, not a `Solution`               |

`apply_to_solution` is **out-of-place** (it returns a new `Solution` via
`eqx.tree_at`; the original is untouched) and is a no-op if the interpolation is
already vectorized or `None`:

```python
vec = VectorizedDenseInterpolation.apply_to_solution(sol)

assert isinstance(vec, dfx.Solution)
assert isinstance(vec.interpolation, VectorizedDenseInterpolation)
assert VectorizedDenseInterpolation.apply_to_solution(vec) is vec  # idempotent
assert not isinstance(sol.interpolation, VectorizedDenseInterpolation)  # out-of-place
```

The wrapped `Solution` keeps every other diffrax field, so `soln.ts`, `soln.ys`
and `soln.stats` are unchanged; only `soln.interpolation` — and therefore
`soln.evaluate` — differs.

## Two call forms

`__call__` is `plum`-dispatched, so a solver takes its arguments either
positionally (diffrax's own order) or by keyword. The keyword form makes `dt0`
optional, defaulting to `None`:

```python
t1_only = dfx.SaveAt(t1=True)

positional = solver(term, 0.0, 3.0, 0.1, 1.0, saveat=t1_only)
by_keyword = solver(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only)
no_dt0 = solver(term, t0=0.0, t1=3.0, y0=1.0, saveat=t1_only)  # dt0 defaults to None

assert positional.ys.shape == by_keyword.ys.shape == no_dt0.ys.shape == (1,)
```

`terms` is positional-only in both. The keyword overload pops
`t0`/`t1`/`dt0`/`y0`/`args` out of `**kwargs` and re-enters the positional one,
and each form is separately jitted.

## `max_steps` and `event` are three-way arguments

These two are fields you can also override per call, and the override is
genuinely three-way — omitting the argument is **not** the same as passing
`None`. The parameter defaults to `dataclasses.MISSING`, and only that sentinel
means "use the field".

```python
assert DiffEqSolver(dfx.Dopri5()).max_steps == 4096  # diffrax's default, scraped
assert DiffEqSolver(dfx.Dopri5()).event is None

solver(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only)  # use the field
solver(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only, max_steps=10_000)
solver(
    term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only, max_steps=None
)  # unbounded
```

For `event` that means a solver carrying a termination condition can be asked
for an un-terminated solve, and a plain solver can be given one for a single
call:

```python
event = dfx.Event(lambda t, y, args, **kw: y - 0.5)  # stop when y crosses 0.5
stopping = DiffEqSolver(dfx.Dopri5(), stepsize_controller=pid, event=event)

stopped = stopping(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only)
disabled = stopping(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only, event=None)
one_off = solver(term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=t1_only, event=event)

assert float(stopped.ts[0]) < 3.0  # terminated early
assert float(disabled.ts[0]) == 3.0  # ran to t1
assert float(one_off.ts[0]) < 3.0  # event supplied per-call
```

`max_steps=None` means unbounded, which some `SaveAt` options reject:

```py
solver(
    term, t0=0.0, t1=3.0, dt0=0.1, y0=1.0, saveat=dfx.SaveAt(steps=True), max_steps=None
)
# ValueError: `max_steps=None` is incompatible with saving at `steps=True`
```

That is exactly why the per-call override exists — configure a solver once with
a bounded `max_steps`, and drop the bound only for the calls that can take it.

## What is static, and what retraces

`AbstractDiffEqSolver.__call__` is wrapped in `eqx.filter_jit`, so the whole
solve is jitted for you. Consequences worth knowing before you profile:

- `max_steps` is a `static=True` field **and** a static argument. Every distinct
  value is a separate compilation. Sweeping `max_steps` in a loop recompiles
  each time.
- `throw` and `vectorize_interpolation` are Python `bool`s read at trace time —
  also static.
- A structurally different `solver`, `stepsize_controller`, or `adjoint`
  retraces. Reusing one `DiffEqSolver` instance across calls is the point.
- `y0`'s dtype and shape are traced normally, so `y0=1` (int) and `y0=1.0`
  (float) are two compilations.
- Don't wrap `solver(...)` in another `jax.jit` expecting a speedup; it is
  already jitted. Do wrap a _larger_ function that calls it.

## Constructing a solver: `from_`

`DiffEqSolver.from_` is a `plum`-dispatched classmethod that normalises a
`DiffEqSolver`, a `dfx.AbstractSolver`, a `Mapping`, or an `eqx.Partial` into a
`DiffEqSolver`. Use it in library code that accepts "a solver" loosely; the
[README](../../README.md#diffeqsolver) has a worked example of each.

Three things the examples don't show: passing an existing instance returns it
unchanged rather than copying; that overload requires an **exact** type match,
so a subclass instance raises `TypeError`; and the `eqx.Partial` must wrap
`diffrax.diffeqsolve` itself, contributing only its keywords.
`VectorizedDenseInterpolation.from_` mirrors all of this, plus an optional
positional `batch_shape`.

## Writing your own solver type

Both public classes are `Abstract*`/`@final` pairs of `equinox.Module`s: the
base declares `eqx.AbstractVar` fields, the concrete subclass supplies them. To
make your own variant, subclass the **abstract** class — you inherit both
`__call__` forms and every `from_` overload for free.

```python
from dataclasses import KW_ONLY
from typing import Any, final

import diffrax as dfx
import equinox as eqx

from diffraxtra import AbstractDiffEqSolver


@final
class TightSolver(AbstractDiffEqSolver):
    """A solver whose defaults are tighter than diffrax's."""

    solver: dfx.AbstractSolver[Any]

    _: KW_ONLY

    stepsize_controller: dfx.AbstractStepSizeController[Any, Any] = eqx.field(
        default=dfx.PIDController(rtol=1e-10, atol=1e-10)
    )
    adjoint: dfx.AbstractAdjoint = eqx.field(default=dfx.RecursiveCheckpointAdjoint())
    event: dfx.Event | None = None
    max_steps: int | None = eqx.field(default=16_384, static=True)


tight = TightSolver(dfx.Dopri5())
decay = dfx.ODETerm(lambda t, y, args: -y)
assert tight(decay, t0=0.0, t1=3.0, dt0=0.1, y0=1.0).ys.shape == (1,)
assert isinstance(TightSolver.from_(dfx.Dopri5()), TightSolver)  # inherited
```

You must declare **all five** `AbstractVar`s — `solver`, `stepsize_controller`,
`adjoint`, `event`, `max_steps`. Equinox does not check this when the class is
defined; it raises on first construction:

```py
TightSolver(dfx.Dopri5())
# TypeError: Can't instantiate abstract class TightSolver
#            with abstract attributes {'event'}
```

So a missing field survives import and every type check, and fails only when
something actually builds one. Keep `max_steps` `static=True` — it is a
compile-time constant everywhere it is used.

## Things it does not do

- **`evaluate(t0, t1)` is `evaluate(t1) - evaluate(t0)`.** That is two full
  vmapped passes, not a cheaper increment; and it needs `y` to support `-`, so a
  PyTree `y0` fails even though single-time evaluation of the same solution
  works:

  ```py
  soln = solver(pytree_term, ..., y0={"a": 1.0}, vectorize_interpolation=True)
  t0, t1 = ts, 0.0

  soln.evaluate(t0)  # fine: {'a': (4,)}
  soln.evaluate(t0, t1)  # -> evaluate(t1) - evaluate(t0)
  # TypeError: unsupported operand type(s) for -: 'dict' and 'dict'
  ```

- **`t0` and `t1` are batched arrays**, of shape `batch_shape` — not the scalars
  a `diffrax.AbstractPath` normally has. Diffrax code that assumes they are
  scalar (using the wrapper as a control path, for instance) will not behave.

## Troubleshooting

| Symptom                                                                | Cause / fix                                                                                                                                            |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `soln.evaluate(array_of_times)` fails on a normal diffrax solution     | That is plain `diffrax.DenseInterpolation` — one time at a time. Wrap it: `vectorize_interpolation=True` or `apply_to_solution`.                       |
| Output has one dimension too many or too few                           | Re-read the shape contract: `(*batch_shape, *jnp.shape(t), *jnp.shape(y0))`. A scalar `t` contributes nothing; an unbatched solve contributes nothing. |
| `vmap got inconsistent sizes for array axes to be mapped`              | A hand-passed `batch_shape` whose ndim doesn't match the solve's. Let it be inferred.                                                                  |
| `batch_shape == ()` on a solve you thought was batched                 | It is read from `t0_if_trivial`, so it reflects `vmap`ping the _solve_. A vector `y0` is not a batch.                                                  |
| `ValueError: max_steps=None is incompatible with saving at steps=True` | Pass a concrete `max_steps` for that call. The field is per-solver; the override is per-call.                                                          |
| `TypeError: unsupported operand type(s) for -`                         | `evaluate(t0, t1)` on a PyTree `y0`. Evaluate twice and combine yourself.                                                                              |
| Recompiling on every call                                              | A new `max_steps` each time (static), or a fresh solver/controller object each time. Build the `DiffEqSolver` once.                                    |
| `TypeError: Cannot convert <class ...> to <class ...>`                 | `from_` on an instance of a different (sub)class. The identity overload requires `type(obj) is cls`.                                                   |

## Version notes

diffraxtra reads `diffrax.diffeqsolve`'s defaults out of its signature at import
time, so `max_steps=4096`, the default `adjoint`, and the default
`stepsize_controller` are diffrax's values, and change when diffrax changes.
`diffrax>=0.6` is the supported floor and is tested in CI; minimum supported
versions follow [SPEC 0](https://scientific-python.org/specs/spec-0000/).
