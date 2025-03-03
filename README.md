<h1 align='center'> diffraxtra </h1>
<h3 align="center"><code>diffrax</code> extras</h3>

<p align="center">
    <a href="https://pypi.org/project/diffraxtra/"> <img alt="PyPI: diffraxtra" src="https://img.shields.io/pypi/v/diffraxtra?style=flat" /> </a>
    <a href="https://pypi.org/project/diffraxtra/"> <img alt="PyPI versions: diffraxtra" src="https://img.shields.io/pypi/pyversions/diffraxtra" /> </a>
    <a href="https://pypi.org/project/diffraxtra/"> <img alt="diffraxtra license" src="https://img.shields.io/github/license/GalacticDynamics/diffraxtra" /> </a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/diffraxtra/actions"> <img alt="CI status" src="https://github.com/GalacticDynamics/diffraxtra/workflows/CI/badge.svg" /> </a>
    <a href="https://codecov.io/gh/GalacticDynamics/diffraxtra"> <img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/diffraxtra/graph/badge.svg" /> </a>
    <a href="https://scientific-python.org/specs/spec-0000/"> <img alt="ruff" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /> </a>
    <a href="https://docs.astral.sh/ruff/"> <img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /> </a>
    <a href="https://pre-commit.com"> <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /> </a>
</p>

---

Extras for [diffrax][diffrax-link].

- `DiffEqSolver`: an object-oriented interface to `diffrax.diffeqsolve`.
- `VectorizedDenseInterpolation`: a vectorized form of
  `diffrax.DenseInterpolation` that works on batched results from
  `diffrax.diffeqsolve`.

For example,

<!-- invisible-code-block: python
import jax
jax.config.update("jax_enable_x64", True)
-->

```python
import jax.numpy as jnp
import diffrax as dfx
from diffraxtra import DiffEqSolver

# Construct a solver object.
solver = DiffEqSolver(dfx.Dopri5(),
                      stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

# And a differential equation to solve.
term = dfx.ODETerm(lambda t, y, args: -y)

# Then solve the differential equation.
saveat = dfx.SaveAt(t1=True, dense=True)
soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
              vectorize_interpolation=True)

print(soln)
# Solution(
#   t0=f32[], t1=f32[], ts=f32[1],
#   ys=f32[1],
#   interpolation=VectorizedDenseInterpolation(
#     scalar_interpolation=DenseInterpolation( ... ),
#     batch_shape=(),
#     y0_shape=()
#   ),
#   ...
# )

soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
# Array([[0.90483742, 0.81872516],
#         [0.74080871, 0.67031456]], dtype=float64)

```

## Installation

[![PyPI platforms][pypi-platforms]][pypi-link]
[![PyPI version][pypi-version]][pypi-link]

```bash
pip install diffraxtra
```

## Documentation

### `DiffEqSolver`

```pycon
>>> import jax.numpy as jnp
>>> import diffrax as dfx
>>> from diffraxtra import DiffEqSolver

```

Construct a solver object.

```pycon
>>> solver = DiffEqSolver(dfx.Dopri5(),
...                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

```

And a differential equation to solve.

```pycon
>>> term = dfx.ODETerm(lambda t, y, args: -y)

```

Then solve the differential equation.

```pycon
>>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1)
>>> soln
Solution( t0=f64[], t1=f64[], ts=f64[1],
          ys=f64[1], ... )

```

The solution can be saved at specific times.

```pycon
>>> saveat = dfx.SaveAt(ts=[0., 1., 2., 3.])
>>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
>>> soln
Solution( t0=f64[], t1=f64[], ts=f64[4],
          ys=f64[4], ... )

```

The solution can be densely interpolated.

```pycon
>>> saveat = dfx.SaveAt(t1=True, dense=True)
>>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
>>> soln
Solution( t0=f64[], t1=f64[], ts=f64[1],
          ys=f64[1], ... )
>>> soln.evaluate(0.5).round(3)
Array(0.607, dtype=float64)

```

Using the `VectorizedDenseInterpolation` class, the interpolation can be
vectorized, enabling evaluation of batched solutions over batches of times.

```pycon
>>> from diffraxtra import VectorizedDenseInterpolation
>>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat)
>>> soln = VectorizedDenseInterpolation.apply_to_solution(soln)
>>> soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
Array([[0.90483742, 0.81872516],
       [0.74080871, 0.67031456]], dtype=float64)

```

This can be more conveniently done using the `vectorize_interpolation` argument.

```pycon
>>> soln = solver(term, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
...               vectorize_interpolation=True)
>>> soln.evaluate(jnp.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2))
Array([[0.90483742, 0.81872516],
       [0.74080871, 0.67031456]], dtype=float64)

```

There are many ways to construct a `DiffEqSolver` object. For example, we can
can make a new one from an existing `DiffEqSolver` object

```pycon
>>> solver = DiffEqSolver(dfx.Dopri5())
>>> DiffEqSolver.from_(solver) is solver
True

```

From a `diffrax.AbstractSolver` object.

```pycon
>>> solver = DiffEqSolver.from_(dfx.Dopri5())
>>> solver
DiffEqSolver(solver=Dopri5())

```

(Where all other arguments are their default values and printed only if
changed.)

From a `collections.abc.Mapping`

```pycon
>>> solver = DiffEqSolver.from_({"solver": dfx.Dopri5(),
...       "stepsize_controller": dfx.PIDController(rtol=1e-5, atol=1e-5)})
>>> solver
DiffEqSolver(
  solver=Dopri5(), stepsize_controller=PIDController(rtol=1e-05, atol=1e-05)
)

```

For a full enumeration of the ways to construct a `DiffEqSolver` object, see
`diffraxtra.DiffEqSolver.from_`.

### `VectorizedDenseInterpolation`

Vectorized wrapper around a `diffrax.DenseInterpolation`

This also works on non-batched interpolations.

```pycon
>>> import jax
>>> import jax.numpy as jnp
>>> import diffrax as dfx

```

We'll start with a non-batched interpolation:

```pycon
>>> vector_field = lambda t, y, args: -y
>>> term = dfx.ODETerm(vector_field)
>>> solver = dfx.Dopri5()
>>> ts = jnp.array([0.0, 1, 2, 3])
>>> saveat = dfx.SaveAt(ts=ts, dense=True)
>>> stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

>>> sol = dfx.diffeqsolve(
...     term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
...     stepsize_controller=stepsize_controller)
>>> interp = VectorizedDenseInterpolation(sol.interpolation)
>>> interp
VectorizedDenseInterpolation(
  scalar_interpolation=DenseInterpolation(
    ts=f64[1,4097],
    ts_size=weak_i64[1],
    infos={'k': f64[1,4096,7], 'y0': f64[1,4096], 'y1': f64[1,4096]},
    interpolation_cls=<class 'diffrax._solver.dopri5._Dopri5Interpolation'>,
    direction=weak_i64[1],
    t0_if_trivial=f64[1],
    y0_if_trivial=f64[1]
  ),
  batch_shape=(),
  y0_shape=()
)

```

This can be evaluated by the normal means:

```pycon
>>> interp.evaluate(ts[-1])  # scalar evaluation
Array(0.04978961, dtype=float64)

```

It also works on arrays, without needed to manually apply `jax.vmap`:

```pycon
>>> interp.evaluate(ts)  # It works on arrays!
Array([1. , 0.36788338, 0.13533922, 0.04978961], dtype=float64)

```

```pycon
>>> interp.evaluate(ts, ts[0])  # t1 - t0 mixed scalar and array
Array([0. , 0.63211662, 0.86466078, 0.95021039], dtype=float64)

```

Better yet, the time array may be arbitrarily shaped:

```pycon
>>> interp.evaluate(ts.reshape(2, 2)).round(3)
Array([[1.   , 0.368],
       [0.135, 0.05 ]], dtype=float64)

```

As a convenience, we can also apply the `VectorizedDenseInterpolation` to the
solution to modify the interpolation "in-place" (when in a jitted context,
otherwise out-of-place, returning a copy):

```pycon
>>> sol = VectorizedDenseInterpolation.apply_to_solution(sol)
>>> isinstance(sol, dfx.Solution)
True
>>> isinstance(sol.interpolation, VectorizedDenseInterpolation)
True

```

Now we'll batch the interpolation:

```pycon
>>> @jax.vmap
... def solve(y0):
...     sol = dfx.diffeqsolve(
...         term, solver, t0=0, t1=3, dt0=0.1, y0=y0, saveat=saveat,
...         stepsize_controller=stepsize_controller)
...     return sol
>>> sol = solve(jnp.array([1, 2, 3]))
>>> interp = VectorizedDenseInterpolation(sol.interpolation)

```

```pycon
>>> interp.evaluate(ts[-1]).round(3)  # scalar eval of batched interp
Array([0.05 , 0.1  , 0.149], dtype=float64)

```

```pycon
>>> interp.evaluate(ts).astype(jnp.float64).round(3)  # array eval of batched interp
Array([[1.   , 0.368, 0.135, 0.05 ],
       [2.   , 0.736, 0.271, 0.1  ],
       [3.   , 1.104, 0.406, 0.149]], dtype=float64)

```

```pycon
>>> interp.evaluate(ts, ts[0]).round(3)  # mixed scalar and array eval
Array([[0.   , 0.632, 0.865, 0.95 ],
       [0.   , 1.264, 1.729, 1.9  ],
       [0.   , 1.896, 2.594, 2.851]], dtype=float64)

```

```pycon
>>> ys = interp.evaluate(ts.reshape(2, 2)).round(3)  # arbitrary shape eval
>>> ys
Array([[[1.   , 0.368],
        [0.135, 0.05 ]],
        [[2.   , 0.736],
        [0.271, 0.1  ]],
        [[3.   , 1.104],
        [0.406, 0.149]]], dtype=float64)
>>> ys.shape  # (batch, *times)
(3, 2, 2)

```

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you enjoyed using this library and would like to cite the software you use
then click the link above.

## Development

[![Actions Status][actions-badge]][actions-link]
[![codecov][codecov-badge]][codecov-link]
[![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link]
[![pre-commit][pre-commit-badge]][pre-commit-link]
[![ruff][ruff-badge]][ruff-link]

We welcome contributions!

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/diffraxtra/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/diffraxtra/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/diffraxtra/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/diffraxtra
[pre-commit-badge]:         https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit-link]:          https://pre-commit.com
[pypi-link]:                https://pypi.org/project/diffraxtra/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/diffraxtra
[pypi-version]:             https://img.shields.io/pypi/v/diffraxtra
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/DOI/10.5281/zenodo.14806581.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.14806581


[diffrax-link]: https://docs.kidger.site/diffrax/

<!-- prettier-ignore-end -->
