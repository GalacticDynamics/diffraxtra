"""Regression tests for reducing dense solution storage."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffraxtra import VectorizedDenseInterpolation, subsample


def solve(t0=0.0, t1=3.0, y0=1.0, *, dense=True):
    """Solve a smooth equation with deliberately oversized dense buffers."""
    return dfx.diffeqsolve(
        dfx.ODETerm(lambda _t, y, _args: jax.tree.map(lambda x: -x, y)),
        dfx.Dopri5(),
        t0,
        t1,
        jnp.where(t1 >= t0, 0.01, -0.01),
        y0,
        saveat=dfx.SaveAt(t0=True, t1=True, dense=dense),
        max_steps=512,
    )


@pytest.mark.parametrize("backward", [False, True])
def test_dense_accuracy_and_storage(backward):
    """Keep endpoints and metadata while actually reducing interpolation storage."""
    sol = solve(3.0, 0.0) if backward else solve()
    small = subsample(sol, max_steps=32)
    assert isinstance(small, dfx.Solution)
    assert small is not sol
    assert small.interpolation.ts.shape == (33,)
    assert int(small.interpolation.ts_size) == 33
    assert sol.interpolation.ts.shape == (513,)
    assert small.ts is sol.ts
    assert small.ys is sol.ys
    assert eqx.tree_equal(small.stats, sol.stats)
    assert small.result == sol.result
    assert small.interpolation.t0 == sol.interpolation.t0
    assert small.interpolation.t1 == sol.interpolation.t1
    times = jnp.linspace(sol.t0, sol.t1, 503)
    np.testing.assert_allclose(
        jax.vmap(small.evaluate)(times), jax.vmap(sol.evaluate)(times), rtol=3e-6
    )
    for t in [sol.t0, sol.t1, 1.5]:
        np.testing.assert_allclose(small.evaluate(t), sol.evaluate(t), rtol=1e-12)
        np.testing.assert_allclose(small.derivative(t), sol.derivative(t), rtol=1e-10)
    np.testing.assert_allclose(
        small.evaluate(0.2, 0.8), small.evaluate(0.8) - small.evaluate(0.2)
    )
    assert jnp.isnan(small.evaluate(-1.0))
    assert all(x.shape[0] == 32 for x in jax.tree.leaves(small.interpolation.infos))


def test_pytree_jit_and_grad():
    """Resampling supports structured states, compilation, and autodiff."""
    sol = solve(y0={"a": jnp.array([1.0, 2.0]), "b": (jnp.array(3.0),)})
    small = eqx.filter_jit(lambda s: subsample(s, max_steps=16))(sol)
    for actual, expected in zip(
        jax.tree.leaves(small.evaluate(0.7)),
        jax.tree.leaves(sol.evaluate(0.7)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
    derivative = jax.grad(lambda t: small.evaluate(t)["b"][0])(0.7)
    np.testing.assert_allclose(derivative, -3 * jnp.exp(-0.7), rtol=1e-4)


def test_vectorized_batch():
    """Flattened vectorized interpolation retains its batch and time semantics."""
    sol = jax.vmap(solve)(jnp.zeros((2,)), jnp.full((2,), 3.0), jnp.array([1.0, 2.0]))
    sol = VectorizedDenseInterpolation.apply_to_solution(sol)
    small = eqx.filter_jit(lambda s: subsample(s, max_steps=32))(sol)
    assert isinstance(small.interpolation, VectorizedDenseInterpolation)
    assert small.interpolation.ts.shape == (2, 33)
    times = jnp.linspace(0, 3, 20).reshape(4, 5)
    assert small.evaluate(times).shape == (2, 4, 5)
    np.testing.assert_allclose(small.evaluate(times), sol.evaluate(times), rtol=3e-6)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_size(size):
    """A fixed positive integer is required for output shapes."""
    with pytest.raises((TypeError, ValueError), match="max_steps"):
        subsample(solve(), max_steps=size)


def test_requires_dense():
    """Missing dense output is diagnosed explicitly."""
    with pytest.raises(ValueError, match="dense"):
        subsample(solve(dense=False), max_steps=10)


def test_trivial_interval():
    """A zero-duration solve retains its sole defined value."""
    sol = solve(1.0, 1.0)
    small = subsample(sol, max_steps=4)
    np.testing.assert_array_equal(small.evaluate(1.0), sol.evaluate(1.0))
    assert jnp.isnan(small.evaluate(0.0))


def test_vmap_subsample():
    """Resampling can be vmapped over ordinary dense solutions."""
    sol = jax.vmap(lambda y: subsample(solve(y0=y), max_steps=8))(jnp.array([1.0, 2.0]))
    assert sol.interpolation.ts.shape == (2, 9)


@pytest.mark.parametrize("solver", [dfx.Tsit5(), dfx.Dopri8(), dfx.Euler()])
def test_adaptive_and_other_interpolants(solver):
    """Resampling reads each solver's curve rather than reusing its coefficients."""
    sol = dfx.diffeqsolve(
        dfx.ODETerm(lambda _t, y, _args: -y),
        solver,
        0.0,
        2.0,
        0.01,
        1.0,
        saveat=dfx.SaveAt(dense=True),
        stepsize_controller=(
            dfx.ConstantStepSize()
            if isinstance(solver, dfx.Euler)
            else dfx.PIDController(rtol=1e-5, atol=1e-7)
        ),
    )
    small = subsample(sol, max_steps=64)
    times = jnp.linspace(0, 2, 201)
    np.testing.assert_allclose(
        jax.vmap(small.evaluate)(times), jax.vmap(sol.evaluate)(times), atol=3e-5
    )


def test_convergence():
    """Cubic resampling error decreases as the sampling grid is refined."""
    sol = solve()
    times = jnp.linspace(0, 3, 501)
    reference = jax.vmap(sol.evaluate)(times)
    errors = [
        jnp.max(
            jnp.abs(jax.vmap(subsample(sol, max_steps=n).evaluate)(times) - reference)
        )
        for n in [8, 16, 32]
    ]
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10


def test_event_uses_actual_interval():
    """An early termination samples only the interval actually integrated."""
    sol = dfx.diffeqsolve(
        dfx.ODETerm(lambda _t, y, _args: -y),
        dfx.Dopri5(),
        0.0,
        10.0,
        0.1,
        1.0,
        saveat=dfx.SaveAt(dense=True),
        event=dfx.Event(lambda _t, y, _args, **_kwargs: y < 0.5),
    )
    small = subsample(sol, max_steps=16)
    end = sol.interpolation.t1
    assert end < sol.t1
    assert small.interpolation.t1 == end
    np.testing.assert_allclose(small.evaluate(end), sol.evaluate(end), rtol=1e-12)


def test_one_interval():
    """The smallest valid output still interpolates the whole domain."""
    sol = solve()
    small = subsample(sol, max_steps=1)
    assert small.interpolation.ts.shape == (2,)
    for t in [0.0, 3.0]:
        np.testing.assert_allclose(small.evaluate(t), sol.evaluate(t), rtol=1e-12)


def test_trim_is_exact():
    """A large enough budget preserves original knots, coefficients, and curve."""
    sol = solve()
    small = eqx.filter_jit(lambda s: subsample(s, max_steps=320))(sol)
    assert small.interpolation.ts.shape == (321,)
    assert small.interpolation.ts_size == sol.interpolation.ts_size
    np.testing.assert_array_equal(small.interpolation.ts, sol.interpolation.ts[:321])
    times = jnp.linspace(0, 3, 401)
    np.testing.assert_array_equal(
        jax.vmap(small.evaluate)(times), jax.vmap(sol.evaluate)(times)
    )


def test_mixed_trim_and_resample_batch():
    """One compiled batch can trim one member and resample another."""

    @jax.vmap
    def solve_variable(end):
        sol = solve(t1=end)
        return subsample(sol, max_steps=32)

    small = eqx.filter_jit(solve_variable)(jnp.array([0.1, 3.0]))
    assert int(small.interpolation.ts_size[0]) < 33
    assert int(small.interpolation.ts_size[1]) == 33
    values = jax.vmap(lambda s: s.evaluate(0.05))(small)
    np.testing.assert_allclose(values, jnp.exp(-0.05), rtol=3e-6)


@pytest.mark.parametrize("budget", [32, 400])
def test_storage_bytes(budget):
    """A compact interpolation retains neither a reference nor a second payload."""
    sol = solve()
    small = subsample(sol, max_steps=budget)
    original = jax.tree.leaves(sol.interpolation)
    compact = jax.tree.leaves(small.interpolation)
    assert sum(x.nbytes for x in compact) < sum(x.nbytes for x in original)
    assert not any(x is y for x in compact for y in original if y.ndim > 0)


def test_repeated_subsample():
    """A previously compacted curve can be compacted again."""
    small = subsample(subsample(solve(), max_steps=320), max_steps=32)
    np.testing.assert_allclose(small.evaluate(0.7), jnp.exp(-0.7), rtol=3e-6)


def test_gradient_through_resampling():
    """Compaction remains differentiable with respect to the original solution."""
    grad = jax.grad(lambda y: subsample(solve(y0=y), max_steps=16).evaluate(0.7))(1.0)
    np.testing.assert_allclose(grad, jnp.exp(-0.7), rtol=1e-5)


def test_vectorized_mixed_modes():
    """The vectorized wrapper also supports mixed exact and approximate members."""
    sol = jax.vmap(lambda end: solve(t1=end))(jnp.array([0.1, 3.0]))
    sol = VectorizedDenseInterpolation.apply_to_solution(sol)
    small = eqx.filter_jit(lambda s: subsample(s, max_steps=32))(sol)
    assert int(small.interpolation.ts_size[0]) < 33
    assert int(small.interpolation.ts_size[1]) == 33
    np.testing.assert_allclose(small.evaluate(0.05), sol.evaluate(0.05), rtol=3e-6)


@pytest.mark.filterwarnings("ignore:Complex dtype support in Diffrax:UserWarning")
@pytest.mark.parametrize("budget", [16, 320])
def test_float32_and_complex(budget):
    """Packing preserves state dtypes, including complex-valued states."""
    for dtype in [jnp.float32, jnp.complex64]:
        sol = solve(y0=jnp.asarray(1.0, dtype=dtype))
        small = subsample(sol, max_steps=budget)
        actual = small.evaluate(0.7)
        assert actual.dtype == sol.evaluate(0.7).dtype
        np.testing.assert_allclose(actual, sol.evaluate(0.7), rtol=1e-5)


def test_larger_budget_is_noop():
    """A request above the original capacity does not grow the buffers."""
    sol = solve()
    small = subsample(sol, max_steps=1024)
    assert small.interpolation.ts.shape == sol.interpolation.ts.shape
    np.testing.assert_array_equal(small.evaluate(0.7), sol.evaluate(0.7))
