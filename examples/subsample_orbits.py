"""Compare dense storage and interpolation error for two analytic potentials.

Run with matplotlib and this checkout of diffraxtra installed::

    uv run --with matplotlib python examples/subsample_orbits.py

Positions use kpc, times Myr, and velocities kpc/Myr. Potential values are in
kpc²/Myr². Plotted errors measure the additional error from subsampling,
relative to the original dense solution.
"""

import argparse
import functools as ft
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from diffraxtra import subsample

Potential = Callable[[jax.Array], jax.Array]
BUDGETS = (8, 16, 32, 64)


def harmonic_potential(position: jax.Array, *, omega: float = 0.03) -> jax.Array:
    """Return Φ = ω²r²/2, with ω in Myr⁻¹."""
    return 0.5 * omega**2 * jnp.sum(position**2)


def kepler_potential(position: jax.Array, *, mu: float = 0.45) -> jax.Array:
    """Return Φ = -μ/r, with μ = GM in kpc³/Myr²."""
    return -mu / jnp.linalg.norm(position)


def vector_field(_t: jax.Array, state: jax.Array, potential: Potential) -> jax.Array:
    """Evolve position and velocity using acceleration = -∇Φ."""
    return jnp.concatenate((state[3:], -jax.grad(potential)(state[:3])))


def solve_orbit(potential: Potential, *, speed_fraction: float = 1.0) -> dfx.Solution:
    """Integrate from 8 kpc for one circular-orbit period of the potential."""
    position = jnp.array([8.0, 0.0, 0.0])
    circular_speed = jnp.sqrt(position[0] * jax.grad(potential)(position)[0])
    period = float(2 * jnp.pi * position[0] / circular_speed)
    velocity = jnp.array([0.0, speed_fraction * circular_speed, 0.0])
    return dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Dopri8(),
        t0=0.0,
        t1=period,
        dt0=period / 1000,
        y0=jnp.concatenate((position, velocity)),
        args=potential,
        saveat=dfx.SaveAt(dense=True),
        stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=4096,
    )


def storage_bytes(tree: Any) -> int:
    """Count array storage in the dense interpolation only."""
    return sum(x.nbytes for x in jax.tree.leaves(tree) if eqx.is_array(x))


def compare_solution(
    sol: dfx.Solution, budget: int, times: jax.Array, reference: jax.Array
) -> tuple[jax.Array, jax.Array, int]:
    """Check a compact solution and return its values, position errors, and bytes."""
    small = eqx.filter_jit(ft.partial(subsample, max_steps=budget))(sol)
    values = jax.vmap(small.evaluate)(times)
    errors = jnp.linalg.norm(values[:, :3] - reference[:, :3], axis=-1)
    size = storage_bytes(small.interpolation)
    np.testing.assert_array_less(size, storage_bytes(sol.interpolation))
    np.testing.assert_allclose(
        values[jnp.array([0, -1])], reference[jnp.array([0, -1])], rtol=0, atol=1e-12
    )
    if budget >= int(sol.stats["num_accepted_steps"]):
        roundoff = 32 * np.finfo(reference.dtype).eps
        np.testing.assert_allclose(
            values,
            reference,
            rtol=roundoff,
            atol=roundoff * float(jnp.max(jnp.abs(reference))),
        )
    return values, errors, size


def plot_orbit(axes: tuple[Axes, Axes], name: str, sol: dfx.Solution) -> None:
    """Plot an orbit and its subsampling errors, checking convergence."""
    orbit_ax, error_ax = axes
    times = jnp.linspace(sol.t0, sol.t1, 2001)
    reference = jax.vmap(sol.evaluate)(times)
    original_bytes = storage_bytes(sol.interpolation)
    accepted = int(sol.stats["num_accepted_steps"])
    orbit_ax.plot(reference[:, 0], reference[:, 1], label="original", color="black")
    print(f"{name}: {accepted} accepted steps, {original_bytes:,} dense bytes")  # noqa: T201
    previous_error = np.inf
    for budget in BUDGETS:
        values, errors, size = compare_solution(sol, budget, times, reference)
        error = float(jnp.max(errors))
        np.testing.assert_array_less(error, previous_error + 1e-12)
        previous_error = error
        label = f"{budget} intervals"
        orbit_ax.plot(values[:, 0], values[:, 1], "--", label=label)
        error_ax.semilogy(times, jnp.maximum(errors, 1e-15), label=label)
        print(  # noqa: T201
            f"  {budget:2d} intervals: {size:7,d} bytes "
            f"({original_bytes / size:.1f}x smaller), "
            f"max position error {error:.3e} kpc"
        )
    orbit_ax.set(title=name, xlabel="x [kpc]", ylabel="y [kpc]", aspect="equal")
    error_ax.set(
        title=f"{name}: error relative to dense reference",
        xlabel="t [Myr]",
        ylabel="position error [kpc]",
    )
    for ax in axes:
        ax.legend(fontsize="small")


def main(output: Path) -> None:
    """Render the comparisons using a circular harmonic and eccentric Kepler orbit."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    plot_orbit(tuple(axes[0]), "Harmonic oscillator", solve_orbit(harmonic_potential))
    plot_orbit(
        tuple(axes[1]), "Kepler", solve_orbit(kepler_potential, speed_fraction=0.8)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("subsample-orbits.png"))
    main(parser.parse_args().output)
