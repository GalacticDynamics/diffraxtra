"""Compare dense storage and interpolation error for two galax potentials.

Run with galax, matplotlib, and this checkout of diffraxtra installed::

    python examples/subsample_orbits.py --output /tmp/subsample-orbits.png

The ODE uses kpc, Myr, and kpc/Myr, the galax galactic unit system. The
reference is the original dense solution; plotted errors measure the extra
error introduced by resampling, rather than the integration error itself.
"""

import argparse
import functools as ft
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import galax.potential as gp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import unxt as u

from diffraxtra import subsample


def storage_bytes(tree: Any) -> int:
    """Count array storage in the dense interpolation only."""
    return sum(x.nbytes for x in jax.tree.leaves(tree) if eqx.is_array(x))


def main(output: Path) -> None:
    """Integrate circular and eccentric orbits and check resampling convergence."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    potentials = {
        "Harmonic oscillator": gp.HarmonicOscillatorPotential(
            omega=u.Q(0.03, "1/Myr"), units="galactic"
        ),
        "Kepler": gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, (name, potential) in enumerate(potentials.items()):
        radius = 8.0  # kpc
        acceleration = u.ustrip(
            "kpc/Myr2",
            potential.acceleration(u.Q([radius, 0.0, 0.0], "kpc"), t=u.Q(0.0, "Myr")),
        )
        circular_speed = jnp.sqrt(-radius * acceleration[0])
        # An eccentric Kepler orbit exercises nonuniform orbital speeds.
        speed = circular_speed * (0.8 if name == "Kepler" else 1.0)
        period = 2 * jnp.pi * radius / circular_speed
        end = float(period)

        def vector_field(t: jax.Array, y: jax.Array, potential: Any) -> jax.Array:
            force = potential.acceleration(u.Q(y[:3], "kpc"), t=u.Q(t, "Myr"))
            return jnp.concatenate((y[3:], u.ustrip("kpc/Myr2", force)))

        sol = dfx.diffeqsolve(
            dfx.ODETerm(vector_field),
            dfx.Dopri8(),
            0.0,
            end,
            end / 1000,
            jnp.array([radius, 0.0, 0.0, 0.0, speed, 0.0]),
            args=potential,
            saveat=dfx.SaveAt(dense=True),
            stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-12),
            max_steps=4096,
        )
        times = jnp.linspace(0.0, end, 2001)
        reference = jax.vmap(sol.evaluate)(times)
        original_bytes = storage_bytes(sol.interpolation)
        accepted = int(sol.stats["num_accepted_steps"])
        axes[row, 0].plot(
            reference[:, 0], reference[:, 1], label="original", color="black"
        )
        print(f"{name}: {accepted} accepted steps, {original_bytes:,} dense bytes")  # noqa: T201
        previous_error = np.inf
        for count in (8, 16, 32, 64):
            small = eqx.filter_jit(ft.partial(subsample, max_steps=count))(sol)
            values = jax.vmap(small.evaluate)(times)
            errors = jnp.linalg.norm(values[:, :3] - reference[:, :3], axis=-1)
            error = float(jnp.max(errors))
            np.testing.assert_array_less(error, previous_error + 1e-12)
            previous_error = error
            reduced_bytes = storage_bytes(small.interpolation)
            np.testing.assert_array_less(reduced_bytes, original_bytes)
            if count >= accepted:
                np.testing.assert_array_equal(values, reference)
            else:
                np.testing.assert_allclose(
                    values[jnp.array([0, -1]), :],
                    reference[jnp.array([0, -1]), :],
                    atol=1e-12,
                )
            axes[row, 0].plot(
                values[:, 0], values[:, 1], "--", label=f"{count} intervals"
            )
            axes[row, 1].semilogy(
                times, jnp.maximum(errors, 1e-15), label=f"{count} intervals"
            )
            print(  # noqa: T201
                f"  {count:2d} intervals: {reduced_bytes:7,d} bytes "
                f"({original_bytes / reduced_bytes:.1f}x smaller), "
                f"max position error {error:.3e} kpc"
            )
        axes[row, 0].set(title=name, xlabel="x [kpc]", ylabel="y [kpc]", aspect="equal")
        axes[row, 1].set(
            title=f"{name}: error relative to dense reference",
            xlabel="t [Myr]",
            ylabel="position error [kpc]",
        )
        axes[row, 0].legend(fontsize="small")
        axes[row, 1].legend(fontsize="small")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("subsample-orbits.png"))
    main(parser.parse_args().output)
