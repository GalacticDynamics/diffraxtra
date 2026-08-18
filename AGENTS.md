# diffraxtra — Agent Instructions

`diffraxtra` is two extras for [`diffrax`](https://docs.kidger.site/diffrax/):
`DiffEqSolver`, an `equinox.Module` that pre-configures `diffrax.diffeqsolve`
behind a callable object, and `VectorizedDenseInterpolation`, which makes the
`diffrax.DenseInterpolation` produced by a batched solve evaluable at
arbitrarily-shaped arrays of times.

The design goal is to extend diffrax without diverging from it: follow diffrax's
own API conventions, keep dependencies minimal, and keep every object a PyTree
that survives `jit`, `vmap` and `grad`. When in doubt, do what diffrax does.

For _using_ diffraxtra from outside this repo — the shape contract, the two call
forms, the `from_` constructors, subclassing the ABCs — read
[skills/diffraxtra/SKILL.md](skills/diffraxtra/SKILL.md). This file is for
working inside the repo.

## Essential Commands

```bash
uv run pytest                # the whole suite (almost entirely doctests)
uv run nox -s test           # the same, in an isolated env (chains to `pytest`)
uv run nox -s lint           # chains to `precommit`, `pylint`, and `mypy`
uv run nox -s precommit      # pre-commit on all files
uv run nox -s pylint         # pylint over src/
uv run nox -s mypy           # mypy over src/
uv run nox -s build          # sdist + wheel
```

Always use `uv run` — never bare `python`/`pytest`. The noxfile uses
[`nox-uv`](https://github.com/dantebben/nox-uv): each session declares
`uv_groups=[...]` matching a PEP-735 dependency group in `pyproject.toml`, so a
new session needs its group to exist.

## Architecture

The public API is the four names re-exported from
[src/diffraxtra/\_\_init\_\_.py](src/diffraxtra/__init__.py); everything else
lives under `src/diffraxtra/_src/` and is private. Each public class is an
abstract/`@final` pair of `equinox.Module`s, with `eqx.AbstractVar` fields on
the ABC.

| Module                                                   | Provides                                                                                                                                                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [\_src/diffeq_abc.py](src/diffraxtra/_src/diffeq_abc.py) | `AbstractDiffEqSolver`: the five `AbstractVar` fields, the plum-dispatched `__call__` pair that forwards to `dfx.diffeqsolve`, the `params` signature scrape, and four `from_` overloads                           |
| [\_src/diffeq.py](src/diffraxtra/_src/diffeq.py)         | `DiffEqSolver`: the `@final` concrete module, plus the `default_stepsize_controller` / `default_max_steps` / `default_adjoint` re-exports                                                                          |
| [\_src/interp.py](src/diffraxtra/_src/interp.py)         | `AbstractVectorizedDenseInterpolation` and `VectorizedDenseInterpolation`: the batch-flattening `__init__`, the doubly-vmapped `evaluate`, the `DenseInterpolation` property forwards, and three `from_` overloads |
| [conftest.py](conftest.py)                               | sybil collection, x64, and the `DIFFRAX_LT_070` skip machinery                                                                                                                                                     |

`__call__` is **two** methods: a `@dispatch`ed positional one on the class, and
a keyword-only overload registered underneath as
`@AbstractDiffEqSolver.__call__.dispatch`, which pops
`t0`/`t1`/`dt0`/`y0`/`args` out of `**kwargs` and re-enters the positional one.
Both are separately `eqx.filter_jit`-wrapped. That is why
`solver(term, t0=..., t1=..., y0=...)` works without `dt0`.

## Defaults are read out of diffrax at import time

[diffeq_abc.py](src/diffraxtra/_src/diffeq_abc.py) does not hard-code
`diffeqsolve`'s defaults. It scrapes them:

```py
params = inspect.signature(dfx.diffeqsolve.__wrapped__).parameters
```

`.__wrapped__` exists only because diffrax wraps `diffeqsolve` in
`eqx.filter_jit`. If upstream stops wrapping it, renames a parameter, or drops
one, **`import diffraxtra` fails** — at import, not at some downstream call
site. `default_saveat`, `default_progress_meter`, `default_event`,
`default_throw` (here) and `default_stepsize_controller`, `default_max_steps`,
`default_adjoint` (in [diffeq.py](src/diffraxtra/_src/diffeq.py)) all come from
this one line, so diffraxtra's "defaults" are whatever the installed diffrax
says they are. A doctest printing `max_steps=4096` is asserting on upstream, not
on us.

## Doctests are the test suite

`tests/` holds exactly one test — the version check. Everything else collected
is an example run by [sybil](https://sybil.readthedocs.io) from
[conftest.py](conftest.py); `testpaths` is `README.md`, `src/`, `skills/`,
`tests/`.

- **A new public function, field, or `from_` overload needs a doctest**, or it
  ships untested and uncovered. There is no unit-test layer to fall back on.
- `>>>` examples in `src/**.py` and `README.md` are parsed as reST doctests.
  ` ```python ` blocks in Markdown (i.e.
  [skills/diffraxtra/SKILL.md](skills/diffraxtra/SKILL.md)) are executed by the
  MyST parser as plain code, so a claim made there only becomes a test if you
  write it as an `assert`. Fence a block ` ```py ` to have it collected but not
  run — use that for examples that are meant to fail or are pseudocode.
- **diffrax 0.6 and 0.7 print different `repr`s**, and the suite handles this in
  two different ways. `conftest.py` computes `DIFFRAX_LT_070` and puts it in the
  sybil namespace, so a `src/` docstring can guard one example with
  `.. skip: next if(DIFFRAX_LT_070, reason="...")`; `README.md` is not guardable
  that way, so `pytest_collection_modifyitems` **drops every README.md item
  wholesale** when diffrax < 0.7. A repr-printing example added to `src/` needs
  the skip directive, or CI's oldest-dependency job goes red.
- x64 is enabled in `conftest.py`, which is why the examples print `f64[…]`.
- `filterwarnings = ["error"]` — a new warning anywhere is a test failure.
- Output is matched with `ELLIPSIS | NORMALIZE_WHITESPACE`; that is what makes
  the `Solution( t0=f64[], ... )` style of eliding work.

## Pitfalls

- **Two fields use a `MISSING` sentinel, and the sentinel is the point.**
  `__call__` defaults both `max_steps` and `event` to `dataclasses.MISSING`, so
  "not passed" (use the field) stays distinguishable from an explicit `None`
  (unbounded / no event). Never "simplify" either default to `None`. `max_steps`
  is additionally `static=True`, so each distinct value is a separate
  compilation.
- **`__call__` is `eqx.filter_jit`-wrapped, twice.** `throw`,
  `vectorize_interpolation` and `max_steps` are static; a structurally different
  `solver` / `stepsize_controller` / `adjoint` retraces. Both dispatch overloads
  need the decorator — the keyword one is jitted independently of the positional
  one it delegates to.
- **The classes are not `strict=True`.** Equinox still enforces the
  `AbstractVar` contract, but only at **instantiation**: a subclass that forgets
  a field defines fine and then raises
  `TypeError: Can't instantiate abstract class ... with abstract attributes {'event'}`
  on first construction. Adding an `AbstractVar` to `AbstractDiffEqSolver` is
  therefore a breaking change for every downstream subclass, deferred to
  runtime.
- **`VectorizedDenseInterpolation.y0_shape` is computed, stored, and never
  read.** Nothing in the package consumes it. It is public and has doctests, so
  don't delete it casually — but don't assume it is validated or load-bearing
  either.
- **The `from_` overloads are module-bottom side effects.** They all shadow one
  name, so `# type: ignore[no-redef]` and ruff's `F811` ignore are load-bearing,
  not noise. Note the asymmetry: solver overloads register on the ABC
  (`AbstractDiffEqSolver.from_`), interpolation overloads register on the
  concrete `VectorizedDenseInterpolation`.
- **`jax` and `numpy` are imported in `src/` but not declared** in
  `[project.dependencies]` (which lists `diffrax`, `equinox`, `jaxtyping`,
  `plum-dispatch`, `typing_extensions`). They arrive transitively.
- **CI runs `--resolution lowest-direct`**, so new code must work on
  `diffrax>=0.6`, not just the newest release.
- **Commits use gitmoji** plus conventional commits (`cz_gitmoji`): `✨ feat:`,
  `🐛 fix:`, `📝 docs:`, `💥 boom:`. Match the existing log.
- **Import aliases are enforced by ruff**: `dfx`, `eqx`, `np`. Fields are
  documented with `#:` comments above them, not docstrings below. pylint runs
  over `src/`, so a few `# pylint: disable=` comments are load-bearing too.

## Dependencies

`diffrax>=0.6`, `equinox>=0.11.5`, `jaxtyping>=0.2.35`, `plum-dispatch>=2.5.7`,
`typing_extensions>=4.12.2`; Python >=3.11. Minimum supported versions follow
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

## Further Reading

- [skills/diffraxtra/SKILL.md](skills/diffraxtra/SKILL.md) — using diffraxtra:
  the shape contract, the two call forms, the constructors, subclassing,
  troubleshooting
- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) —
  what to look for when reviewing a diffraxtra change
- [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) — environment setup
- [README.md](README.md) — install and quick start
- [diffrax docs](https://docs.kidger.site/diffrax/) — the library being extended
