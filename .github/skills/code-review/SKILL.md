---
name: code-review
description:
  Use when reviewing a pull request or diff in the diffraxtra repository. Covers
  the diffraxtra-specific defects that generic review misses — a default
  hard-coded instead of scraped from diffrax's signature, an argument added to
  `__call__` but not forwarded, the `max_steps` MISSING sentinel collapsed to
  `None`, batch-shape and flattening invariants in the interpolation, `from_`
  overloads registered on the wrong class, and a new public name with no doctest
  in a repo whose test suite is doctests.
---

# Reviewing diffraxtra changes

diffraxtra is a thin layer over diffrax: `DiffEqSolver.__call__` forwards to
`diffrax.diffeqsolve`, and `VectorizedDenseInterpolation` rearranges the axes of
a `diffrax.DenseInterpolation`. Two properties follow, and they generate nearly
every real defect here:

- **Behaviour is defined upstream.** Defaults are read out of diffrax's
  signature at import time, arguments are forwarded by hand, and the
  interpolation reaches into `DenseInterpolation`'s field layout. The failure
  mode is drift from diffrax, not a logic error in a 130-line module.
- **Shape bookkeeping is manual and unvalidated.** Batch dimensions are
  flattened on construction and restored on evaluation by explicit `reshape`
  calls. Nothing checks them. A wrong shape becomes a wrong answer, or an error
  raised three frames from the cause.

## Scope of this review

Leave these alone — they are gated elsewhere:

- Formatting, lint, and types. ruff runs with `extend-select = ["ALL"]`, mypy is
  `strict` over `src`, and pylint runs too — all via
  [noxfile.py](../../../noxfile.py) and
  [.pre-commit-config.yaml](../../../.pre-commit-config.yaml) in CI. Don't
  restate an error the `format` job already catches.
- Generic security checklists. No user input, no network, no deserialisation of
  untrusted data.
- The numerical correctness of a solver, stepsize controller, or adjoint. That
  is diffrax's problem. diffraxtra only has to forward the arguments and get the
  axes right.

## What changed → what to check

| Change                                                      | Check                                                                 |
| ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `params`, `default_*`, or any literal default               | [Scraped defaults](#scraped-defaults)                                 |
| `__call__`'s signature or its `dfx.diffeqsolve(...)` call   | [Forwarding](#forwarding-to-diffeqsolve)                              |
| anything touching `max_steps` or `event`                    | [The MISSING sentinels](#the-missing-sentinels)                       |
| `interp.py`'s `__init__`, `evaluate`, or a property forward | [Shape bookkeeping](#shape-bookkeeping)                               |
| a new or edited `from_` overload                            | [`from_` overloads](#from_-overloads)                                 |
| a new field on any `Module`                                 | [Fields and the abstract contract](#fields-and-the-abstract-contract) |
| docstring examples, `conftest.py`, `testpaths`, `skills/`   | [Tests are doctests](#tests-are-doctests)                             |
| `pyproject.toml` dependencies, `ci.yml`                     | [Versions and dependencies](#versions-and-dependencies)               |

## Scraped defaults

`diffeqsolve`'s defaults are read out of its signature at import time — see
[AGENTS.md](../../../AGENTS.md#defaults-are-read-out-of-diffrax-at-import-time)
for the mechanism and why it is an import-time failure mode. Two things to check
in a diff:

- **A literal default is a regression.** `saveat=dfx.SaveAt(t1=True)` written
  out by hand, or `max_steps: int | None = 4096`, silently forks from diffrax
  the next time upstream changes. New defaults come from `params[...]`.
- The `default_*` names are in
  [diffeq.py](../../../src/diffraxtra/_src/diffeq.py)'s `__all__`. Removing or
  renaming one is a public API break even though the file is private.
- A change to the scrape itself deserves a note on which diffrax versions were
  checked, not just "works on mine".

## Forwarding to `diffeqsolve`

`__call__` is **two** plum-dispatched methods: the positional one on the class,
and a keyword-only overload registered below as
`@AbstractDiffEqSolver.__call__.dispatch`, which pops
`t0`/`t1`/`dt0`/`y0`/`args` out of `**kwargs` and re-enters the positional one.
In the positional form every diffrax argument appears twice — once in the
signature, once in the `dfx.diffeqsolve(...)` call — and the two lists have to
be read against each other:

- **A keyword added to the signature but not forwarded is silently ignored.**
  There is no type error and no test failure unless a doctest exercises it.
- The first six arguments are passed **positionally** (`terms`, `self.solver`,
  `t0`, `t1`, `dt0`, `y0`), matching diffrax's own order. A PR that inserts an
  argument, or that adapts to an upstream reordering, has to get this right at
  the call site as well as in the signature.
- diffraxtra's own additions (currently just `vectorize_interpolation`) must be
  keyword-only and must not be forwarded. Check that a new extra option is
  applied _after_ the solve, like `vectorize_interpolation` is, rather than
  smuggled into the diffrax call.
- **Both overloads are separately `eqx.filter_jit`-wrapped.** A new overload
  without the decorator runs uncompiled, silently. A new _named_ argument only
  has to be added to the positional signature — the keyword overload forwards
  the rest of `**kwargs` — but anything it `pop`s explicitly (currently `t0`,
  `t1`, `dt0`, `y0`, `args`) does need both.
- A new Python-level `bool` or `int` argument becomes a static one under
  `filter_jit` — i.e. a recompilation axis. Say so in the docstring if it is one
  a caller would plausibly vary.

## The MISSING sentinels

`max_steps` and `event` both default to `dataclasses.MISSING` in `__call__`, so
that "not passed" (use the field) stays distinguishable from an explicit `None`
(unbounded / no event):

```py
max_steps = self.max_steps if max_steps is MISSING else max_steps
event = self.event if event is MISSING else event
```

**Collapsing either to `None` is a behaviour change, not a cleanup.** For
`max_steps` it would make it impossible to request an unbounded solve from a
solver whose field is bounded — which is the reason the override exists, since
`SaveAt(steps=True)` rejects `max_steps=None`. For `event` it would make it
impossible to disable a solver's termination condition for one call. If a PR
touches these lines, the `is MISSING` identity checks and the `_MISSING_TYPE`
annotations both have to survive. A **new** overridable field should follow the
same pattern, not invent a second convention. `max_steps` is also `static=True`;
leave it that way.

## Shape bookkeeping

[interp.py](../../../src/diffraxtra/_src/interp.py) maintains one invariant by
hand, in two places:

- `__init__` flattens every batch dimension into a single leading axis
  (`x.reshape(-1, *x.shape[self.batch_ndim :])`) and stores `batch_shape` to
  restore it. `batch_shape` is inferred from
  `jnp.shape(scalar_interpolation.t0_if_trivial)` — from the solve's batching,
  not from `y0`.
- `evaluate` reverses it with three reshapes that assume that layout: axis 0 is
  the flat batch, the time axes are re-expanded from `t0shape`, and
  `x.shape[2:]` is the `y` shape.

So: **a change to the flattening must be mirrored in `evaluate`, and vice
versa.** The suspicious edits are anything that stops flattening in `__init__`,
anything that lets `t0` reach the inner `vmap` with more than one dimension, and
any new index literal (`x.shape[0]`, `x.shape[2:]`). None of these fail loudly —
ask for a doctest asserting the full `(*batch_shape, *t.shape, *y0_shape)`
contract on a batched, non-scalar-`y0` example, since that is the only case
where all three parts are non-empty.

Also in this file:

- **New property forwards must be batched, or say that they aren't.** `t0`,
  `t1`, `direction`, `t0_if_trivial` all `reshape`/`cast` back to `batch_shape`;
  `ts`, `ts_size`, `infos` and `y0_if_trivial` return the **flat** array
  unchanged. That inconsistency is existing behaviour — a new property should
  pick a side deliberately and document which.
- `y0_shape` is currently stored and never read. A PR that starts consuming it
  is relying on a value nothing validates; ask for validation at construction
  first.
- `evaluate(t0, t1)` recurses into two full evaluations and subtracts. Watch for
  changes that assume `y` supports arithmetic (it may be a PyTree) or that
  double an already-expensive path.

## `from_` overloads

Both classes expose a `plum`-dispatched `from_`, registered as module-bottom
functions that all shadow one name.

- **Register on the same class the file already uses.** Solver overloads go on
  the ABC (`@AbstractDiffEqSolver.from_.dispatch`), so subclasses inherit them;
  interpolation overloads go on the concrete
  `@VectorizedDenseInterpolation.from_.dispatch`. Mixing these up changes who
  inherits what.
- `# type: ignore[no-redef]` on every overload after the first, and ruff's
  `F811` ignore in `pyproject.toml`, are load-bearing. A PR removing them is
  removing the dispatch pattern.
- The identity overload checks `type(obj) is not cls` and raises `TypeError`;
  that exactness is deliberate — don't relax it to `isinstance`.
- The first parameter is `type[...]`, and dispatch resolves on it. A new
  overload that annotates it loosely, or that dispatches on a broad type like
  `Any` or `ArrayLike`, will collide with an existing method.
- **Every overload needs a doctest** — see below.

## Fields and the abstract contract

Every class is an `Abstract*`/`@final` pair of `equinox.Module`s. A new field
has to be added in two places: as an `eqx.AbstractVar[...]` on the ABC and as a
real field on the concrete class. Adding it to only one is either an unenforced
contract or an undeclared extension.

The classes are **not** `strict=True`, so equinox checks the `AbstractVar`
contract only at **instantiation** — a subclass missing a field imports fine,
type-checks fine, and raises
`TypeError: Can't instantiate abstract class ... with abstract attributes {...}`
when something first constructs one. That makes **adding an `AbstractVar` a
breaking change for every downstream subclass, deferred to runtime**: it
deserves a changelog line, not a silent field addition. (`event` was added
exactly this way.)

Match the existing style — a `#:` comment above the field, not a docstring below
— and reserve `static=True` for values that are genuinely compile-time
constants, since a static field that varies is a recompilation per value.

## Tests are doctests

`tests/` contains one test (the version check). Everything else is examples
collected by sybil from [conftest.py](../../../conftest.py) over `README.md`,
`src`, `skills`, `tests`.

- **A new public function, field, or overload with no `>>>` example is untested
  and uncovered.** This is the single most common gap in a PR here.
- ` ```python ` blocks in
  [skills/diffraxtra/SKILL.md](../../../skills/diffraxtra/SKILL.md) are
  **executed** by the MyST parser; ` ```py ` blocks are collected but skipped. A
  runnable example accidentally fenced ` ```py ` is a test that never runs; a
  deliberately-failing example fenced ` ```python ` breaks CI. Claims in that
  file only become tests when written as `assert`.
- **A repr-printing example needs a version guard.** diffrax 0.6 and 0.7 print
  different `repr`s, and CI tests both. `conftest.py` exposes `DIFFRAX_LT_070`
  in the sybil namespace for `.. skip: next if(DIFFRAX_LT_070, reason="...")`
  directives in `src/` docstrings; `README.md` can't use those, so
  `pytest_collection_modifyitems` drops **every** README item when diffrax <
  0.7. A new `>>> obj` example printing a diffrax repr without the skip
  directive passes locally and fails the `check_oldest` job.
- `conftest.py` enables x64, which is why examples print `f64[…]`.
- `filterwarnings = ["error"]` — a change that introduces a warning fails the
  suite, including a warning from a newer diffrax.
- Doctests match with `ELLIPSIS | NORMALIZE_WHITESPACE`. A `repr` written out in
  full is more brittle than it looks; the `Solution( t0=f64[], ... )` eliding
  style is there on purpose.

## Versions and dependencies

CI runs a `check_oldest` job with `--resolution lowest-direct`, so a PR using a
newer diffrax API has to raise the floor in `pyproject.toml` — passing on the
latest release is not enough. `jax` and `numpy` are imported but undeclared
(they arrive through diffrax); a new import from a package outside diffrax's own
dependency tree needs a real entry. A new nox session needs a matching PEP-735
dependency group, since the noxfile names one per session via `uv_groups=[...]`.

## Further reading

- [AGENTS.md](../../../AGENTS.md) — architecture, commands, and the pitfalls
  this skill draws from.
- [skills/diffraxtra/SKILL.md](../../../skills/diffraxtra/SKILL.md) — the shape
  contract, the constructors, subclassing, and a troubleshooting table.
- [diffrax docs](https://docs.kidger.site/diffrax/) — the upstream this package
  tracks.
