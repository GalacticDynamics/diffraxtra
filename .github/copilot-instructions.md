# Project Overview

This repository provides extra functionality for `diffrax`, a library for
differentiable ODEs.

- **Language**: Python
- **Main API**: Extensions and utilities for `diffrax`
  - Additional differential equation types and solvers
  - Interpolation utilities
  - Helper functions for ODE solving workflows
- **Design goals**: Extend diffrax capabilities, maintain JAX compatibility,
  provide useful abstractions for differential equations and their solutions.
- **JAX integration**: All objects are PyTree compatible and work seamlessly
  with JAX transformations like `jit`, `vmap`, and `grad`.

## Folder Structure

- `/src/diffraxtra`: Contains the source code.
  - `_src/`: the private code
    - `diffeq_abc.py`: Abstract base classes for differential equations
    - `diffeq.py`: Concrete differential equation implementations
    - `interp.py`: Interpolation utilities
  - `__init__.py`: Public API.
- `README.md`: Project documentation and usage examples. The Python code blocks
  are also tested as part of the test suite.
- Tests:
  - `noxfile.py`: Nox configuration for sessions like linting, testing, and
    building.
  - `conftest.py`: Pytest configuration and fixtures.
  - `/tests`: Contains the tests.

## Coding Style

- Always use type hints (standard typing, `diffrax` types, JAX types, etc.).
- Follow diffrax conventions and patterns where possible.
- Keep dependencies minimal; the core dependencies are listed in
  `pyproject.toml`.
- Docstrings should be concise and include testable usage examples.
- Maintain compatibility with diffrax's API design patterns.

## Tooling

- This repo uses `uv` for managing virtual environments and running commands.
- This repo uses `nox` for testing and automation.
- Before committing, to do a full linting and testing, run:

  ```bash
  uv run nox -s check
  ```

## Testing

- Use `pytest` for all test suites.
- Add unit tests for every new function or class.
- Test compatibility with diffrax workflows and transformations.
- For JAX-related behavior:
  - Confirm PyTree registration works correctly.
  - Verify compatibility with transformations like `jit`, `vmap`, and `grad`.
  - Test numerical accuracy of ODE solutions where applicable.
  - Tests should run on CPU by default; no accelerators required.

## Final Notes

Prefer clarity over cleverness. Maintain compatibility with diffrax patterns.
When in doubt, follow diffrax's API conventions and JAX best practices.
