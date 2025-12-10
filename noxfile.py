"""Nox setup."""

import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.notify("precommit")
    s.notify("pylint")
    s.notify("mypy")


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run the pre-commit hooks."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "src", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def mypy(s: nox.Session, /) -> None:
    """Run mypy."""
    s.run("mypy", "src", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the tests."""
    s.notify("pytest", posargs=s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run the tests."""
    s.run("pytest", *s.posargs)


# =============================================================================
# Packaging


@session(uv_groups=["build"])
def rm_build(_: nox.Session, /) -> None:
    """Remove the build directory."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    rm_build(s)
    s.run("python", "-m", "build")
