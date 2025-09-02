"""Nox configuration."""

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2024.3.2"
nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def check(session: nox.Session, /) -> None:
    """Run all checks."""
    lint(session)
    test(session)


# =============================================================================


@nox.session
def lint(session: nox.Session, /) -> None:
    """Run the linter."""
    precommit(session)
    pylint(session)


@nox.session
def precommit(session: nox.Session, /) -> None:
    """Run the pre-commit hooks."""
    session.run(
        "uv",
        "run",
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def pylint(session: nox.Session, /) -> None:
    """Run PyLint."""
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.run("uv", "sync", "--group", "lint")
    session.run("uv", "run", "pylint", "src", *session.posargs)


# =============================================================================
# Testing


@nox.session
def test(session: nox.Session, /) -> None:
    """Run the tests."""
    session.run("uv", "sync", "--group", "test")
    session.run("uv", "run", "pytest", *session.posargs)


# =============================================================================
# Packaging


@nox.session
def rm_build(_: nox.Session, /) -> None:
    """Remove the build directory."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)


@nox.session
def build(session: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    rm_build(session)
    session.run("uv", "build", *session.posargs)
