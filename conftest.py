"""Doctest configuration."""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any, Final

import diffrax as dfx
import jax
from packaging.version import Version, parse
from sybil import Document, Region, Sybil
from sybil.parsers.myst import (
    DocTestDirectiveParser as MystDocTestDirectiveParser,
    PythonCodeBlockParser as MystPythonCodeBlockParser,
    SkipParser as MystSkipParser,
)
from sybil.parsers.rest import (
    DocTestParser as ReSTDocTestParser,
    SkipParser as ReSTSkipParser,
)
from sybil.sybil import SybilCollection

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    MystDocTestDirectiveParser(optionflags=optionflags),
    MystPythonCodeBlockParser(doctest_optionflags=optionflags),
    MystSkipParser(),
]


DIFFRAX_LT_070: Final = parse(dfx.__version__) < Version("0.7")


# TODO: instead use a fixture
# (https://sybil.readthedocs.io/en/latest/integration.html#pytest)
def setup_namespace(namespace: dict[str, Any]) -> None:
    """Add pytest fixtures to the Sybil namespace."""
    namespace["DIFFRAX_LT_070"] = DIFFRAX_LT_070


# TODO: figure out native parser for `pycon` that doesn't require a new line at
# the end.
readme = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags)],
    patterns=["README.md"],
    setup=setup_namespace,
)
docs = Sybil(
    parsers=parsers,
    patterns=["*.md"],
    setup=setup_namespace,
)
python = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags), ReSTSkipParser(), *parsers],
    patterns=["*.py"],
    setup=setup_namespace,
)

pytest_collect_file = SybilCollection([docs, readme, python]).pytest()


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:  # noqa: ARG001
    """Skip README.md tests when diffrax < 0.7 due to repr differences."""
    if DIFFRAX_LT_070:
        items[:] = [
            item
            for item in items
            if not (hasattr(item, "fspath") and item.fspath.basename == "README.md")
        ]
