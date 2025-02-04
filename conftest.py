"""Doctest configuration."""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Document, Region, Sybil
from sybil.parsers.myst import (
    DocTestDirectiveParser as MystDocTestDirectiveParser,
    PythonCodeBlockParser as MystPythonCodeBlockParser,
    SkipParser as MystSkipParser,
)
from sybil.parsers.rest import DocTestParser as ReSTDocTestParser
from sybil.sybil import SybilCollection

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    MystDocTestDirectiveParser(optionflags=optionflags),
    MystPythonCodeBlockParser(doctest_optionflags=optionflags),
    MystSkipParser(),
]

# TODO: figure out native parser for `pycon` that doesn't require a new line at
# the end.
readme = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags)],
    patterns=["README.md"],
)
docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags), *parsers], patterns=["*.py"]
)

pytest_collect_file = SybilCollection([docs, readme, python]).pytest()
