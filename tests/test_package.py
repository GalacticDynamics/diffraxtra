"""Test the package metadata."""

import importlib.metadata

import diffraxtra


def test_version():
    """Test that the package version matches the metadata."""
    assert importlib.metadata.version("diffraxtra") == diffraxtra.__version__
