"""pytest configuration for the docstring examples (doctests) in padaquad/.

The test suite's own fixtures are in tests/conftest.py. Here, every docstring
example starts from the same random seed (padaquad's starting mesh is random)
and runs with padaquad's log quieted, so its printed output is the same on
every machine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _doctest_environment(request: pytest.FixtureRequest) -> Iterator[None]:
    """Seed torch and silence padaquad's log for doctests; no-op for tests."""
    if type(request.node).__name__ != "DoctestItem":
        yield
        return
    logger = logging.getLogger("padaquad")
    level = logger.level
    logger.setLevel(logging.ERROR)
    torch.manual_seed(0)
    try:
        yield
    finally:
        logger.setLevel(level)
