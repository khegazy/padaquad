from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure tests/ is on sys.path so _helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import SEED


@pytest.fixture(autouse=True)
def seed():
    """Start every test from the same random seed.

    PadaQuad's starting mesh is random. Without this, a test that does not seed
    itself starts from whatever state the tests before it left, which changes
    with the tests selected and their order, so a failure could not be
    reproduced. Tests that request ``seed`` explicitly get the same seeding.
    """
    torch.manual_seed(SEED)
