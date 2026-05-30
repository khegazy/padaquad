"""Tests for Butcher tableau coefficient validity."""

from __future__ import annotations

import pytest
import torch
from _helpers import UNIFORM_METHOD_NAMES

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_uniform_tableau_b_sums_to_one(method_name):
    """The b weights of each uniform method must sum to 1 (consistency condition)."""
    method = UNIFORM_METHODS[method_name]
    b_sum = torch.sum(method.tableau.b)
    assert torch.abs(b_sum - 1.0) < 1e-7, (
        f"Uniform method '{method_name}': b weights sum to {b_sum.item()}, expected 1.0"
    )


@pytest.mark.parametrize(
    "method_name",
    [name for name, cls in VARIABLE_METHODS.items() if cls().order >= 3],
    ids=[name for name, cls in VARIABLE_METHODS.items() if cls().order >= 3],
)
def test_variable_tableau_b_sums_to_one(method_name):
    """Variable method b weights (computed from c) must sum to 1 for all valid c positions."""
    method = VARIABLE_METHODS[method_name]()
    n_samples = 100

    # Create c tensors with the interior node ranging from 0.01 to 0.99
    c_tensor = torch.concatenate(
        [
            torch.zeros((n_samples, 1, 1)),
            torch.linspace(0.01, 0.99, n_samples)[:, None, None],
            torch.ones((n_samples, 1, 1)),
        ],
        dim=1,
    )
    b_tensor, _ = method.tableau_b(c_tensor)
    b_sums = torch.sum(b_tensor, dim=-1)

    assert torch.all(torch.abs(b_sums - 1.0) < 1e-5), (
        f"Variable method '{method_name}': some b weight sums deviate from 1.0. "
        f"Max deviation: {torch.max(torch.abs(b_sums - 1.0)).item():.2e}"
    )
