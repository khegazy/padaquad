"""Unit tests for _VARIABLE_SECOND_ORDER.tableau_b, device methods, and variable dtype methods."""

from __future__ import annotations

import torch

from padaquad import UNIFORM_METHODS
from padaquad.methods import (
    _VARIABLE_SECOND_ORDER,
    _Interpolatory3Variable,
    _Tableau,
)

# ---------------------------------------------------------------------------
# _VARIABLE_SECOND_ORDER.tableau_b
# ---------------------------------------------------------------------------


class TestVariableSecondOrderTableauB:
    """Tests for _VARIABLE_SECOND_ORDER.tableau_b: constant weights."""

    def test_returns_constant_b(self):
        """b matches the adaptive_heun tableau b values."""
        method = _VARIABLE_SECOND_ORDER()
        c = torch.rand(5, 2, 1)
        b, _b_error = method.tableau_b(c)
        expected_b = UNIFORM_METHODS["adaptive_heun"].tableau.b
        assert torch.allclose(b, expected_b)

    def test_shape(self):
        """Output shapes are [1, C] for C=2."""
        method = _VARIABLE_SECOND_ORDER()
        c = torch.rand(5, 2, 1)
        b, b_error = method.tableau_b(c)
        assert b.shape == (1, 2)
        assert b_error.shape == (1, 2)

    def test_ignores_c_values(self):
        """Different c values produce identical b outputs."""
        method = _VARIABLE_SECOND_ORDER()
        c1 = torch.zeros(3, 2, 1)
        c2 = torch.ones(3, 2, 1)
        b1, _ = method.tableau_b(c1)
        b2, _ = method.tableau_b(c2)
        assert torch.equal(b1, b2)


# ---------------------------------------------------------------------------
# _Tableau.to_device / MethodClass.to_device
# ---------------------------------------------------------------------------


class TestTableauToDevice:
    """Tests for device movement (CPU-only)."""

    def test_tableau_to_device_cpu(self):
        """_Tableau.to_device('cpu') keeps all tensors on CPU."""
        tableau = _Tableau(
            c=torch.tensor([0.0, 1.0]),
            b=torch.tensor([0.5, 0.5]),
            b_error=torch.tensor([1.0, -1.0]),
        )
        tableau.to_device("cpu")
        assert tableau.c.device.type == "cpu"
        assert tableau.b.device.type == "cpu"
        assert tableau.b_error.device.type == "cpu"

    def test_method_class_to_device_cpu(self):
        """MethodClass.to_device('cpu') moves tableau to CPU."""
        # Clone to avoid mutating the singleton (cpu->cpu is a no-op
        # in terms of values, but keeping the test isolated is good
        # practice now that singletons are no longer self-protecting).
        method = UNIFORM_METHODS["bosh3"].clone()
        method.to_device("cpu")
        assert method.tableau.c.device.type == "cpu"
        assert method.tableau.b.device.type == "cpu"

    def test_variable_third_order_to_device(self):
        """_Interpolatory3Variable().to_device('cpu') moves b_delta to CPU."""
        method = _Interpolatory3Variable()
        method.to_device("cpu")
        assert method.b_delta.device.type == "cpu"

    def test_variable_second_order_to_device(self):
        """_VARIABLE_SECOND_ORDER().to_device('cpu') moves tableau to CPU."""
        method = _VARIABLE_SECOND_ORDER()
        method.to_device("cpu")
        assert method.tableau.c.device.type == "cpu"
        assert method.tableau.b.device.type == "cpu"


# ---------------------------------------------------------------------------
# Variable subclass dtype conversion
# ---------------------------------------------------------------------------


class TestVariableSubclassDtype:
    """Tests for variable method dtype conversion."""

    def test_third_order_to_dtype_float32(self):
        """_Interpolatory3Variable b_delta converts to float32."""
        method = _Interpolatory3Variable()
        method.to_dtype(torch.float32)
        assert method.b_delta.dtype == torch.float32

    def test_second_order_to_dtype_float32(self):
        """_VARIABLE_SECOND_ORDER tableau converts to float32."""
        method = _VARIABLE_SECOND_ORDER()
        method.to_dtype(torch.float32)
        assert method.tableau.b.dtype == torch.float32
        assert method.tableau.c.dtype == torch.float32
