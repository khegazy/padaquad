"""Unit tests for the variable-mode adaptive_heun weights, device methods, and
variable dtype methods."""

from __future__ import annotations

import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS
from padaquad.methods import _Interpolatory3Variable, _Tableau

# ---------------------------------------------------------------------------
# Variable-mode adaptive_heun: the uniform heun object, constant weights
# ---------------------------------------------------------------------------


class TestVariableHeunWeights:
    """Variable-mode adaptive_heun is the uniform heun rule: constant weights."""

    def test_is_the_uniform_heun(self):
        """One object serves both registries."""
        assert VARIABLE_METHODS["adaptive_heun"] is UNIFORM_METHODS["adaptive_heun"]

    def test_returns_constant_b(self):
        """b matches the adaptive_heun tableau b values on every panel."""
        method = VARIABLE_METHODS["adaptive_heun"]
        nodes = torch.rand(5, 2, 1, dtype=torch.float64)
        b, _b_error = method.weights(nodes)
        expected_b = UNIFORM_METHODS["adaptive_heun"].tableau.b
        assert torch.equal(b[..., 0], expected_b.expand(5, -1))

    def test_shape(self):
        """Output shapes are [N, C, 1] for N=5, C=2."""
        method = VARIABLE_METHODS["adaptive_heun"]
        b, b_error = method.weights(torch.rand(5, 2, 1, dtype=torch.float64))
        assert b.shape == (5, 2, 1)
        assert b_error.shape == (5, 2, 1)

    def test_ignores_node_values(self):
        """Different node positions produce identical b outputs."""
        method = VARIABLE_METHODS["adaptive_heun"]
        b1, _ = method.weights(torch.zeros(3, 2, 1, dtype=torch.float64))
        b2, _ = method.weights(torch.ones(3, 2, 1, dtype=torch.float64))
        assert torch.equal(b1, b2)


# ---------------------------------------------------------------------------
# _Tableau.to_device / UniformMethodClass.to_device
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
        """UniformMethodClass.to_device('cpu') moves tableau to CPU."""
        # Clone to avoid mutating the singleton (cpu->cpu is a no-op
        # in terms of values, but keeping the test isolated is good
        # practice now that singletons are no longer self-protecting).
        method = UNIFORM_METHODS["bosh3"].clone()
        method.to_device("cpu")
        assert method.tableau.c.device.type == "cpu"
        assert method.tableau.b.device.type == "cpu"

    def test_variable_third_order_to_device(self):
        """_Interpolatory3Variable().to_device('cpu') moves c and b_delta to CPU."""
        method = _Interpolatory3Variable()
        method.to_device("cpu")
        assert method.c.device.type == "cpu"
        assert method.b_delta.device.type == "cpu"


# ---------------------------------------------------------------------------
# _Interpolatory3Variable dtype conversion
# ---------------------------------------------------------------------------


class TestVariableSubclassDtype:
    """Tests for variable method dtype conversion."""

    def test_third_order_to_dtype_float32(self):
        """_Interpolatory3Variable c and b_delta convert to float32."""
        method = _Interpolatory3Variable()
        method.to_dtype(torch.float32)
        assert method.c.dtype == torch.float32
        assert method.b_delta.dtype == torch.float32
