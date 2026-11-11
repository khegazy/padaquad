"""Unit tests for the variable-mode adaptive_heun weights, the generic
VariableMethodClass, the node-fraction check, device methods, and variable dtype
methods."""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS
from padaquad.methods import UniformMethodClass, VariableMethodClass, _Tableau

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

    def test_variable_method_to_device(self):
        """VariableMethodClass.to_device('cpu') moves the node fractions to CPU."""
        method = VARIABLE_METHODS["interpolatory3_variable"].clone()
        method.to_device("cpu")
        assert method.c.device.type == "cpu"


# ---------------------------------------------------------------------------
# VariableMethodClass dtype conversion
# ---------------------------------------------------------------------------


class TestVariableMethodDtype:
    """Tests for variable method dtype conversion."""

    def test_to_dtype_float32(self):
        """VariableMethodClass.to_dtype converts the node fractions; the weights
        follow the dtype of the nodes they are computed from."""
        method = VARIABLE_METHODS["interpolatory3_variable"].clone()
        method.to_dtype(torch.float32)
        assert method.c.dtype == torch.float32
        b, b_error = method.weights(method.c.view(1, 3, 1))
        assert b.dtype == b_error.dtype == torch.float32


# ---------------------------------------------------------------------------
# The generic VariableMethodClass and the node-fraction check
# ---------------------------------------------------------------------------


_C3 = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)


def _equal_weights(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Module-level weight function: 1/3 per node, error weights 0. [N, 3]."""
    w = torch.full(c.shape[:2], 1.0 / 3.0, dtype=c.dtype, device=c.device)
    return w, torch.zeros_like(w)


class TestVariableMethodClass:
    """A position-dependent rule is one weight function plus one instance."""

    def test_weight_function_sees_normalized_nodes(self):
        """weights() hands the weight function each panel's nodes shifted to 0
        and scaled by the panel width: an off-centre panel [2.0, 2.6, 5.0]
        arrives as [0, 0.2, 1]."""
        seen = []

        def spy(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            seen.append(c.clone())
            return _equal_weights(c)

        rule = VariableMethodClass(
            order=1, error_exponent=2, c=_C3, weight_function=spy
        )
        nodes = torch.tensor([[[2.0], [2.6], [5.0]]], dtype=torch.float64)
        b, b_error = rule.weights(nodes)
        assert torch.allclose(
            seen[0], torch.tensor([[[0.0], [0.2], [1.0]]], dtype=torch.float64)
        )
        assert b.shape == b_error.shape == (1, 3, 1)
        assert rule.n_nodes == 3
        assert torch.equal(rule.node_fractions(), _C3)

    def test_wrong_weight_shape_raises(self):
        """A weight function returning [1, C] instead of [N, C] is rejected
        (it would broadcast silently in the weighted sum)."""

        def broadcast_weights(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            w = torch.full((1, 3), 1.0 / 3.0, dtype=c.dtype)
            return w, w

        rule = VariableMethodClass(
            order=1, error_exponent=2, c=_C3, weight_function=broadcast_weights
        )
        nodes = _C3.view(1, 3, 1).expand(4, 3, 1)
        with pytest.raises(ValueError, match="weight_function must return"):
            rule.weights(nodes)

    @pytest.mark.parametrize(
        "c",
        [
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),  # not 1-D
            torch.tensor([0.0], dtype=torch.float64),  # one node
            torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64),  # does not start at 0
            torch.tensor([0.0, 0.5, 0.9], dtype=torch.float64),  # does not end at 1
            torch.tensor([0.0, 0.7, 0.5, 1.0], dtype=torch.float64),  # decreases
        ],
        ids=["not-1d", "one-node", "start", "end", "decreasing"],
    )
    def test_malformed_node_fractions_raise(self, c):
        """Both generic rule classes reject malformed node fractions."""
        with pytest.raises(ValueError, match="node fractions"):
            VariableMethodClass(
                order=1, error_exponent=2, c=c, weight_function=_equal_weights
            )
        with pytest.raises(ValueError, match="node fractions"):
            UniformMethodClass(
                order=1,
                error_exponent=2,
                tableau=_Tableau(
                    c=c, b=torch.ones_like(c), b_error=torch.zeros_like(c)
                ),
            )

    def test_repeated_node_fractions_accepted(self):
        """Repeated fractions are valid (dopri5 has two nodes at 1)."""
        c = torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.float64)
        rule = UniformMethodClass(
            order=1,
            error_exponent=2,
            tableau=_Tableau(c=c, b=torch.ones_like(c), b_error=torch.zeros_like(c)),
        )
        assert rule.n_nodes == 4
