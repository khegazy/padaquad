"""
Base classes and data structures for the padaquad integration library.

Defines the core abstractions that all solvers build on:

- ``steps`` enum: the three integration step strategies (fixed, adaptive uniform,
  adaptive variable).
- ``IntegrationResult`` / ``MethodOutput``: dataclasses carrying results through the
  integration pipeline.
- ``SolverBase``: abstract base class providing dtype/device management, default
  parameter handling, and the ``integrate()`` / ``_calculate_integral()`` interface
  that concrete solvers must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .distributed import DistributedEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable

    from .results import IntegrationResult, MethodOutput


class steps(Enum):
    """
    Enumeration of integration step strategies.

    - FIXED: Fixed step sizes (not adaptive).
    - ADAPTIVE_UNIFORM: Adaptive step sizes with quadrature points placed at
      fixed fractional positions within each step (tableau c values are constant).
    - ADAPTIVE_VARIABLE: Adaptive step sizes with quadrature points at arbitrary
      positions. Tableau b weights are recomputed dynamically based on actual
      point positions.
    """

    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2


def get_sampling_type(sampling_type: str) -> steps:
    """
    Convert a string sampling type name into the corresponding ``steps`` enum.

    Accepts both short names ('uniform', 'variable') and full names
    ('adaptive_uniform', 'adaptive_variable').

    Args:
        sampling_type: One of 'fixed', 'uniform', 'adaptive_uniform',
            'variable', or 'adaptive_variable'.

    Returns:
        The corresponding ``steps`` enum value.

    Raises:
        KeyError: If sampling_type is not a recognized name.
    """
    types = {
        "fixed": steps.FIXED,
        "adaptive_uniform": steps.ADAPTIVE_UNIFORM,
        "uniform": steps.ADAPTIVE_UNIFORM,
        "adaptive_variable": steps.ADAPTIVE_VARIABLE,
        "variable": steps.ADAPTIVE_VARIABLE,
    }
    return types[sampling_type]


# Result dataclasses live in ``results.py``; re-exported here for
# backward-compatible imports.


class SolverBase(ABC, DistributedEnvironment):
    """
    Abstract base class for all numerical integration solvers.

    Provides common infrastructure inherited by both the serial solver
    (which wraps torchdiffeq) and the parallel adaptive solvers. Handles:

    - Dtype management: converts all tensors to a consistent dtype, and sets
      dtype-appropriate assertion tolerances (atol_assert, rtol_assert) used
      for internal sanity checks (e.g., verifying time ordering).
    - Device management: inherited from DistributedEnvironment.
    - Default parameter storage: f, y0, mesh_init, mesh_final can be set at
      construction and reused across multiple integrate() calls.
    - Warm-start caching: stores mesh_previous and previous_f_id
      so that repeated integration of the same function can reuse the optimized
      time mesh from the prior run.

    Subclasses must implement:
        - ``_set_solver_dtype(dtype)``: update solver-specific state when dtype changes.
        - ``_calculate_integral(t, y, y0)``: compute integral + error for a batch of steps.
        - ``integrate(...)``: run the full integration procedure.
    """

    def __init__(
        self,
        method: str = "gk21",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        y0: torch.Tensor | None = None,
        f: Callable | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        output_speed_info: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the solver base with integration parameters.

        Args:
            method: Name of the RK integration method (e.g. 'dopri5', 'bosh3').
                Converted to lowercase.
            atol: Absolute error tolerance for adaptive step control. Steps with
                error below atol + rtol*|integral| are accepted.
            rtol: Relative error tolerance for adaptive step control.
            y0: Initial value of the integral (accumulator starting point).
                Shape: [D].
            f: The integrand function f(t) -> Tensor. Takes time points
                of shape [N, T] and returns evaluations of shape [N, D].
                Can also be provided at integrate() call time.
            mesh_init: Lower bound of integration. Shape: [T].
            mesh_final: Upper bound of integration. Shape: [T].
            output_speed_info: If True, logs timing information for each
                sub-operation during integration to a dedicated file
                ``padaquad_speed.log``.
            dtype: Floating point precision for all computations. Supported:
                torch.float64 (default, academic-grade), torch.float32 (fast
                on GPU). torch.float16 is refused: its ~1e-3 precision floor
                sits above typical adaptive tolerances and cannot support
                meaningful error control.
            device: Device string (e.g. 'cuda', 'cpu'). Passed to
                DistributedEnvironment for device assignment.
            *args: Additional arguments forwarded to DistributedEnvironment.
            **kwargs: Additional keyword arguments forwarded to
                DistributedEnvironment.
        """
        super().__init__(*args, **kwargs, device_type=device)

        # Speed timing logger — writes to a dedicated file, no other output
        if output_speed_info:
            self.speed_logger = logging.getLogger("padaquad.speed")
            self.speed_logger.propagate = False
            self.speed_logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler("padaquad_speed.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.speed_logger.addHandler(handler)
        else:
            self.speed_logger = None

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        # Construction-time defaults; each integrate() call falls back to these
        # when its corresponding argument is None (the active value is the local
        # resolved in _check_variables).
        self.init_f = f
        self.init_y0 = (
            y0.to(self.device)
            if y0 is not None
            else torch.tensor([0], dtype=torch.float64, device=self.device)
        )
        self.init_mesh_init = (
            mesh_init.to(self.device)
            if mesh_init is not None
            else torch.tensor([0], dtype=torch.float64, device=self.device)
        )
        self.init_mesh_final = (
            mesh_final.to(self.device)
            if mesh_final is not None
            else torch.tensor([1], dtype=torch.float64, device=self.device)
        )
        # Cached time barriers from last integration for warm-starting.
        # Used only when integrate(..., reuse_mesh=True) is passed; otherwise
        # ignored. The cached barriers are the *optimal* mesh from the
        # previous successful run (post-prune-and-refine).
        self.mesh_previous = None
        # id() of the last integrated function. Stored as a sanity-check
        # signal so that reuse_mesh=True can warn if the cached mesh was
        # tuned for a different integrand. Replaces the prior __name__
        # comparison which collided across all lambdas.
        self.previous_f_id = None

        self._set_dtype(dtype)

    # -------------------------------------------------------------------------------- #
    #                                 ABSTRACT METHODS                                 #
    # -------------------------------------------------------------------------------- #

    @abstractmethod
    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """
        Update solver-specific state when dtype changes.

        Called by ``_set_dtype()`` after converting base tensors. Subclasses
        must implement this to convert their own tensors (e.g., RK tableaux).

        Args:
            dtype: The new dtype to convert to.
        """

    @abstractmethod
    def _calculate_integral(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        y0: torch.Tensor | None,
    ) -> MethodOutput:
        """
        Compute the integral and error estimate for a batch of integration steps.

        This is the core numerical method that subclasses must implement.
        Given time points and integrand evaluations at those points, applies
        the specific RK quadrature rule to produce integral contributions
        and error estimates for each step.

        Args:
            t: Time points for each step, with C quadrature points per step.
                Shape: [N, C, T] where N=steps, C=quadrature points, T=time dims.
            y: Integrand evaluations at the time points in t.
                Shape: [N, C, D] where D=output dimensionality of the integrand.
            y0: Starting value for the integral accumulator. Shape: [D].

        Returns:
            MethodOutput containing the integral, error, per-step contributions,
            per-step errors, and step sizes.
        """

    @abstractmethod
    def integrate(
        self,
        f: Callable,
        y0: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
        f_args: tuple = (),
    ) -> IntegrationResult:
        """
        Perform numerical path integration of f from mesh_init to mesh_final.

        This is the main entry point that users call. Subclasses implement
        the actual integration logic (serial or parallel adaptive).

        The integrand f should be a function of time only: f(t) -> Tensor.
        It does NOT depend on accumulated state y (this is numerical quadrature,
        not ODE solving). This independence between steps is what enables
        parallel evaluation.

        Args:
            f: The integrand function. Takes time points of shape [N, T]
                and returns values of shape [N, D].
            y0: Initial value of the integral (accumulator start). Shape: [D].
            mesh_init: Lower integration bound. Shape: [T].
            mesh_final: Upper integration bound. Shape: [T].
            t: Optional initial time points. If provided, these serve as the
                starting mesh for adaptive refinement. Shape depends on solver:
                [N, T] for step barriers, [N, C, T] for full quadrature points.
            f_args: Extra arguments passed to f after the time tensor.

        Returns:
            IntegrationResult containing the computed integral, error estimates,
            time mesh, and optimization diagnostics.

        Note:
            Handling of the input time ``t`` differs between parallel and serial
            solvers. See each solver's documentation for details.
        """

    # -------------------------------------------------------------------------------- #
    #                         DATA TYPE AND VARIABLE HANDLING                          #
    # -------------------------------------------------------------------------------- #

    def _set_dtype(self, dtype: torch.dtype) -> None:
        """
        Set the floating-point precision for all solver tensors.

        Converts y0, mesh_init, mesh_final, and cached barriers to the new dtype.
        Also sets dtype-appropriate assertion tolerances used for internal
        sanity checks (e.g., verifying that time points are monotonically
        ordered despite floating-point rounding).

        The assertion tolerances by dtype are:
            - float64: atol_assert=1e-15, rtol_assert=1e-7
            - float32: atol_assert=1e-7,  rtol_assert=1e-5

        float16 is refused (Bug B4): float16's ~1e-3 precision floor
        sits *above* the typical adaptive integration tolerance, so
        the solver cannot meaningfully verify "step error < tol" for
        any user-relevant tolerance. Use float32 (still fast on GPU)
        or float64 (academic-grade precision) instead.

        Args:
            dtype: Target dtype. Must be float64 or float32.

        Raises:
            ValueError: If dtype is float16 or any unsupported type.
        """
        if hasattr(self, "dtype") and dtype == self.dtype:
            return

        # Bug B4 guard: refuse float16 + adaptive at construction time.
        if dtype == torch.float16:
            raise ValueError(
                "torch.float16 is too coarse for adaptive error control "
                "(precision floor ~1e-3 exceeds typical integration "
                "tolerances). Use torch.float32 or torch.float64 instead."
            )
        if dtype not in (torch.float64, torch.float32):
            raise ValueError(
                f"Given dtype must be torch.float64 or torch.float32; got {dtype}"
            )

        self.dtype = dtype
        self.init_y0 = self.init_y0.to(self.dtype)
        self.init_mesh_init = self.init_mesh_init.to(self.dtype)
        self.init_mesh_final = self.init_mesh_final.to(self.dtype)
        if self.mesh_previous is not None:
            self.mesh_previous = self.mesh_previous.to(self.dtype)

        # Set assertion tolerances appropriate for this precision level.
        # These are used in internal sanity checks (e.g., time ordering),
        # NOT for integration error control (that uses self.atol/self.rtol).
        if self.dtype == torch.float64:
            self.atol_assert = 1e-15
            self.rtol_assert = 1e-7
        else:  # float32
            self.atol_assert = 1e-7
            self.rtol_assert = 1e-5

        self._set_solver_dtype(self.dtype)

    def set_dtype_by_input(
        self,
        mesh: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
    ) -> None:
        """
        Infer and set the solver dtype from the dtype of input tensors.

        Checks mesh first, then mesh_init, then mesh_final. The first
        non-None tensor's dtype is used. This allows the solver to
        automatically match the precision of user-provided inputs.

        Args:
            mesh: Optional mesh-of-barriers tensor.
            mesh_init: Optional lower integration bound tensor.
            mesh_final: Optional upper integration bound tensor.
        """
        if mesh is not None:
            self._set_dtype(mesh.dtype)
        elif mesh_init is not None:
            self._set_dtype(mesh_init.dtype)
        elif mesh_final is not None:
            self._set_dtype(mesh_final.dtype)

    def _check_variables(
        self,
        f: Callable | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
    ) -> tuple[
        Callable | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """
        Fill in missing arguments with stored defaults and move to correct device.

        Any argument that is None is replaced with the value stored at
        construction time. Tensor arguments are cast to the solver's dtype
        and moved to the solver's device.

        Args:
            f: Integrand function, or None to use self.init_f.
            mesh_init: Lower integration bound, or None to use self.init_mesh_init.
            mesh_final: Upper integration bound, or None to use self.init_mesh_final.
            y0: Initial integral value, or None to use self.init_y0.

        Returns:
            Tuple of (f, mesh_init, mesh_final, y0) with defaults filled
            in and tensors on the correct device/dtype.
        """
        f = self.init_f if f is None else f
        mesh_init = self.init_mesh_init if mesh_init is None else mesh_init
        mesh_final = self.init_mesh_final if mesh_final is None else mesh_final
        y0 = self.init_y0 if y0 is None else y0

        mesh_init = mesh_init.to(self.dtype).to(self.device)
        mesh_final = mesh_final.to(self.dtype).to(self.device)
        y0 = y0.to(self.dtype).to(self.device)

        return f, mesh_init, mesh_final, y0

    # -------------------------------------------------------------------------------- #
    #                         TRAINING AND EVALUATION TOGGLING                         #
    # -------------------------------------------------------------------------------- #

    def _integral_loss(
        self,
        result: IntegrationResult,
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Default loss function: returns the integral value itself.

        Used when no custom loss_fxn is provided. Can be overridden or
        replaced at integrate() call time to compute a custom loss for
        gradient-based optimization.

        Args:
            result: The current IntegrationResult from this batch.

        Returns:
            Loss tensor for backpropagation. Shape: same as result.integral.
        """
        return result.integral

    # -------------------------------------------------------------------------------- #
    #                                    DESTRUCTOR                                    #
    # -------------------------------------------------------------------------------- #

    def __del__(self) -> None:
        """Destructor that cleans up the distributed process group."""
        self.end_process()
