"""
Base class and sampling modes shared by the padaquad solvers.

- ``steps``: the enum of sampling modes, and ``get_sampling_type``, which turns a
  name such as ``"uniform"`` into a ``steps`` member.
- ``DEFAULT_SAMPLING_TYPE`` and ``DEFAULT_METHOD``: the sampling mode and the
  method used when the caller names none.
- ``SolverBase``: the abstract base class of the solvers. It keeps the dtype,
  the device, the construction-time defaults and the warm-start mesh, and
  declares the ``integrate()`` / ``_calculate_integral()`` interface that the
  concrete solvers implement.

The result dataclasses ``IntegrationResult`` and ``MethodOutput`` live in
``results.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from .distributed import DistributedEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable

    from .results import IntegrationResult, MethodOutput


class steps(Enum):
    """
    Sampling modes, which select the solver class and its list of methods.

    - ``FIXED``: a non-adaptive mode. No solver supports it:
      ``adaptive_quadrature`` and ``integrate`` raise ``ValueError`` for it.
    - ``ADAPTIVE_UNIFORM``: adaptive splitting, with the nodes of every
      subinterval at the method's fixed node fractions and fixed weights
      (``UniformAdaptiveQuadrature``, methods in ``UNIFORM_METHODS``).
    - ``ADAPTIVE_VARIABLE``: adaptive splitting, with the weights computed from
      the actual node positions (``VariableAdaptiveQuadrature``, methods in
      ``VARIABLE_METHODS``).
    """

    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2


# Sampling type used when the caller gives none (adaptive_quadrature, integrate).
DEFAULT_SAMPLING_TYPE = "uniform"

# Default quadrature method of the solvers and of ``padaquad.integrate()``.
DEFAULT_METHOD = "gk21"


def get_sampling_type(sampling_type: str) -> steps:
    """
    Return the ``steps`` member named by a string.

    Accepts the short names ('uniform', 'variable'), the full names
    ('adaptive_uniform', 'adaptive_variable') and 'fixed', in any letter case.

    Parameters
    ----------
    sampling_type : str
        One of 'fixed', 'uniform', 'adaptive_uniform', 'variable' or
        'adaptive_variable'. No solver supports 'fixed'.

    Returns
    -------
    steps
        The corresponding ``steps`` enum value.

    Raises
    ------
    ValueError
        If sampling_type is not a recognized name.
    """
    types = {
        "fixed": steps.FIXED,
        "adaptive_uniform": steps.ADAPTIVE_UNIFORM,
        "uniform": steps.ADAPTIVE_UNIFORM,
        "adaptive_variable": steps.ADAPTIVE_VARIABLE,
        "variable": steps.ADAPTIVE_VARIABLE,
    }
    name = sampling_type.lower() if isinstance(sampling_type, str) else sampling_type
    if name not in types:
        raise ValueError(
            f"Unknown sampling_type {sampling_type!r}; valid names are {sorted(types)}."
        )
    return types[name]


# Result dataclasses live in ``results.py``; this module imports them only for
# type annotations, so they cannot be imported from here.


class SolverBase(ABC, DistributedEnvironment):
    """
    Abstract base class of the solvers: dtype, device, defaults and warm starts.

    - Dtype: keeps the tolerances and every stored tensor in one dtype
      (float64 or float32), and sets looser tolerances for internal
      consistency checks, such as the check that the nodes are in order.
    - Device: inherited from ``DistributedEnvironment``.
    - Defaults: ``f``, ``y0``, ``mesh_init`` and ``mesh_final`` given at
      construction are used by every ``integrate`` call that does not pass
      them.
    - Warm starts: stores the optimal mesh of the last call that returned, in
      ``mesh_previous``; ``integrate(..., same_f=True)`` starts from it. See
      :ref:`ug-same-f`.

    Subclasses implement the abstract methods: ``integrate``, the integral and
    error estimate of a batch of subintervals, and the update of their own
    tensors when the dtype changes.
    """

    # Set by ``_set_dtype`` (first called at the end of ``__init__``). Declared
    # without a value: ``_set_dtype`` uses ``hasattr(self, "dtype")`` to detect
    # its first call.
    dtype: torch.dtype

    def __init__(
        self,
        *,
        method: str,
        atol: float,
        rtol: float,
        y0: torch.Tensor | None,
        f: Callable[..., Any] | None,
        mesh_init: torch.Tensor | None,
        mesh_final: torch.Tensor | None,
        output_speed_info: bool,
        dtype: torch.dtype,
        device: str | torch.device | None,
        **distributed_options: Any,
    ) -> None:
        """
        Initialize the solver base with integration parameters.

        Every argument is required and keyword-only: the defaults live once, in
        ``AdaptiveQuadrature.__init__``, the constructor users call.

        Parameters
        ----------
        method : str
            Name of the quadrature method, for example 'gk21' or 'dopri5', in
            any letter case.
        atol : float
            Absolute tolerance. See :ref:`ug-error-control`.
        rtol : float
            Relative tolerance. See :ref:`ug-error-control`.
        y0 : torch.Tensor or None
            Value added to the integral, shape ``[D]``. None uses 0.
        f : callable or None
            The integrand: points of shape ``[N, T]`` in, values of shape
            ``[N, D]`` out. None means ``integrate`` must be given one.
        mesh_init : torch.Tensor or None
            Lower bound, shape ``[T]``. None uses 0.
        mesh_final : torch.Tensor or None
            Upper bound, shape ``[T]``. None uses 1.
        output_speed_info : bool
            If True, writes the time each part of the integration takes to
            the file ``padaquad_speed.log``.
        dtype : torch.dtype
            torch.float64 or torch.float32. torch.float16 raises ``ValueError``:
            its rounding error (about 1e-3) exceeds typical tolerances.
        device : str or torch.device or None
            'cpu', 'cuda', an indexed 'cuda:N', or a torch.device; None picks
            CUDA when it is available.
        **distributed_options : Any
            Options for multi-process runs: backend, master_addr, master_port
            and is_slurm.
        """
        super().__init__(device=device, **distributed_options)

        # Speed timing logger — writes to a dedicated file, no other output
        self.speed_logger: logging.Logger | None
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
        self.atol = torch.tensor(atol, dtype=dtype, device=self.device)
        self.rtol = torch.tensor(rtol, dtype=dtype, device=self.device)
        # Construction-time defaults; each integrate() call falls back to these
        # when its corresponding argument is None (the active value is the local
        # resolved in _check_variables).
        self.init_f: Callable[..., Any] | None = f
        self.init_y0 = (
            y0.to(self.device)
            if y0 is not None
            else torch.tensor([0], dtype=dtype, device=self.device)
        )
        self.init_mesh_init = (
            mesh_init.to(self.device)
            if mesh_init is not None
            else torch.tensor([0], dtype=dtype, device=self.device)
        )
        self.init_mesh_final = (
            mesh_final.to(self.device)
            if mesh_final is not None
            else torch.tensor([1], dtype=dtype, device=self.device)
        )
        # Stored time barriers from the last integration for warm-starting.
        # Used only when integrate(..., same_f=True) is passed;
        # otherwise ignored. They are the *optimal* mesh of the last call that
        # returned (post-prune-and-refine).
        self.mesh_previous: torch.Tensor | None = None

        self._set_dtype(dtype)

    # -------------------------------------------------------------------------------- #
    #                                 ABSTRACT METHODS                                 #
    # -------------------------------------------------------------------------------- #

    @abstractmethod
    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """
        Update solver-specific state when dtype changes.

        Called by ``_set_dtype()`` after converting base tensors, including
        its first call at the end of ``__init__``. Subclasses must implement
        this to build or convert their own tensors (e.g. the method's tables).

        Parameters
        ----------
        dtype : torch.dtype
            The new dtype to convert to.
        """

    @abstractmethod
    def _calculate_integral(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        y0: torch.Tensor,
    ) -> MethodOutput:
        """
        Compute the integral and error estimate for a batch of subintervals.

        Given the nodes of each subinterval and the integrand values there,
        applies the method's weights to produce each subinterval's contribution
        to the integral and its error estimate.

        Parameters
        ----------
        t : torch.Tensor
            Nodes of each subinterval. Shape: [N, C, T] (N subintervals, C
            nodes each, T coordinates of the integration variable t).
        y : torch.Tensor
            Integrand values at the nodes in t. Shape: [N, C, D] (D outputs
            of the integrand).
        y0 : torch.Tensor
            Starting value for the integral accumulator. Shape: [D].

        Returns
        -------
        MethodOutput
            The integral, its error estimate, each subinterval's contribution
            and error estimate, and the subinterval widths.
        """

    @abstractmethod
    def integrate(
        self,
        f: Callable[..., Any] | None = None,
        y0: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        *,
        f_args: tuple[Any, ...] = (),
    ) -> IntegrationResult:
        """
        Integrate f from mesh_init to mesh_final.

        This is the entry point users call. Every override must accept these
        arguments (it may add more); see ``AdaptiveQuadrature.integrate`` for
        the full parameter list. ``f_args`` is keyword-only here so an override
        may place it after its own extra parameters.

        The integrand depends only on the integration variable t and on
        ``f_args``, as ``f(t, *f_args)``, never on the running value of the
        integral (this is quadrature, not ODE solving). Because of this, the
        subintervals are independent and can be evaluated in parallel.

        Parameters
        ----------
        f : callable or None
            The integrand: points of shape ``[N, T]`` in, values of shape
            ``[N, D]`` out. None uses the one given at construction.
        y0 : torch.Tensor or None
            Value added to the returned integral. Shape: ``[D]``.
        mesh : torch.Tensor or None
            Starting mesh. Shape: ``[M, T]``.
        mesh_init : torch.Tensor or None
            Lower bound. Shape: ``[T]``.
        mesh_final : torch.Tensor or None
            Upper bound. Shape: ``[T]``.
        f_args : tuple
            Extra arguments passed to f after t.

        Returns
        -------
        IntegrationResult
            The integral, its error estimate, the mesh and diagnostics.
        """

    # -------------------------------------------------------------------------------- #
    #                         DATA TYPE AND VARIABLE HANDLING                          #
    # -------------------------------------------------------------------------------- #

    def _set_dtype(self, dtype: torch.dtype) -> None:
        """
        Set the floating-point precision for all solver tensors.

        Converts atol, rtol, y0, mesh_init, mesh_final and the stored
        warm-start mesh (``mesh_previous``) to the new dtype, then calls
        ``_set_solver_dtype``. Also sets dtype-appropriate assertion tolerances
        used for internal sanity checks (e.g. verifying that the nodes are in
        increasing order despite floating-point rounding).

        The assertion tolerances by dtype are:
            - float64: atol_assert=1e-15, rtol_assert=1e-7
            - float32: atol_assert=1e-7,  rtol_assert=1e-5

        float16 is refused (Bug B4): its rounding error, about 1e-3, sits
        *above* typical tolerances, so the solver cannot verify that a
        subinterval's error is below the tolerance. Use float32 (fast on GPU)
        or float64 instead. Does nothing when ``dtype`` is already the
        solver's dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Target dtype. Must be float64 or float32.

        Raises
        ------
        ValueError
            If dtype is float16 or any unsupported type.
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

        # Update integration control error tolerances
        self.atol = self.atol.to(self.dtype)
        self.rtol = self.rtol.to(self.dtype)

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
        Switch the solver to the dtype of the given mesh or bounds.

        ``integrate`` calls this, so the solver follows the dtype of the mesh
        or bounds passed to it. The first of ``mesh``, ``mesh_init`` and
        ``mesh_final`` that is not None sets the dtype; if all are None, the
        dtype is unchanged. The switch persists: later calls keep the new dtype
        until another input changes it.

        Parameters
        ----------
        mesh : torch.Tensor or None
            Starting mesh. Shape: ``[M, T]``.
        mesh_init : torch.Tensor or None
            Lower bound. Shape: ``[T]``.
        mesh_final : torch.Tensor or None
            Upper bound. Shape: ``[T]``.

        Raises
        ------
        ValueError
            If that tensor's dtype is not torch.float64 or torch.float32.
        """
        if mesh is not None:
            self._set_dtype(mesh.dtype)
        elif mesh_init is not None:
            self._set_dtype(mesh_init.dtype)
        elif mesh_final is not None:
            self._set_dtype(mesh_final.dtype)

    def _check_variables(
        self,
        f: Callable[..., Any] | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
    ) -> tuple[Callable[..., Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fill in missing arguments with stored defaults and move to correct device.

        Any argument that is None is replaced with the value stored at
        construction time. Tensor arguments are cast to the solver's dtype
        and moved to the solver's device.

        Parameters
        ----------
        f : Callable[..., Any] | None
            Integrand function, or None to use self.init_f.
        mesh_init : torch.Tensor | None
            Lower integration bound, or None to use self.init_mesh_init.
        mesh_final : torch.Tensor | None
            Upper integration bound, or None to use self.init_mesh_final.
        y0 : torch.Tensor | None
            Initial integral value, or None to use self.init_y0.

        Returns
        -------
        tuple[Callable[..., Any], torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (f, mesh_init, mesh_final, y0) with defaults filled
            in and tensors on the correct device/dtype.

        Raises
        ------
        ValueError
            If no integrand was given here or at construction.
        """
        f = self.init_f if f is None else f
        if f is None:
            raise ValueError(
                "No integrand given: pass f to integrate() or to the solver "
                "constructor."
            )
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
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Default loss function: returns the integral itself.

        Used when ``integrate`` gets no ``loss_fxn``. With
        ``take_gradient=True`` the solver backpropagates this loss after every
        batch; because the loss is linear in the integral, the gradients of the
        batches add up to the gradient of the whole integral.

        Parameters
        ----------
        result : IntegrationResult
            The result of the current batch.
        *args : Any
            Accepted and ignored; the solver passes only ``result``.
        **kwargs : Any
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            Loss tensor for backpropagation. Shape: same as result.integral.
        """
        return result.integral

    # -------------------------------------------------------------------------------- #
    #                                    DESTRUCTOR                                    #
    # -------------------------------------------------------------------------------- #

    def __del__(self) -> None:
        """Destructor that cleans up the distributed process group.

        Does nothing if construction failed before the distributed state was
        set, so a constructor error is not followed by a second, unrelated one.
        """
        if hasattr(self, "is_distributed"):
            self.end_process()
