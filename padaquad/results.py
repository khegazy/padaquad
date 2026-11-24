"""
Result dataclasses of an integration.

``IntegrationResult`` is what ``integrate()`` and a solver's ``integrate``
method return: the integral, its error estimate, the mesh to warm-start the
next call from, and per-subinterval diagnostics.

``MethodOutput`` is internal. It holds the integral and the error estimate of
one batch of subintervals, as computed by the solver's
``_calculate_integral``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class IntegrationState:
    """
    Where an ``integrate`` call ended, for the next call to start from.

    ``integrate`` returns it beside the result, ``result, state =
    solver.integrate(...)``; pass ``state=state`` to the next call of the
    same, or a slightly changed, integrand. It never changes after it is made: ``integrate``
    returns a new one, and ``dataclasses.replace(state, ...)`` makes an edited
    copy. ``dataclasses.asdict(state)`` gives a dictionary that ``torch.save``
    and ``torch.load`` (with its default ``weights_only=True``) handle, and
    ``IntegrationState(**d)`` rebuilds it. See :ref:`ug-reuse`.

    Attributes
    ----------
    mesh : torch.Tensor or None
        The mesh to start from, shape [M, T]: the optimal mesh of the call
        that returned the state (low-error subintervals merged, failing ones
        split), detached from autograd, on the solver's device, its ends equal
        to the bounds. None makes a fresh mesh on the bounds, which must then
        be given.
    max_batch : int
        The batch size to start with: the largest number of points per call of
        the integrand known to fit.
    max_batch_converged : bool
        True when the batch size will not grow again: it was given by the user,
        or a call ran out of memory. While False, a call that needs larger
        batches tries 10 times the batch size (see :ref:`ug-memory`).

    Raises
    ------
    ValueError
        If ``mesh`` is not a tensor or None, ``max_batch`` is not an int of at
        least 1, or ``max_batch_converged`` is not a bool.
    """

    mesh: torch.Tensor | None
    max_batch: int
    max_batch_converged: bool

    def __post_init__(self) -> None:
        """Validate the fields (see the class's Raises)."""
        if not (self.mesh is None or isinstance(self.mesh, torch.Tensor)):
            raise ValueError("IntegrationState.mesh must be a tensor or None.")
        if isinstance(self.max_batch, bool) or not isinstance(self.max_batch, int):
            raise ValueError("IntegrationState.max_batch must be an int.")
        if self.max_batch < 1:
            raise ValueError(
                f"IntegrationState.max_batch must be at least 1; got {self.max_batch}."
            )
        if not isinstance(self.max_batch_converged, bool):
            raise ValueError("IntegrationState.max_batch_converged must be a bool.")


@dataclass(kw_only=True)
class IntegrationResult:
    r"""
    Result of an integration: the integral, its error estimate, the mesh and diagnostics.

    ``integrate`` returns it with an ``IntegrationState``, the mesh and batch
    size to start the next call from: ``result, state = solver.integrate(...)``.

    Every field is required and passed by keyword (``kw_only=True``), except
    ``tracked_variables``, ``loss``, ``converged``, ``n_start_steps``,
    ``n_start_steps_split`` and ``max_split_depth``, which default to
    ``None``. Below, ``N`` is the number of accepted subintervals,
    ``C`` the number of nodes per subinterval, ``P = C + (N - 1)(C - 1)`` the
    number of distinct nodes, and ``M`` the number of mesh points (see
    :ref:`ug-shapes`).

    ``integral``, ``integral_error``, ``y0``, ``mesh_init`` and
    ``mesh_final`` are on the solver's device; the other tensors are on
    ``result_device`` (default ``"cpu"``).

    The solver also builds one result per batch of accepted subintervals and
    passes it to ``loss_fxn``. That result covers only the batch: ``integral``
    is the batch's contribution, ``y0`` is zero, ``loss``, ``converged`` and
    the split counts are ``None``, and ``nodes``, ``y`` and
    ``tracked_variables`` keep one row per subinterval (shapes ``[N, C, T]``,
    ``[N, C, D]`` and ``[N, C, ...]``) instead of the flattened shapes below.
    Its tensors are on the solver's device, except ``y`` and
    ``tracked_variables`` with ``integrate(trade_speed_for_memory=True)``,
    which are on ``result_device`` (see :ref:`ug-trade-speed-for-memory`).
    With ``integrate(record_evaluations=False)``, ``y`` and
    ``tracked_variables`` are ``None`` in both results.

    Attributes
    ----------
    integral : torch.Tensor
        ``y0`` plus the integral of ``f`` from ``mesh_init`` to
        ``mesh_final``. Shape ``[D]``.
    integral_error : torch.Tensor
        Sum of the subintervals' signed error estimates. Shape ``[D]``. With
        ``enforce_convergence``, subintervals accepted and later split again
        are removed from it exactly. See :ref:`ug-error-control`.
    mesh_init : torch.Tensor
        Lower integration bound. Shape ``[T]``.
    mesh_final : torch.Tensor
        Upper integration bound. Shape ``[T]``.
    nodes : torch.Tensor
        Nodes of all accepted subintervals in one non-decreasing sequence; the
        node shared by two neighbouring subintervals appears once. Shape
        ``[P, T]``.
    h : torch.Tensor
        Width of each accepted subinterval. Shape ``[N, T]``.
    y : torch.Tensor or None
        Integrand values at ``nodes``. Shape ``[P, D]``. ``None`` with
        ``integrate(record_evaluations=False)``.
    tracked_variables : tuple of torch.Tensor or None
        Extra values the integrand returned beside its values, at ``nodes``
        and not integrated; ``None`` if it returned none. Each tensor is
        detached, with shape ``[P, ...]``. See :ref:`ug-tracked-variables`.
    mesh_quadratures : torch.Tensor
        Each subinterval's contribution to the integral,
        :math:`h \sum_i b_i y_i`. Shape ``[N, D]``.
    mesh_quadrature_errors : torch.Tensor
        Absolute value of each subinterval's error estimate,
        :math:`|h \sum_i b^{\text{error}}_i y_i|`. Shape ``[N, D]``.
    error_ratios : torch.Tensor
        Each subinterval's error ratio when it was accepted, against the
        tolerance of that moment (with ``error_norm="failure_fraction"``,
        the fraction of failing outputs instead). Shape ``[N]``.
    loss : torch.Tensor or None
        Sum over batches of ``loss_fxn`` applied to the per-batch results,
        minus ``loss_fxn`` of the subintervals the re-check removed (so a loss
        that is a sum over subintervals is that of the returned ones); by
        default the integral without ``y0``. See :ref:`ug-autodiff`.
    gradient_taken : bool
        The call's ``take_gradient``: whether ``loss.backward()`` ran after
        each batch (it runs only when the loss requires a gradient).
    y0 : torch.Tensor
        The offset added to the integral, after conversion to the solver's
        dtype and device. Shape ``[D]``.
    converged : bool or None
        ``True`` if every accepted subinterval meets the tolerance judged
        against the final integral (in cumulative mode, the final running
        integral through it); otherwise ``False``, with a logged warning.
        See :ref:`ug-convergence`.
    n_start_steps : int or None
        Number of subintervals in the starting mesh (the given ``mesh``, cut to
        the bounds, or a new random one).
    n_start_steps_split : int or None
        Number of starting subintervals that were split at least once during
        the integration (splits made to build the state's mesh do not count).
    max_split_depth : int or None
        Largest number of times one starting subinterval was halved: 0 if
        none was split, at most ``max_adaptive_splits``.

    Examples
    --------
    >>> import torch
    >>> from padaquad import integrate
    >>> _ = torch.manual_seed(0)  # the starting mesh is random
    >>> result, state = integrate(
    ...     lambda t: torch.exp(t),
    ...     atol=1e-10,
    ...     rtol=1e-10,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([1.0], dtype=torch.float64),
    ...     y0=torch.tensor([1.0], dtype=torch.float64),
    ...     device="cpu",
    ...     max_batch=10_000,
    ... )
    >>> print(round(float(result.integral), 8))  # 1 + (e - 1) = e
    2.71828183
    >>> print(result.converged, abs(float(result.integral_error)) < 1e-10)
    True True
    >>> print(round(float(result.h.sum()), 8))  # the widths add up to 1
    1.0
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    mesh_init: torch.Tensor
    mesh_final: torch.Tensor
    nodes: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor | None
    mesh_quadratures: torch.Tensor
    mesh_quadrature_errors: torch.Tensor
    error_ratios: torch.Tensor
    gradient_taken: bool
    y0: torch.Tensor
    tracked_variables: tuple[torch.Tensor, ...] | None = None
    loss: torch.Tensor | None = None
    converged: bool | None = None
    n_start_steps: int | None = None
    n_start_steps_split: int | None = None
    max_split_depth: int | None = None

    @property
    def percent_start_steps_split(self) -> float | None:
        """
        Percentage of the starting mesh's subintervals that were split at least once.

        Equal to ``100 * n_start_steps_split / n_start_steps``; ``None`` in the
        per-batch result passed to ``loss_fxn``, where the counts are ``None``.

        Returns
        -------
        float or None
            The percentage, from 0 to 100, or None.
        """
        if self.n_start_steps is None or self.n_start_steps_split is None:
            return None
        return 100.0 * self.n_start_steps_split / self.n_start_steps


@dataclass
class MethodOutput:
    """
    Integral and error estimate of one batch of subintervals.

    Produced by ``AdaptiveQuadrature._calculate_integral()``. The field names
    are those of ``IntegrationResult``. The error estimates are detached from
    the autograd graph and keep their sign.

    Attributes
    ----------
    integral : torch.Tensor
        Sum of ``mesh_quadratures`` (plus the ``y0`` passed in, zero inside the
        integration loop). Shape: [D].
    integral_error : torch.Tensor
        Sum of ``mesh_quadrature_errors``. Shape: [D].
    mesh_quadratures : torch.Tensor
        Each subinterval's integral, h_i * sum_j(b_j * y_ij). Shape: [N, D].
    mesh_quadrature_errors : torch.Tensor
        Each subinterval's signed error estimate, h_i * sum_j(b_error_j * y_ij):
        the method's result minus the embedded method's. Shape: [N, D].
    h : torch.Tensor
        Width of each subinterval (last node minus first node). Shape: [N, T].
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    mesh_quadratures: torch.Tensor
    mesh_quadrature_errors: torch.Tensor
    h: torch.Tensor
