"""
Distributed environment setup for multi-GPU and multi-node execution.

Manages PyTorch distributed process groups, device assignment, and SLURM
integration. This is the base class for all solvers, providing device
and rank management even in single-process (non-distributed) mode.

Refactored from rsarm's contribution at:
https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
This program is distributed under the MIT License (see MIT.md)
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _resolve_device(
    device: str | torch.device | None, index_default: int, distributed: bool = False
) -> torch.device:
    """
    Normalize a requested device to one ``torch.device``.

    CPU resolves to the bare ``torch.device("cpu")``; CUDA always resolves to an
    explicit index, so devices compare equal regardless of how they were spelled
    (``torch.device("cuda") != torch.device("cuda", 0)``). Only ``cpu`` and
    ``cuda`` are supported because the memory queries handle only those two.

    Parameters
    ----------
    device : str | torch.device | None
        Requested device, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``, or
        ``torch.device("cuda", 1)``. None auto-detects (CUDA if available).
    index_default : int
        GPU index used when ``device`` carries no index.
    distributed : bool
        If True, an explicit index must equal ``index_default``
        (the GPU assigned to this rank).

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    ValueError
        If the device type is unsupported, CUDA is unavailable, the
        index conflicts with this rank's GPU, or the index is not visible.
    RuntimeError, TypeError
        Raised by ``torch.device`` for an unparsable
        string or an unsupported argument type.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            f"device must be 'cpu' or an available 'cuda' device, got {device}"
        )
    if distributed and device.index not in (None, index_default):
        raise ValueError(
            f"{device} conflicts with this rank's GPU cuda:{index_default}; "
            "pass 'cuda' or None in distributed runs"
        )
    index = index_default if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"cuda:{index} requested but only {torch.cuda.device_count()} GPU(s) "
            "are visible"
        )
    return torch.device("cuda", index)


class DistributedEnvironment:
    """
    Manages the distributed computing environment for integration solvers.

    Handles both single-process and multi-process (distributed) execution.
    In single-process mode, sets sensible defaults (rank=0, world_size=1).
    In distributed mode, initializes a PyTorch process group using environment
    variables (RANK, LOCAL_RANK, WORLD_SIZE) or SLURM equivalents.

    This class is the root of the solver class hierarchy. All solvers inherit
    from it to get device management, even when not running distributed.

    Attributes
    ----------
    is_distributed : bool
        Whether running in a multi-process distributed setup.
    is_slurm : bool
        Whether the job is running under SLURM workload manager.
    is_master : bool
        Whether this process is the master (rank 0).
    rank : int
        Global rank of this process across all nodes.
    local_rank : int
        Rank of this process within its node (used for GPU assignment).
    world_size : int
        Total number of processes across all nodes.
    seed_offset : int
        Offset for random seeds to ensure different processes
        use different random sequences (equals rank).
    device : torch.device
        Resolved compute device, ``torch.device("cpu")`` or
        ``torch.device("cuda", index)``; the single source of truth for
        where the solver runs.
    """

    is_distributed: bool
    is_slurm: bool
    is_master: bool
    rank: int
    local_rank: int
    world_size: int
    seed_offset: int
    device: torch.device

    def __init__(
        self,
        backend: str = "nccl",
        device: str | torch.device | None = None,
        master_addr: str | None = None,
        master_port: str | None = None,
        is_slurm: bool = False,
    ) -> None:
        """
        Initialize the distributed environment.

        Detects whether distributed execution is active, initializes the
        appropriate process group, and assigns a compute device.

        Parameters
        ----------
        backend : str
            Communication backend for distributed training.
            Options: 'nccl' (GPU), 'gloo' (CPU/GPU), 'mpi'.
        device : str | torch.device | None
            Device to run on: 'cpu', 'cuda', an indexed 'cuda:N', a
            torch.device, or None to auto-detect (CUDA if available). An
            index-less 'cuda' uses this rank's GPU (local_rank when
            distributed with several GPUs visible, else 0). In distributed
            runs an explicit index must match that GPU.
        master_addr : str | None
            IP address of the master node. Required (along with
            master_port) for distributed mode if not set via environment.
        master_port : str | None
            Port on the master node. Required (along with
            master_addr) for distributed mode if not set via environment.
        is_slurm : bool
            If True, reads rank/world_size from SLURM environment
            variables instead of standard PyTorch distributed variables.
        """
        self.is_slurm = is_slurm
        self.backend = backend.lower()
        # Determine if this process is part of a distributed job
        self.check_distributed()

        # Initialize process group or set single-process defaults
        if self.is_distributed:
            self.init_distributed_process(master_addr, master_port)
        else:
            self.init_single_process()

        # Assign a specific device (e.g. cuda:1) from the request and local rank
        self.set_device(device)

    def check_distributed(self) -> bool:
        """
        Detect whether this process is part of a distributed job.

        For SLURM jobs, checks SLURM_NTASKS > 1. For standard PyTorch
        distributed, checks if the RANK environment variable is set.
        Also validates that the requested communication backend is available.

        Returns
        -------
        bool
            True if running in distributed mode, False otherwise.
        """
        if self.is_slurm:
            self.is_distributed = int(os.environ["SLURM_NTASKS"]) > 1
        else:
            self.is_distributed = int(os.environ.get("RANK", "-1")) != -1

        # Verify the requested distributed backend is available
        if self.is_distributed:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if self.backend == "nccl":
                assert dist.is_nccl_available()
            elif self.backend == "mpi":
                assert dist.is_mpi_available()
            elif self.backend == "gloo":
                assert dist.is_gloo_available()
            else:
                raise SyntaxWarning(f"Cannot check if {self.backend} is available.")

        return self.is_distributed

    def init_single_process(self) -> None:
        """
        Set default values for single-process (non-distributed) execution.

        Sets rank=0, world_size=1, and marks this process as master.
        """
        self.is_master = True
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.seed_offset = 0

    def init_slurm_environment(self) -> tuple[str, str]:
        """
        Map SLURM environment variables to PyTorch distributed variables.

        Translates SLURM_NTASKS -> WORLD_SIZE, SLURM_LOCALID -> LOCAL_RANK,
        SLURM_PROCID -> RANK, and determines the master node address.

        Returns
        -------
        tuple[str, str]
            Tuple of (master_addr, master_port).

        Raises
        ------
        ValueError
            Currently always raised because the hostlist package
            dependency is deprecated. Needs a replacement for hostname
            resolution from SLURM_JOB_NODELIST.
        """
        # hostname = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])[0]
        hostname = "UNKNOWN"
        raise ValueError(
            "Find new way to get hostname, hostlist package uses deprecated dependencies"
        )
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33333")
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        return os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]

    def init_distributed_process(
        self, master_addr: str | None, master_port: str | None
    ) -> None:
        """
        Initialize a PyTorch distributed process group.

        Reads rank, local_rank, and world_size from environment variables.
        Optionally sets MASTER_ADDR and MASTER_PORT if provided. Then
        initializes the process group using the configured backend.

        Parameters
        ----------
        master_addr : str | None
            IP address of the master node, or None to use
            the existing MASTER_ADDR environment variable.
        master_port : str | None
            Port of the master node, or None to use
            the existing MASTER_PORT environment variable.

        Raises
        ------
        ValueError
            If only one of master_addr/master_port is provided
            (both or neither must be given).
        """
        if self.is_slurm:
            addr, port = self.init_slurm_environment()
            master_addr = addr if master_addr is None else master_addr
            master_port = port if master_port is None else master_port

        # Read process identity from environment
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.is_master = self.rank == 0
        self.seed_offset = self.rank

        # Validate that master address and port are both provided or both absent
        master_addr_port_none = (master_addr is None) and (master_port is None)
        master_addr_port_not_none = (master_addr is not None) and (
            master_port is not None
        )
        if not master_addr_port_none and not master_addr_port_not_none:
            raise ValueError("Master address and port must both be specified")

        if master_addr is not None and master_port is not None:
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
        self.master_addr = os.environ.get("MASTER_ADDR", None)
        self.master_port = os.environ.get("MASTER_PORT", None)

        # Initialize the distributed process group for inter-process communication
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            init_method="env://",
        )

    def set_device(self, device: str | torch.device | None = None) -> None:
        """
        Resolve and assign this process's compute device to ``self.device``.

        An index-less request uses the GPU matching local_rank when several GPUs
        are visible, else GPU 0. Only distributed runs set the process-wide CUDA
        device, because NCCL collectives use the current device; single-process
        runs leave global torch state unchanged.

        Parameters
        ----------
        device : str | torch.device | None
            Requested device, as accepted by ``_resolve_device``.

        Raises
        ------
        ValueError
            If the device is invalid (see ``_resolve_device``).
        """
        self.device = _resolve_device(
            device,
            self.local_rank if torch.cuda.device_count() > 1 else 0,
            self.is_distributed,
        )
        if self.is_distributed and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        logger.info("Process %s: running on %s", self.rank, self.device)

    def end_process(self) -> None:
        """
        Clean up the distributed process group.

        Only takes action in distributed mode. Should be called when the
        solver is destroyed (called automatically by SolverBase.__del__).
        """
        if self.is_distributed:
            dist.destroy_process_group()

    def get_cuda_memory_usage(self) -> tuple[int, int] | int:
        """
        Query CUDA memory info for this process's device.

        Returns
        -------
        tuple[int, int] | int
            Tuple of (free_memory_bytes, total_memory_bytes) from
            torch.cuda.mem_get_info, or -1 if not running on CUDA.
        """
        if self.device.type != "cuda":
            return -1
        return torch.cuda.mem_get_info(self.device)
