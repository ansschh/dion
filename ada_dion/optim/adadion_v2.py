import math
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from dataclasses import dataclass
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate, Shard
from torch.distributed.tensor import randn as dtensor_randn
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, Generator, List, Tuple, Optional, Union

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError:
    _StridedShard = None

from .dion_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    dtensor_from_local,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach

import torch.nn.functional as F
from dion.dion import orthogonalize as dion_orthogonalize


@dataclass
class _DionParamConfig:
    """
    Per-parameter configuration for Dion optimizer.
    """

    # Dimensions of the tensor that is sharded
    outer_shard_tensor_dim: Optional[int] = None
    inner_shard_tensor_dim: Optional[int] = None

    # Dimensions of the device mesh that the tensor is sharded over
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None

    # Use transposed version of the algorithm
    is_transposed: bool = False

    # Whether to all-reduce compressed P and R instead of full gradient
    # This should always be False for 1D tensors
    compressed_all_reduce = False

    # Sharding configurations for the Q matrix
    Q_sharded_placements: Optional[Tuple[Placement]] = None
    Q_inner_unsharded_placements: Optional[Tuple[Placement]] = None


@dataclass
class DionMixedPrecisionConfig:
    """
    Configuration for mixed precision in Dion optimizer.
    None means that optimizer states will use the same dtype as each parameter.
    """

    # Momentum state for all algorithms
    momentum_dtype: Optional[torch.dtype] = None
    # Dion Q matrix
    Q_dtype: Optional[torch.dtype] = None
    # Adam variance state
    variance_dtype: Optional[torch.dtype] = None
    # TODO look into separate dtypes for communication operations


def _quantize_int(x: int, q: int) -> int:
    """Round x to nearest multiple of q (min q)."""
    if q <= 1:
        return x
    return max(q, q * round(x / q))


class AdaDionV2(Optimizer):
    """
    AdaDion V2: Copy of Microsoft Dion with adaptive rank support.

    Without adaptive rank: identical to dion.Dion.
    With adaptive rank: slices Q at local tensor level (no DTensor shape changes).
    """

    def __init__(
        self,
        params: ParamsT,
        replicate_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        outer_shard_mesh: Optional[DeviceMesh] = None,
        inner_shard_mesh: Optional[DeviceMesh] = None,
        replicate_mesh_grad_sync: bool = True,
        rank_fraction_max: float = 0.7,
        rank_multiple_of: int = 1,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        epsilon: float = 1e-8,
        power_iters: int = 1,
        qr_method: str = "rcqr",
        cqr_warmup_steps: int = 150,
        rcqr_oversample: float = 1.25,
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
        # --- Adaptive rank ---
        adaptive_rank: bool = False,
        init_rank_fraction: float = 0.25,
        erank_ema_beta: float = 0.9,
        rank_scale: float = 1.5,
        rank_min: int = 16,
        rank_quantize: int = 8,
        rank_step_up: int = 16,
        rank_step_down: int = 8,
        use_quality_control: bool = False,
        aerr_ema_beta: float = 0.95,
        aerr_target: float = 0.08,
        aerr_up_margin: float = 0.15,
        aerr_down_margin: float = 0.15,
        adapt_step: int = 1,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank_fraction_max <= 0 or rank_fraction_max > 1:
            raise ValueError(f"Invalid rank_fraction_max: {rank_fraction_max}")
        if rank_multiple_of <= 0:
            raise ValueError(f"Invalid rank multiple of: {rank_multiple_of}")
        if power_iters != 1:
            raise ValueError("Async Dion only supports power_iters=1")
        if qr_method != "rcqr":
            raise ValueError("Async Dion only supports qr_method='rcqr'")

        # Check device mesh
        if replicate_mesh is not None:
            if not isinstance(replicate_mesh, (DeviceMesh, ProcessGroup)):
                raise TypeError(
                    f"Replicate mesh must be a DeviceMesh or ProcessGroup, but got {type(replicate_mesh)}."
                )
        if outer_shard_mesh is not None:
            if not isinstance(outer_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Outer shard mesh must be a DeviceMesh, but got {type(outer_shard_mesh)}."
                )
            if outer_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Outer shard mesh must be 1D, but got {outer_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if outer_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Outer shard mesh must be different from replicate mesh."
                )
        if inner_shard_mesh is not None:
            if not isinstance(inner_shard_mesh, DeviceMesh):
                raise TypeError(
                    f"Inner shard mesh must be a DeviceMesh, but got {type(inner_shard_mesh)}."
                )
            if inner_shard_mesh.ndim != 1:
                raise ValueError(
                    f"Inner shard mesh must be 1D, but got {inner_shard_mesh.ndim}D. Try using a 1D sub-mesh."
                )
            if inner_shard_mesh == replicate_mesh:
                raise ValueError(
                    "Inner shard mesh must be different from replicate mesh."
                )
            if inner_shard_mesh == outer_shard_mesh:
                raise ValueError("Outer and inner shard meshes must be different.")

        # Adaptive rank config
        self._adaptive_rank = adaptive_rank
        self._ada_config = dict(
            init_rank_fraction=init_rank_fraction,
            erank_ema_beta=erank_ema_beta,
            rank_scale=rank_scale,
            rank_min=rank_min,
            rank_quantize=rank_quantize,
            rank_step_up=rank_step_up,
            rank_step_down=rank_step_down,
            use_quality_control=use_quality_control,
            aerr_ema_beta=aerr_ema_beta,
            aerr_target=aerr_target,
            aerr_up_margin=aerr_up_margin,
            aerr_down_margin=aerr_down_margin,
            adapt_step=adapt_step,
        )

        # Default arguments for each param group
        defaults = dict(
            rank_fraction=rank_fraction_max,
            rank_multiple_of=rank_multiple_of,
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            oversample=rcqr_oversample,
            algorithm="dion",
            step=0,
        )
        super().__init__(params, defaults)

        # This is intentionally not in self.state so it doesn't get checkpointed
        # State here may change upon resharding a checkpoint, so we recompute it
        self._param_config: Dict[Tensor, _DionParamConfig] = {}

        self._replicate_mesh = replicate_mesh
        self._outer_shard_mesh = outer_shard_mesh
        self._inner_shard_mesh = inner_shard_mesh
        self._replicate_mesh_grad_sync = replicate_mesh_grad_sync

        # Get world size for the replicate mesh
        if isinstance(replicate_mesh, DeviceMesh):
            self._replicate_world_size = replicate_mesh.size()
        elif isinstance(replicate_mesh, ProcessGroup):
            self._replicate_world_size = dist.get_world_size(replicate_mesh)
        elif replicate_mesh is None:
            self._replicate_world_size = 1
        else:
            raise TypeError(f"Invalid replicate mesh type: {type(replicate_mesh)}.")

        # Get global ranks for outer and inner shard meshes
        if self._outer_shard_mesh is not None:
            self._outer_shard_ranks = dist.get_process_group_ranks(
                self._outer_shard_mesh.get_group()
            )
        else:
            self._outer_shard_ranks = None
        if self._inner_shard_mesh is not None:
            self._inner_shard_ranks = dist.get_process_group_ranks(
                self._inner_shard_mesh.get_group()
            )
        else:
            self._inner_shard_ranks = None

        # Mixed precision
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config
        # TODO check what happens when loading state dict with different precision

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "dion":
                dion_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        dion_tasks = self._create_dion_tasks(dion_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    @torch.no_grad()
    def synchronize_for_checkpoint(self):
        """
        Synchronize the internal optimizer states across the replicated mesh.

        Dion uses compressed gradient synchronization with decoupled momentum, which
        results in optimizer states diverging across the replicated data-parallel mesh.
        To ensure consistency of distributed checkpoints, we must manually synchronize
        the optimizer states before saving a checkpoint. If replicate_mesh is None or
        replicate_mesh_grad_sync is False, this function is a no-op.
        """

        if self._replicate_mesh is None or not self._replicate_mesh_grad_sync:
            # Nothing to do
            return

        # Get all tensors in optimizer states
        state_tensors: List[Tensor] = []
        for state in self.state.values():
            assert isinstance(state, dict)
            for val in state.values():
                if isinstance(val, Tensor):
                    state_tensors.append(val)

        # Heuristic to determine a reasonable batch size
        if self._inner_shard_mesh is not None:
            batch_size = self._inner_shard_mesh.size()
        elif self._outer_shard_mesh is not None:
            batch_size = self._outer_shard_mesh.size()
        else:
            batch_size = self._replicate_world_size

        # Batching allows for coalesced all-reduce
        for batch in create_param_batches(state_tensors, batch_size):
            batch = to_local(batch)
            reduced_batch = all_reduce_replicate_mesh(
                batch, self._replicate_mesh, return_dtensor=False
            )
            torch._foreach_copy_(batch, reduced_batch)

    def _create_dion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Dion matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            oversample = torch.tensor(group["oversample"])

            # Split parameters in this param group by sharding
            split_param_dict = self._split_params_by_sharding(group_params)
            for sharding_type, split_params in split_param_dict.items():
                if not split_params:
                    continue

                # Pick the appropriate update function and batch size based on sharding
                # Batch size should equal the number of devices in the mesh
                if sharding_type == "inner_sharded":
                    dion_update_func = dion_update_fsdp_tp
                    batch_size = self._inner_shard_mesh.size()
                    use_dtensor = True
                elif sharding_type == "outer_sharded":
                    dion_update_func = dion_update_fsdp
                    batch_size = self._outer_shard_mesh.size()
                    use_dtensor = True
                elif sharding_type == "non_sharded":
                    dion_update_func = dion_update_ddp
                    batch_size = self._replicate_world_size
                    use_dtensor = False
                else:
                    raise RuntimeError("Unknown sharding type")

                # Create batches of parameters
                for params in create_param_batches(split_params, batch_size):
                    gradients = [p.grad for p in params]
                    states = [self._get_or_initialize_state(p, group) for p in params]
                    momentums = [s["momentum"] for s in states]
                    Qs = [s["Q"] for s in states]
                    param_config = self._get_dion_param_config(params[0])

                    # Adaptive rank: pass r_list and state refs
                    r_list = None
                    ada_states = None
                    if self._adaptive_rank:
                        r_list = [s["r"] for s in states]
                        ada_states = states  # pass state refs for erank update

                    if not use_dtensor:
                        params = to_local(params)
                        gradients = to_local(gradients)
                        momentums = to_local(momentums)
                        Qs = to_local(Qs)

                    dion_kwargs = dict(
                        X=pad_batch(params, batch_size),
                        G=pad_batch(gradients, batch_size),
                        M=pad_batch(momentums, batch_size),
                        Q=pad_batch(Qs, batch_size),
                        lr=lr, mu=mu, weight_decay=weight_decay,
                        epsilon=epsilon, param_config=param_config,
                        replicate_mesh=self._replicate_mesh,
                        replicate_mesh_grad_sync=self._replicate_mesh_grad_sync,
                        oversample=oversample,
                        r_list=r_list, ada_states=ada_states,
                        ada_config=self._ada_config if self._adaptive_rank else None,
                    )
                    if use_dtensor:  # only FSDP paths need outer_shard_mesh
                        dion_kwargs["outer_shard_mesh"] = self._outer_shard_mesh
                    yield AsyncTask(dion_update_func(**dion_kwargs))

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                lion_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, group) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            # If replicate_mesh_grad_sync is True, the optimizer needs to all-reduce gradients
            # If replicate_mesh_grad_sync is False, gradients are already synchronized
            all_reduce_mesh = (
                self._replicate_mesh if self._replicate_mesh_grad_sync else None
            )

            yield AsyncTask(
                adamw_update_allreduce_grad(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                    replicate_mesh=all_reduce_mesh,
                )
            )

    def _get_or_initialize_state(self, param: Tensor, group: dict) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        algo = group["algorithm"]
        if not state:
            if algo == "dion":
                self._init_opt_state_dion(
                    param,
                    state,
                    rank_fraction=group["rank_fraction"],
                    rank_multiple_of=group["rank_multiple_of"],
                )
            elif algo == "adamw":
                self._init_opt_state_adam(param, state)
            elif algo == "lion" or algo == "clion":
                self._init_opt_state_momentum(param, state)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")
        return state

    def _get_dion_param_config(self, x: Tensor) -> _DionParamConfig:
        """
        Get the Dion-specific parameter configuration for a given tensor.
        If the configuration is not already initialized, it will be created.
        Lazy initialization is necessary because PyTorch allows new parameters
        to be added to the optimizer after it has been created.
        """
        if x in self._param_config:
            return self._param_config[x]

        if x.ndim > 2:
            raise NotImplementedError(
                f"Tensors with more than 2 dimensions are not supported. Got {x.ndim}D tensor."
            )

        # Check for allowed DeviceMesh and DTensor combinations
        # We only allow DTensor + DeviceMesh or regular Tensor + ProcessGroup
        using_device_mesh = (
            isinstance(self._replicate_mesh, DeviceMesh)
            or isinstance(self._outer_shard_mesh, DeviceMesh)
            or isinstance(self._inner_shard_mesh, DeviceMesh)
        )
        using_process_group = isinstance(self._replicate_mesh, ProcessGroup)
        if using_device_mesh and not isinstance(x, DTensor):
            raise TypeError("When using DeviceMesh, all parameters must be DTensor.")
        if using_process_group and isinstance(x, DTensor):
            raise TypeError(
                "When using DTensor parameters, the data parallel group must be specified by a DeviceMesh instead of ProcessGroup."
            )

        # State is initialized for both matrix and scalar parameters
        config = _DionParamConfig()

        # By default, we transpose matrices so that dim0 >= dim1
        # This can change depending on sharding
        if x.ndim == 2:
            m, n = x.shape
            config.is_transposed = m < n

        # Detect sharding dimensions for DTensor
        if isinstance(x, DTensor) and x.ndim == 2:
            device_mesh = x.device_mesh
            placements = x.placements
            assert len(placements) == device_mesh.ndim

            dim_map = [None for _ in range(x.ndim)]

            for mesh_dim, placement in enumerate(placements):
                # StridedShard not allowed
                if _StridedShard is not None and isinstance(placement, _StridedShard):
                    raise NotImplementedError(
                        f"StridedShard is not supported. Ensure that FSDP and TP shard different dimensions of each matrix."
                    )

                # Skip non-sharded device mesh dimensions
                if not placement.is_shard():
                    continue
                tensor_dim = placement.dim

                # Check for double sharding on same tensor dimension
                if dim_map[tensor_dim] is not None:
                    raise RuntimeError(
                        f"Got double-sharded DTensor for tensor dimension {placement.dim}."
                    )
                dim_map[tensor_dim] = mesh_dim

                # Get global ranks corresponding to this mesh dimension
                mesh_dim_ranks = dist.get_process_group_ranks(
                    device_mesh.get_group(mesh_dim)
                )

                # Check if it matches the outer or inner shard ranks
                outer_sharded, inner_sharded = False, False
                if mesh_dim_ranks == self._outer_shard_ranks:
                    config.outer_shard_tensor_dim = tensor_dim
                    config.outer_shard_mesh_dim = mesh_dim
                    outer_sharded = True
                if mesh_dim_ranks == self._inner_shard_ranks:
                    config.inner_shard_tensor_dim = tensor_dim
                    config.inner_shard_mesh_dim = mesh_dim
                    inner_sharded = True

                # Check for double sharding on same mesh dimension
                if outer_sharded and inner_sharded:
                    raise RuntimeError(
                        "Cannot have outer and inner sharding over the same process group."
                    )

                # Check for sharding on unrecognized mesh dimension
                # Ignore edge case for single GPU "sharding" = Replicate()
                # Make sure to check that size(mesh_dim) > 1
                if (
                    device_mesh.size(mesh_dim) > 1
                    and not outer_sharded
                    and not inner_sharded
                ):
                    raise RuntimeError(
                        f"Got DTensor sharded on unrecognized {mesh_dim=}, which does not match outer_shard_mesh or inner_shard_mesh."
                    )

            # Set transpose so that orthogonalization happens over the inner sharding dimension
            # Standard Dion orthogonalizes over tensor dimension 0
            if config.inner_shard_tensor_dim == 0 or config.outer_shard_tensor_dim == 1:
                config.is_transposed = False
            # Transposed Dion orthogonalizes over tensor dimension 1
            if config.outer_shard_tensor_dim == 0 or config.inner_shard_tensor_dim == 1:
                config.is_transposed = True

        self._param_config[x] = config
        return config

    def _split_params_by_sharding(
        self, params: List[Tensor]
    ) -> Dict[str, List[Tensor]]:
        """
        Sort parameters into inner-sharded, outer-sharded, and non-sharded lists.
        This determines the parallelization strategy used to compute the update.
        The "inner sharding" dimension needs to use distributed orthogonalization.
        """
        inner_sharded = []
        outer_sharded = []
        non_sharded = []

        for p in params:
            config = self._get_dion_param_config(p)
            if config.inner_shard_mesh_dim is not None:
                inner_sharded.append(p)
            elif config.outer_shard_mesh_dim is not None:
                outer_sharded.append(p)
            else:
                non_sharded.append(p)

        return {
            "inner_sharded": inner_sharded,
            "outer_sharded": outer_sharded,
            "non_sharded": non_sharded,
        }

    def _init_opt_state_momentum(self, param: Tensor, state: Dict[str, Any]):
        # Create the momentum buffer
        # If param is DTensor, this will also be a DTensor
        state["momentum"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.momentum_dtype
        )

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        self._init_opt_state_momentum(param, state)
        state["variance"] = torch.zeros_like(
            param, dtype=self._mixed_precision_config.variance_dtype
        )

    def _init_opt_state_dion(
        self,
        param: Tensor,
        state: Dict[str, Any],
        rank_fraction: float,
        rank_multiple_of: int,
    ):
        """
        Initialize the optimizer state for Dion.
        This includes the momentum buffer and the Q matrix.

        The low-rank factor `r` is computed as `rank_fraction` * min(m, n),
        and rounded up to the next multiple of `rank_multiple_of`.
        """
        if param.ndim != 2:
            raise ValueError(
                f"Expected Dion parameters to be 2D matrix, but got {param.ndim}D. "
                f"For scalar parameters, set 'algorithm' to 'lion' or 'adamw' when creating param group."
            )

        param_config = self._get_dion_param_config(param)
        self._init_opt_state_momentum(param, state)

        # Compute the low-rank factor r
        m, n = param.shape
        r = rank_fraction * min(m, n)
        r = rank_multiple_of * math.ceil(r / rank_multiple_of)
        r = min(r, m, n)
        Q_shape = (m, r) if param_config.is_transposed else (n, r)

        # Set compressed_all_reduce based on if it saves communication cost
        # Otherwise we will all-reduce the gradient matrix instead
        if rank_fraction < 1 and (m + n) * r < m * n:
            param_config.compressed_all_reduce = True

        # Get dtype for Q
        if self._mixed_precision_config.Q_dtype is not None:
            Q_dtype = self._mixed_precision_config.Q_dtype
        else:
            Q_dtype = param.dtype

        if isinstance(param, DTensor):
            # Directly construct Q as DTensor
            # Shard(0) on outer sharding mesh and Shard(1) on inner sharding mesh
            placements = [Replicate() for _ in range(param.device_mesh.ndim)]
            if param_config.outer_shard_mesh_dim is not None:
                placements[param_config.outer_shard_mesh_dim] = Shard(0)
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Shard(1)
            param_config.Q_sharded_placements = tuple(placements)

            # Q is unsharded along the inner sharding dimension only
            if param_config.inner_shard_mesh_dim is not None:
                placements[param_config.inner_shard_mesh_dim] = Replicate()
                param_config.Q_inner_unsharded_placements = tuple(placements)
            else:
                # No inner sharding, so placements are the same as Q_sharded_placements
                param_config.Q_inner_unsharded_placements = None

            # DTensor RNG should automatically produce identical results across DP replicas
            Q = dtensor_randn(
                Q_shape,
                device_mesh=param.device_mesh,
                dtype=Q_dtype,
                placements=param_config.Q_sharded_placements,
            )

        else:
            # Make sure all DP ranks have the same Q
            Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype)
            self._replicate_mesh_broadcast(Q)

        if not self._adaptive_rank:
            # Non-adaptive: store Q as-is (DTensor or regular)
            state["Q"] = Q
        else:
            # Adaptive: extract Q as plain local tensor for sliceable access.
            # Q was already created above via dtensor_randn (same as Dion),
            # so we just extract the local tensor to enable dynamic rank slicing.
            # Also store M as plain local tensor.
            r_cap = int(r)
            dmin = min(m, n)
            init_r = self._ada_config["init_rank_fraction"] * dmin
            init_r = rank_multiple_of * math.ceil(init_r / rank_multiple_of)
            init_r = int(min(max(1, init_r), r_cap))

            p_local = to_local(param) if isinstance(param, DTensor) else param

            # For non-transposed: Q is Shard(0) DTensor (n/world, r).
            # Adaptive path needs full (n, r) replicated for slicing.
            # For transposed: Q is Shard(0) DTensor (m/world, r).
            # Adaptive path needs local shard (m_local, r) as-is.
            if isinstance(Q, DTensor) and not param_config.is_transposed:
                Qbuf = Q.redistribute(placements=[Replicate()]).to_local().clone()
            else:
                Qbuf = (to_local(Q) if isinstance(Q, DTensor) else Q).clone()

            # Store as plain tensors (not DTensor)
            state["Q"] = Qbuf  # plain tensor, never changes shape
            state["momentum"] = torch.zeros_like(p_local)  # override DTensor momentum
            state["r"] = init_r
            state["r_cap"] = r_cap
            state["transpose"] = param_config.is_transposed
            state["erank_ema"] = None
            state["aerr_ema"] = None
            state["ada_step"] = 0

    def _replicate_mesh_broadcast(self, tensor: Tensor):
        """
        Broadcast a tensor from rank 0 over the replicated data-parallel world.
        Tensor is modified in place.
        """
        if self._replicate_mesh is None:
            # No data parallelism used, do nothing
            pass
        elif isinstance(self._replicate_mesh, DeviceMesh):
            for group in self._replicate_mesh.get_all_groups():
                dist.broadcast(tensor, group=group, group_src=0)
        elif isinstance(self._replicate_mesh, ProcessGroup):
            dist.broadcast(tensor, group=self._replicate_mesh, group_src=0)
        else:
            raise TypeError(
                "Data parallel mesh must be either a DeviceMesh or ProcessGroup."
            )

    # ------------------------------------------------------------------
    # Diagnostic methods (called by BaseHybridOptimizersContainer)
    # ------------------------------------------------------------------

    def get_rank(self) -> dict:
        if not self._adaptive_rank:
            return {}
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm", "dion") != "dion":
                continue
            for p in group["params"]:
                if p.ndim != 2:
                    continue
                state = self.state[p]
                if "r" in state:
                    result[f"param_{idx}"] = state["r"]
                idx += 1
        return result

    def get_effective_rank(self) -> dict:
        if not self._adaptive_rank:
            return {}
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm", "dion") != "dion":
                continue
            for p in group["params"]:
                if p.ndim != 2:
                    continue
                state = self.state[p]
                if state.get("erank_ema") is not None:
                    result[f"param_{idx}"] = state["erank_ema"]
                idx += 1
        return result

    def get_aerr(self) -> dict:
        if not self._adaptive_rank:
            return {}
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm", "dion") != "dion":
                continue
            for p in group["params"]:
                if p.ndim != 2:
                    continue
                state = self.state[p]
                if state.get("aerr_ema") is not None:
                    result[f"param_{idx}"] = state["aerr_ema"]
                idx += 1
        return result


def dion_update_ddp(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    Q: List[Tensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: _DionParamConfig,  # shared for all params in batch
    replicate_mesh: Union[DeviceMesh, ProcessGroup, None] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
    r_list: Optional[List[int]] = None,
    ada_states: Optional[List[Dict]] = None,
    ada_config: Optional[Dict] = None,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for DDP.
    This function does not support sharded matrices.

    Batch size should equal the DDP world size. Each device will
    orthogonalize one full matrix in the batch.
    """
    assert param_config.outer_shard_mesh_dim is None
    assert param_config.outer_shard_tensor_dim is None
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # Get rank and world size
    if isinstance(replicate_mesh, DeviceMesh):
        world_size = replicate_mesh.size()
        device_rank = replicate_mesh.get_rank()
    elif isinstance(replicate_mesh, ProcessGroup):
        world_size = dist.get_world_size(replicate_mesh)
        device_rank = dist.get_rank(replicate_mesh)
    else:
        world_size = 1
        device_rank = 0
    assert (
        len(X) == world_size
    ), f"Batch size {len(X)} must match DDP world size {world_size}."

    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield

    # Add new gradient to momentum
    torch._foreach_add_(M, G)

    # Compute low-rank approximation of M = P @ Q^T
    # M_batch shape is (batch_size, m, n)
    # P_batch shape is (batch_size, m, r)
    # Q_batch shape is (batch_size, n, r)
    M_batch, Q_batch = tensor_list_to_batch(M, Q, param_config.is_transposed, r_list=r_list)
    P_batch = M_batch @ Q_batch

    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        # Synchronize P across all DDP ranks by reduce-scatter
        # Each rank will orthogonalize one full matrix in the batch
        P_single = funcol.reduce_scatter_tensor(
            P_batch,
            reduceOp="avg",
            scatter_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        # Gradients are already synchronized, P_batch is identical across DDP world
        # We can just take one matrix of the batch
        P_single = P_batch[device_rank : device_rank + 1]

    # Orthogonalize one matrix in the batch
    P_single = orthogonalize(P_single, oversample=oversample)

    # All gather orthogonal P_batch from the per-device single matrices
    if replicate_mesh is not None:
        P_batch = funcol.all_gather_tensor(
            P_single,
            gather_dim=0,  # dim 0 = batch
            group=replicate_mesh,
        )
        yield
    else:
        assert world_size == 1
        P_batch = P_single  # batch size is 1

    # M_batch shape is (batch_size, m, n)
    # P_batch shape is (batch_size, m, r)
    # R_batch shape is (batch_size, n, r)
    R_batch = M_batch.mT @ P_batch

    # Synchronize R across all DDP ranks by all-reduce
    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight decay and weight update
    # X = (1 - lr * weight_decay) * X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    # Update Q in place
    update_Q_matrix_(Q, Q_batch, r_list=r_list)

    # --- Adaptive rank ---
    if ada_states is not None and ada_config is not None:
        _adaptive_rank_update(R_batch, ada_states, ada_config, Q, param_config)


def dion_update_fsdp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: _DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
    r_list: Optional[List[int]] = None,
    ada_states: Optional[List[Dict]] = None,
    ada_config: Optional[Dict] = None,
    outer_shard_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm for FSDP2 sharding.

    When adaptive (r_list is not None): per-param loop on local tensors with
    manual all-reduces and RCQR orthogonalization. Q is stored as plain tensor.

    When non-adaptive (r_list is None): original Dion DTensor batched path.
    """
    assert param_config.inner_shard_mesh_dim is None
    assert param_config.inner_shard_tensor_dim is None

    # --- Adaptive path: per-param on local tensors ---
    if r_list is not None and ada_states is not None:
        yield from _dion_update_fsdp_adaptive(
            X, G, M, Q, lr, mu, weight_decay, epsilon, param_config,
            replicate_mesh, replicate_mesh_grad_sync, oversample,
            r_list, ada_states, ada_config,
            outer_shard_mesh=outer_shard_mesh,
        )
        return

    # --- Non-adaptive path: original Dion DTensor batched code ---
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh)
        yield

    torch._foreach_add_(M, G)

    M_batch, Q_batch = tensor_list_to_batch(M, Q, param_config.is_transposed)
    P_batch: DTensor = M_batch @ Q_batch

    P_single = P_batch.redistribute(
        placements=[Shard(0) if p.is_partial() else p for p in P_batch.placements],
        async_op=True,
    )
    yield

    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        P_single = all_reduce_replicate_mesh(P_single, replicate_mesh)
        yield

    P_single = orthogonalize(P_single, oversample=oversample)

    P_batch = P_single.redistribute(
        placements=[Replicate() for _ in P_single.placements], async_op=True
    )
    yield

    R_batch: DTensor = M_batch.mT @ P_batch

    assert not any(p.is_partial() for p in R_batch.placements)

    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()
        R_sum_sq = local_column_sum_sq(R_batch)
        R_sum_sq = funcol.all_reduce(
            R_sum_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield
        Q_batch = column_normalize(R_batch, full_column_sum_sq=R_sum_sq, epsilon=epsilon)
    else:
        Q_batch = column_normalize(R_batch, epsilon=epsilon)

    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    foreach_baddbmm_(
        X, P_batch, Q_batch,
        alpha=-scaled_lr, beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    update_Q_matrix_(Q, Q_batch)



def _dion_update_fsdp_adaptive(
    X, G, M, Q, lr, mu, weight_decay, epsilon, param_config,
    replicate_mesh, replicate_mesh_grad_sync, oversample,
    r_list, ada_states, ada_config,
    outer_shard_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    Adaptive rank FSDP update — batched, modeled on Dion's DDP path.

    All Q/P/R are padded to r_cap so params with different active ranks can
    be stacked into a single batch tensor. Each rank in the FSDP mesh
    orthogonalizes one matrix via reduce-scatter / all-gather, matching
    Dion's batched parallelism and RNG stream for identical sketches.
    """
    assert outer_shard_mesh is not None, "adaptive FSDP path requires outer_shard_mesh"
    fsdp_group = outer_shard_mesh.get_group()

    num_real = len(ada_states)
    batch_size = len(X)  # includes padding from pad_batch
    transpose = param_config.is_transposed
    r_cap = ada_states[0]["r_cap"]

    # Convert G to local (M is already plain from adaptive init, X stays as DTensor)
    G = [to_local(g) if isinstance(g, DTensor) else g for g in G]
    orig_dtype = M[0].dtype
    # Use Dion's naming: m = full (non-sharded) dim, n_outer = sharded dim
    # After tensor_list_to_batch transpose: M_batch is (batch, m, n_outer)
    n_outer = M[0].shape[0]  # sharded dim (local shard size)
    m = M[0].shape[1]        # full dim

    # --- 1. M += G (in-place, matching DDP path) ---
    torch._foreach_add_(M, G)

    # --- 2. Pad Q to r_cap (zero inactive columns) ---
    Q_padded = []
    for i in range(num_real):
        r_i = ada_states[i]["r"]
        Qbuf = Q[i]
        q_dim = n_outer if transpose else m
        q = Qbuf[:q_dim, :r_cap].clone()
        q[:, r_i:] = 0
        Q_padded.append(q)
    Q_padded += [torch.zeros_like(Q_padded[0])] * (batch_size - num_real)

    # --- 3. Batched P = M @ Q (M already has G added) ---
    M_batch, Q_batch = tensor_list_to_batch(M, Q_padded, transpose)
    P_batch = M_batch @ Q_batch

    # --- 4. Reduce-scatter across FSDP → each rank gets one full matrix ---
    P_single = funcol.reduce_scatter_tensor(
        P_batch, reduceOp="sum", scatter_dim=0, group=fsdp_group,
    )
    yield

    # --- 5. Orthogonalize (one matrix per rank, same as Dion) ---
    P_single = dion_orthogonalize(P_single, oversample=oversample)

    # --- 6. All-gather back to full batch ---
    P_batch = funcol.all_gather_tensor(
        P_single, gather_dim=0, group=fsdp_group,
    )
    yield

    # --- 7. R = M^T @ P (matching DDP path) ---
    # M_batch: (batch, m, n/outer), P_batch: (batch, m, r_cap)
    # R_batch: (batch, n/outer, r_cap) — contracts over m (full dim), no partial sum
    R_batch = M_batch.mT @ P_batch
    assert R_batch.shape == (batch_size, n_outer, r_cap)

    # --- 8. NaN check ---
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # --- 9. Error feedback: M = M - (1-mu) * P @ R^T (matching DDP path) ---
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=transpose
    )

    # --- 10. Column normalize R → Q_new (same as Dion FSDP) ---
    if param_config.outer_shard_mesh_dim is not None:
        R_sum_sq = local_column_sum_sq(R_batch)
        R_sum_sq = funcol.all_reduce(
            R_sum_sq, reduceOp="sum", group=fsdp_group,
        )
        yield
        Q_batch = column_normalize(R_batch, full_column_sum_sq=R_sum_sq, epsilon=epsilon)
    else:
        Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # --- 11. Weight update: X = (1-lr*wd)*X - scaled_lr * P @ Q^T (matching DDP) ---
    fan_out = X[0].size(0)  # global shape from DTensor
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr
    X_local = to_local(X)  # foreach_baddbmm_ needs all plain or all DTensor
    foreach_baddbmm_(
        X_local,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=transpose
    )

    # --- 12. Update Q buffers and adaptive rank ---
    # Batched sig computation: column norms of R for all params at once
    # R_batch is (batch, n_outer, r_cap) — compute norms over dim -2
    sig_sq_batch = local_column_sum_sq(R_batch).squeeze(-2)  # (batch, r_cap)
    if param_config.outer_shard_mesh_dim is not None:
        sig_sq_batch = funcol.all_reduce(sig_sq_batch, reduceOp="sum", group=fsdp_group)
    sig_batch = sig_sq_batch.sqrt()  # (batch, r_cap)

    for i in range(num_real):
        state = ada_states[i]
        r_i = state["r"]
        r_cap_i = state["r_cap"]
        Qbuf = Q[i]
        q_dim = n_outer if transpose else m

        Qbuf[:q_dim, :r_i].copy_(Q_batch[i, :q_dim, :r_i])

        state["ada_step"] += 1

        sig = sig_batch[i, :r_i] + 1e-12
        probs = sig / (sig.sum() + 1e-12)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        erank = torch.exp(entropy).item()
        if math.isnan(erank):
            erank = float(r_i)

        beta = ada_config["erank_ema_beta"]
        if state["erank_ema"] is None:
            state["erank_ema"] = erank
        else:
            state["erank_ema"] = beta * state["erank_ema"] + (1 - beta) * erank

        # # Quality control (disabled)
        # aerr = None
        # if ada_config.get("use_quality_control", False):
        #     if transpose:
        #         approx_i = R_batch[i] @ P_batch[i].T
        #     else:
        #         approx_i = P_batch[i] @ R_batch[i].T
        #     aerr_num = (B_list[i] - approx_i).float().norm()
        #     aerr_den = B_list[i].float().norm() + 1e-12
        #     if fsdp_group is not None:
        #         n2 = aerr_num.pow(2)
        #         d2 = (aerr_den - 1e-12).pow(2)
        #         dist.all_reduce(n2, group=fsdp_group)
        #         dist.all_reduce(d2, group=fsdp_group)
        #         aerr = (n2.sqrt() / (d2.sqrt() + 1e-12)).item()
        #     else:
        #         aerr = (aerr_num / aerr_den).item()
        #     a_beta = ada_config["aerr_ema_beta"]
        #     if state["aerr_ema"] is None:
        #         state["aerr_ema"] = aerr
        #     else:
        #         state["aerr_ema"] = a_beta * state["aerr_ema"] + (1.0 - a_beta) * aerr

        if state["ada_step"] % ada_config.get("adapt_step", 1) == 0:
            desired = int(round(ada_config["rank_scale"] * state["erank_ema"]))
            desired = min(max(ada_config["rank_min"], desired), r_cap_i)
            # if aerr is not None and state["aerr_ema"] is not None:
            #     a_target = ada_config["aerr_target"]
            #     if aerr > max(a_target, state["aerr_ema"] * (1 + ada_config["aerr_up_margin"])):
            #         desired = max(desired, r_i + ada_config["rank_step_up"])
            #     if aerr < min(0.5 * a_target, state["aerr_ema"] * (1 - ada_config["aerr_down_margin"])):
            #         desired = min(desired, r_i - ada_config["rank_step_down"])
            desired = _quantize_int(min(max(ada_config["rank_min"], desired), r_cap_i),
                                    ada_config["rank_quantize"])
            desired = min(max(ada_config["rank_min"], desired), r_cap_i)
            if desired > r_i:
                r_next = min(desired, r_i + ada_config["rank_step_up"])
            elif desired < r_i:
                r_next = max(desired, r_i - ada_config["rank_step_down"])
            else:
                r_next = r_i
            r_next = int(min(max(ada_config["rank_min"], r_next), r_cap_i))
            if r_next > r_i:
                Qbuf[:, r_i:r_next].copy_(F.normalize(
                    torch.randn(Qbuf.shape[0], r_next - r_i,
                                device=Qbuf.device, dtype=orig_dtype), dim=0))
            state["r"] = r_next


def dion_update_fsdp_tp(
    X: List[DTensor],  # Model weights (modified in place)
    G: List[DTensor],  # Gradient
    M: List[DTensor],  # Momentum buffer (modified in place)
    Q: List[DTensor],  # Q matrix for power iteration (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    mu: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: float,
    param_config: _DionParamConfig,  # shared for all params in batch
    replicate_mesh: Optional[DeviceMesh] = None,
    replicate_mesh_grad_sync: bool = True,
    oversample: float = 1.25,
    r_list: Optional[List[int]] = None,
    ada_states: Optional[List[Dict]] = None,
    ada_config: Optional[Dict] = None,
) -> Generator[None, None, None]:
    """
    Dion optimizer algorithm. Async batched implementation for combined FSDP2 + TP.
    This function supports sharding over both outer and inner shard meshes.

    Batch size should equal the inner shard mesh size. The full matrix will not be
    unsharded for orthogonalization. Each device along the inner shard mesh dimension
    will compute low-rank QR and Cholesky decompositions for one matrix in the batch.
    """
    # Special case for when compressed_all_reduce is False
    # Here we all-reduce the gradient G instead of compressed P and R matrices
    # This is more efficient when rank_fraction == 1
    if (
        replicate_mesh_grad_sync
        and not param_config.compressed_all_reduce
        and replicate_mesh is not None
    ):
        G = all_reduce_replicate_mesh(G, replicate_mesh)
        yield

    # Add new gradient to momentum
    torch._foreach_add_(M, G)

    # Unshard Q along the inner sharding dimension
    if param_config.Q_inner_unsharded_placements is not None:
        # Sharded Q has shape (n/outer, r/inner)
        # Unsharded Q has shape (n/outer, r)
        Q_unshard = [
            q.redistribute(
                placements=param_config.Q_inner_unsharded_placements,
                async_op=True,
            )
            for q in Q
        ]
        yield
    else:
        # Q is not sharded along inner sharding dimension
        Q_unshard = Q

    # Compute low-rank approximation of M = P @ Q^T
    # M_batch shape is (batch_size, m/inner, n/outer)
    # P_batch shape is (batch_size, m/inner, r)
    # Q_batch shape is (batch_size, n/outer, r)
    M_batch, Q_batch = tensor_list_to_batch(M, Q_unshard, param_config.is_transposed)
    Q_unshard = None  # No longer needed, free memory
    P_batch: DTensor = M_batch @ Q_batch

    # All reduce P to get the sharded matrix multiplication result
    P_batch = P_batch.redistribute(
        placements=[Replicate() if p.is_partial() else p for p in P_batch.placements],
        async_op=True,
    )
    yield

    # If compressed_all_reduce is True, also average over replicate mesh
    compressed_all_reduce = (
        replicate_mesh_grad_sync and param_config.compressed_all_reduce
    )
    if compressed_all_reduce and replicate_mesh is not None:
        P_batch = all_reduce_replicate_mesh(P_batch, replicate_mesh)
        yield

    # Orthogonalize P_batch
    P_batch = distributed_orthogonalize(
        P_batch, oversample, shard_mesh_dim=param_config.inner_shard_mesh_dim
    )

    # M_batch shape is (batch_size, m/inner, n/outer)
    # P_batch shape is (batch_size, m/inner, r)
    # R_batch shape is (batch_size, n/outer, r)
    R_batch: DTensor = M_batch.mT @ P_batch

    # All reduce R to get the sharded matrix multiplication result
    R_batch = R_batch.redistribute(
        placements=[Replicate() if p.is_partial() else p for p in R_batch.placements],
        async_op=True,
    )
    yield

    # If compressed_all_reduce is True, also average over replicate mesh
    if compressed_all_reduce and replicate_mesh is not None:
        R_batch = all_reduce_replicate_mesh(R_batch, replicate_mesh)
        yield

    # NaN check
    P_batch, R_batch = fix_all_zero_or_nan(P_batch, R_batch, Q_batch, M_batch)

    # Error feedback update
    # M = M - (1 - mu) * (P @ R.T)
    foreach_baddbmm_(
        M, P_batch, R_batch, alpha=-(1 - mu), transpose=param_config.is_transposed
    )

    # Column normalize R to get new Q
    if param_config.outer_shard_mesh_dim is not None:
        assert isinstance(R_batch, DTensor)
        assert R_batch.placements[param_config.outer_shard_mesh_dim].is_shard()

        # Compute per-shard squared sum and sum across shards
        R_sum_sq = local_column_sum_sq(R_batch)
        R_sum_sq = funcol.all_reduce(
            R_sum_sq,
            reduceOp="sum",
            group=(R_batch.device_mesh, param_config.outer_shard_mesh_dim),
        )
        yield

        Q_batch = column_normalize(
            R_batch,
            full_column_sum_sq=R_sum_sq,
            epsilon=epsilon,
        )

    else:
        # The sum dimension is not sharded, so we can normalize directly
        Q_batch = column_normalize(R_batch, epsilon=epsilon)

    # Compute update scale factor
    fan_out = X[0].size(0)
    fan_in = X[0].size(1)
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr

    # Apply weight decay and weight update
    # X = (1 - lr * weight_decay) * X - scaled_lr * P @ Q.T
    foreach_baddbmm_(
        X,
        P_batch,
        Q_batch,
        alpha=-scaled_lr,
        beta=1 - lr * weight_decay,
        transpose=param_config.is_transposed,
    )

    # Re-shard and update Q in place
    update_Q_matrix_(Q, Q_batch, param_config.Q_sharded_placements)


def _adaptive_rank_update(
    R_batch: Tensor,
    ada_states: List[Dict],
    ada_config: Dict,
    Q: List[Tensor],
    param_config: "_DionParamConfig",
):
    """
    Compute erank from R column norms and update active rank in ada_states.
    Called inline after the Dion update step.
    """
    erank_ema_beta = ada_config["erank_ema_beta"]
    rank_scale = ada_config["rank_scale"]
    rank_min = ada_config["rank_min"]
    rank_quantize = ada_config["rank_quantize"]
    step_up = ada_config["rank_step_up"]
    step_down = ada_config["rank_step_down"]

    R_local = to_local(R_batch)
    num_real = len(ada_states)  # batch may be padded

    for i in range(num_real):
        state = ada_states[i]
        r = state["r"]
        r_cap = state["r_cap"]

        # R column norms as singular value proxies
        sig = torch.linalg.vector_norm(R_local[i].float(), dim=0)  # (r,)
        sig = sig + 1e-12
        probs = sig / (sig.sum() + 1e-12)
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        erank = torch.exp(entropy).item()

        # Update EMA
        if state["erank_ema"] is None:
            state["erank_ema"] = erank
        else:
            state["erank_ema"] = (
                erank_ema_beta * state["erank_ema"]
                + (1.0 - erank_ema_beta) * erank
            )

        # Desired rank
        desired = int(round(rank_scale * state["erank_ema"]))
        desired = min(max(rank_min, desired), r_cap)
        desired = _quantize_int(desired, rank_quantize)
        desired = min(max(rank_min, desired), r_cap)

        # Rate-limit
        if desired > r:
            r_next = min(desired, r + step_up)
        elif desired < r:
            r_next = max(desired, r - step_down)
        else:
            r_next = r
        r_next = int(min(max(rank_min, r_next), r_cap))

        # Fill new Q columns if expanding
        if r_next > r:
            Q_local = to_local(Q[i])
            new_cols = r_next - r
            Q_local[:, r:r_next].copy_(
                F.normalize(
                    torch.randn(
                        Q_local.shape[0], new_cols,
                        device=Q_local.device, dtype=Q_local.dtype,
                    ),
                    dim=0,
                )
            )

        state["r"] = r_next


def all_reduce_replicate_mesh(
    G: Union[Tensor, List[Tensor]],
    replicate_mesh: Optional[DeviceMesh] = None,
    return_dtensor: bool = True,
    reduce_op: str = "avg",
) -> Union[Tensor, List[Tensor]]:
    """
    All-reduce a tensor or list of tensors across replicated data-parallel ranks.
    """
    if replicate_mesh is None:
        # No data parallelism, return original tensors unmodified
        return G

    if isinstance(G, Tensor):
        # Single tensor
        result_local = funcol.all_reduce(
            to_local(G),
            reduceOp=reduce_op,
            group=replicate_mesh,
        )
    else:
        # List of tensors, use coalesced all-reduce
        result_local = funcol.all_reduce_coalesced(
            to_local(G),
            reduceOp=reduce_op,
            group=replicate_mesh,
        )

    if return_dtensor:
        ref = G if isinstance(G, Tensor) else G[0]
        return dtensor_from_local(result_local, ref=ref)
    else:
        return result_local


def tensor_list_to_batch(
    M: List[Tensor],
    Q: List[Tensor],
    is_transposed: bool,
    r_list: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Convert a list of tensors M and Q into 3D batched tensors.
    Outputs M_batch and Q_batch will have dtype matching M.

    If r_list is provided, Q is sliced to first r columns at local tensor level.
    All r values in r_list must be equal (required for torch.stack).
    """
    # Transpose the momentum matrices if needed
    if is_transposed:
        M_batch = torch.stack([m.mT for m in M])
    else:
        M_batch = torch.stack(M)

    if r_list is not None and not isinstance(Q[0], DTensor):
        # DDP path: slice Q to active columns at LOCAL tensor level.
        Q_sliced = [q[:, :r] for q, r in zip(Q, r_list)]
        Q_batch = torch.stack(Q_sliced).to(M_batch.dtype)
    else:
        # Match dtype of Q and M
        Q_batch = torch.stack(Q).to(M_batch.dtype)

    return M_batch, Q_batch


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
) -> Tensor:
    """
    Generate a random sketch matrix S for low-rank approximation.
    P is the input tensor with shape (batch_size, m, r).
    The sketch matrix S will have shape (batch_size, k, m),
    where k = round(oversample * r) to the next multiple of 128.
    """
    assert P.ndim >= 3, "P must have batch dimension"

    batch_size = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if isinstance(P, DTensor):
        S_placements = list(P.placements)
        if shard_mesh_dim is not None:
            # Shard along tensor dimension -1 (size m dimension)
            S_placements[shard_mesh_dim] = Shard(P.ndim - 1)

        S = dtensor_randn(
            (*batch_size, k, m),
            device_mesh=P.device_mesh,
            dtype=P.dtype,
            placements=S_placements,
        )
        S = S * std

    else:
        # Regular tensor case
        if shard_mesh_dim is not None:
            raise TypeError("Must use DTensor parameters for sharded random sketch.")

        S = torch.empty((*batch_size, k, m), device=P.device, dtype=P.dtype).normal_(
            std=std
        )

    return S


# Graph break in torch.compile due to DTensor RNG
@torch.compile()
def orthogonalize(P: Tensor, oversample: float = 1.25) -> Tensor:
    """
    Orthogonalize a batch of matrices.
    The input cannot be sharded along the matrix dimensions.
    If input is DTensor, the output will also be DTensor.
    """
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    P_local = to_local(P)

    if isinstance(P, DTensor):
        # Matrix dimensions (-2, -1) cannot be sharded
        assert not any(p.is_shard(P.ndim - 2) for p in P.placements)
        assert not any(p.is_shard(P.ndim - 1) for p in P.placements)

    # Standard QR is faster if matrix is square or wide
    if P.size(-2) <= P.size(-1):
        P_local, _ = torch.linalg.qr(P_local.to(dtype=torch.float32))

    # Randomized Cholesky QR
    else:
        # Must generate random sketch as DTensor for synchronized RNG
        S = generate_random_sketch_matrix(P, oversample)
        S_local = to_local(S)

        # Orthogonalize P using random sketch QR
        SP = S_local @ P_local
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        P_local = torch.linalg.solve_triangular(
            R, P_local.to(dtype=torch.float32), upper=True, left=False
        )

        # Apply Cholesky QR to better orthogonalize
        PP = P_local.mT @ P_local  # always do float32 matrix multiply
        R, _ = torch.linalg.cholesky_ex(PP, upper=True)
        P_local = torch.linalg.solve_triangular(R, P_local, upper=True, left=False)

    return dtensor_from_local(
        P_local.to(original_dtype).contiguous(),
        ref=P,
    )


# Graph break in torch.compile due to DTensor RNG
@torch.compile()
def distributed_orthogonalize(
    P: DTensor, oversample: float = 1.25, shard_mesh_dim: Optional[int] = None
) -> DTensor:
    """
    Orthogonalize a batch of sharded matrices.
    The input cannot be sharded along the last dimension.
    """
    assert isinstance(P, DTensor)
    assert not any(p.is_partial() for p in P.placements)
    assert not any(p.is_shard(P.ndim - 1) for p in P.placements)
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    original_placements = P.placements

    # Batch-sharded placement shards dimension 0 = batch
    # Each device gets one full matrix in the batch
    fully_replicated_placements = [Replicate() for _ in P.placements]
    batch_sharded_placements = fully_replicated_placements.copy()
    if shard_mesh_dim is not None:
        batch_sharded_placements[shard_mesh_dim] = Shard(0)

    # Standard QR is faster if matrix is square or wide
    if P.size(-2) <= P.size(-1):
        # Get one full matrix in the batch
        P_single = P.redistribute(
            placements=batch_sharded_placements,
        )  # this should do all-to-all

        # Compute Q matrix of QR decomposition
        Q_local, _ = torch.linalg.qr(
            P_single.to_local().to(dtype=torch.float32), mode="reduced"
        )

        # Convert back to DTensor and redistribute to original sharding
        P = dtensor_from_local(
            Q_local.to(original_dtype).contiguous(),
            ref=P_single,
        ).redistribute(
            placements=original_placements,
        )  # this should do all-to-all

    # Randomized Cholesky QR
    else:
        # Compute the random sketch matrix
        S = generate_random_sketch_matrix(P, oversample, shard_mesh_dim=shard_mesh_dim)
        SP: DTensor = S @ P

        # Get one full matrix in the batch
        SP_single = SP.redistribute(
            placements=batch_sharded_placements,
        )  # this should do reduce-scatter

        # Compute R matrix using QR decomposition
        _, R_local = torch.linalg.qr(
            SP_single.to_local().to(dtype=torch.float32), mode="r"
        )

        # Convert back to DTensor and get entire batch of R matrices
        R = dtensor_from_local(R_local, ref=SP_single).redistribute(
            placements=fully_replicated_placements,
        )  # this should do all-gather

        # Solve for orthogonalized batch of P matrix shards
        P_local = torch.linalg.solve_triangular(
            R.to_local(),  # already float32
            P.to_local().to(dtype=torch.float32),
            upper=True,
            left=False,
        )
        P = dtensor_from_local(P_local, ref=P)

        # Apply Cholesky QR to better orthogonalize P
        PP: DTensor = P.mT @ P

        # Get one full matrix in the batch
        PP_single = PP.redistribute(
            placements=batch_sharded_placements,
        )  # this should do reduce-scatter

        # Compute R matrix using Cholesky decomposition
        R_local, _ = torch.linalg.cholesky_ex(PP_single.to_local(), upper=True)

        # Convert back to DTensor and get entire batch of R matrices
        R = dtensor_from_local(R_local, ref=PP_single).redistribute(
            placements=fully_replicated_placements,
        )  # this should do all-gather

        # Solve for orthogonalized batch of P matrix shards
        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local(),
            upper=True,
            left=False,
        )

        # Convert back to DTensor and restore original dtype
        P = dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)

    assert P.dtype == original_dtype, "Output dtype mismatch"
    assert P.placements == original_placements, "Output placements mismatch"
    return P



@torch.compile(fullgraph=False)
def fix_all_zero_or_nan(
    P: Tensor,  # Output of power iteration
    R: Tensor,  # Output of power iteration
    Q_init: Tensor,  # Initial Q matrix
    B: Tensor,  # Buffer to check if all zeros
) -> Tuple[Tensor, Tensor]:
    """
    If input is all zero, P and R will be nan or all zero.
    We want to return the conditional expressions:

        if is_all_zero:
            P = torch.zeros_like(P)
            R = Q_init
        else:
            P = P
            R = R

    Here this is implemented without data-dependent control flow.
    To avoid additional communication, we handle sharded tensors independently.
    """
    B_local = to_local(B)
    is_all_zero = (B_local == 0).all(dim=(-2, -1), keepdim=True)
    not_all_zero = ~is_all_zero
    P_local = to_local(P).nan_to_num() * not_all_zero
    R_local = to_local(R).nan_to_num() * not_all_zero + to_local(Q_init) * is_all_zero
    P = dtensor_from_local(P_local, ref=P)
    R = dtensor_from_local(R_local, ref=R)
    return P, R


@torch.compile(fullgraph=False)
def local_column_sum_sq(X: Tensor) -> Tensor:
    """
    Compute the per-column sum of squares of a tensor, or local shard of a DTensor.
    If the input has shape (m, n), the output will have shape (1, n).
    Regardless of input, the output will be a local tensor with float32 dtype.
    """
    X = to_local(X).to(dtype=torch.float32)
    # Sum over all rows to get one value per column
    X = X.square().sum(dim=-2, keepdim=True)
    return X


@torch.compile(fullgraph=False)
def column_normalize(
    X: Tensor,
    full_column_sum_sq: Optional[Tensor] = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """
    Normalize the columns of a tensor or local shard of a DTensor.
    If the input is a row-sharded DTensor, full_column_sum_sq must be provided.
    The computation is performed internally in float32 for numerical stability.
    """
    if isinstance(X, DTensor) and full_column_sum_sq is None:
        # To compute the per-column norm, we need to sum across all rows
        # If X is row sharded, we require full_column_sum_sq to be provided
        if any(p.is_shard(X.ndim - 2) for p in X.placements):
            raise RuntimeError("Cannot normalize row-sharded DTensor.")

    original_dtype = X.dtype
    X_local = to_local(X).to(dtype=torch.float32)

    # Compute per-column sum of squares if not provided
    if full_column_sum_sq is None:
        full_column_sum_sq = X_local.square().sum(dim=-2, keepdim=True)
    else:
        full_column_sum_sq = to_local(full_column_sum_sq).to(dtype=torch.float32)

    # Normalize each column
    full_column_norm = torch.sqrt(full_column_sum_sq)
    X_local = X_local / (full_column_norm + epsilon)

    X = dtensor_from_local(X_local.to(original_dtype), ref=X)
    return X


@torch.compile(fullgraph=False)
def foreach_baddbmm_(
    X: List[Tensor],  # List of 2D matrices (modified in place)
    A: Tensor,  # 3D batch of matrices
    B: Tensor,  # 3D batch of matrices
    alpha: float = 1.0,
    beta: float = 1.0,
    transpose: bool = False,
):
    """
    Perform batch matrix multiplication and in-place addition.
    This is basically a foreach version of torch.baddbmm().

    If transpose is False, we compute X[i] = beta * X[i] + alpha * (A @ B.mT)[i].
    If transpose is True, we compute X[i] = beta * X[i] + alpha * (B @ A.mT)[i].
    """
    assert A.size(0) == B.size(0), "A and B must have the same batch size"
    assert len(X) == A.size(0), "len(X) must equal the batch dimension of A and B"

    if not transpose:
        update = A @ B.mT
    else:
        update = B @ A.mT

    # Convert DTensor to local tensor for foreach operations
    if isinstance(update, DTensor):
        if any(p.is_partial() for p in update.placements):
            raise NotImplementedError(
                "This function does not support DTensor matrix multiplication resulting in Partial() placements."
            )
        update = update.to_local()
        X = to_local(X)

    update = update.unbind(dim=0)  # Split batch into list of tensors
    update = torch._foreach_mul(update, alpha)  # Scale update by alpha
    torch._foreach_mul_(X, beta)  # Scale existing X by beta
    torch._foreach_add_(X, update)


@torch.compile(fullgraph=False)
def update_Q_matrix_(
    Q: List[Tensor],  # Q matrix for power iteration (modified in place)
    Q_batch: Tensor,  # New Q matrix from orthogonalization
    Q_sharded_placements: Optional[Tuple[Placement]] = None,
    r_list: Optional[List[int]] = None,
):
    """
    Update the list of Q matrices in place with the new 3D stacked Q_batch.
    If r_list is provided, writes to first r columns only (adaptive rank).
    """
    if Q_sharded_placements is not None:
        # Increment all Shard() dimensions by 1 because of the batch dimension
        assert isinstance(Q_batch, DTensor)
        Q_batch_sharded_placements = list(Q_sharded_placements)

        for i in range(len(Q_batch_sharded_placements)):
            if Q_batch_sharded_placements[i].is_shard():
                Q_batch_sharded_placements[i] = Shard(
                    Q_batch_sharded_placements[i].dim + 1
                )

        # Redistribute Q_batch to sharded placements
        Q_batch = Q_batch.redistribute(
            placements=Q_batch_sharded_placements,
        )

    # Match dtype and convert to local tensor
    Q_init_dtype = Q[0].dtype
    Q_batch = to_local(Q_batch).to(Q_init_dtype)

    if r_list is not None and not isinstance(Q[0], DTensor):
        # DDP path: write back to first r columns only
        for q, q_new, r in zip(Q, Q_batch.unbind(dim=0), r_list):
            q[:, :r].copy_(q_new)
    else:
        torch._foreach_copy_(to_local(Q), Q_batch.unbind(dim=0))


def adamw_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    AdamW optimizer algorithm with gradient all-reduce.
    """
    if replicate_mesh is not None:
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)


def lion_update_allreduce_grad(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    replicate_mesh: Optional[DeviceMesh] = None,
) -> Generator[None, None, None]:
    """
    Lion optimizer algorithm with gradient all-reduce.
    """
    if replicate_mesh is not None:
        G = all_reduce_replicate_mesh(G, replicate_mesh, return_dtensor=False)
        yield
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay)
