"""
AdaDion: Adaptive Rank Control for Distributed Orthonormalized Updates.

Aligned with the working DionSim reference implementation. Supports FSDP2/DTensor.

Algorithm overview:
  - Error feedback buffer: B = M + grad
  - Power iteration with transpose handling (m<=n vs m>n)
  - Distributed Cholesky QR orthonormalization
  - Error feedback: M = B - (1 - mu) * approx
  - Weight update: p -= lr * sqrt(dmin/dmax) * update
  - Adaptive rank via effective rank EMA + quality control
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _col_normalize(V: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize each column of V to unit norm (local — assumes V is full)."""
    norms = torch.linalg.vector_norm(V.float(), dim=0).clamp(min=eps)
    return V / norms.to(V.dtype)


def _col_normalize_distributed(
    V: Tensor, process_group, eps: float = 1e-12
) -> Tensor:
    """Normalize each column of V to global unit norm when V is sharded along dim=0.

    Each rank holds a shard of V. Column norms must be computed globally
    via all-reduce of squared norms before dividing.
    """
    col_norm_sq = torch.linalg.vector_norm(V.float(), dim=0).pow(2)
    if process_group is not None:
        dist.all_reduce(col_norm_sq, group=process_group)
    elif dist.is_initialized():
        dist.all_reduce(col_norm_sq)
    norms = col_norm_sq.sqrt().clamp(min=eps)
    return V / norms.to(V.dtype)


def _quantize_int(x: float, q: int) -> int:
    """Round x to the nearest multiple of q."""
    if q <= 1:
        return int(round(x))
    return int(round(x / q) * q)


def _fix_nan(U: Tensor, W: Tensor, V_prev: Tensor) -> tuple[Tensor, Tensor, bool]:
    """Detect NaN/Inf in power iteration outputs and recover."""
    has_bad = (
        torch.isnan(U).any() or torch.isinf(U).any()
        or torch.isnan(W).any() or torch.isinf(W).any()
    )
    if has_bad:
        U = torch.zeros_like(U)
        W = V_prev.clone()
        return U, W, True
    return U, W, False


# ---------------------------------------------------------------------------
# AdaDion optimizer
# ---------------------------------------------------------------------------

class AdaDion(Optimizer):
    """
    AdaDion optimizer: Dion with adaptive rank selection.

    For matrix parameters (2D), implements the Dion algorithm with
    adaptive rank selection matching the working DionSim reference.

    For scalar parameters (1D, embeddings, norms), routes to AdamW
    when ``algorithm="adamw"`` is set in the param group.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        # --- Adaptive rank ---
        adaptive_rank: bool = True,
        erank_ema_beta: float = 0.9,
        rho: float = 1.5,
        rank_min: int = 16,
        rank_max: int = 256,
        rank_quantize: int = 8,
        rank_warmup_steps: int = 10,
        warmup_rank: int = 128,
        rank_step_up: int = 16,
        rank_step_down: int = 8,
        # --- Quality control ---
        use_quality_control: bool = True,
        aerr_ema_beta: float = 0.95,
        aerr_target: float = 0.08,
        aerr_up_margin: float = 0.15,
        aerr_down_margin: float = 0.15,
        # --- Other ---
        qbuf_max_cols: int = 256,
        weight_decay: float = 0.0,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        # --- Distributed ---
        device_mesh: Optional[DeviceMesh] = None,
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            init_rank=init_rank,
            adaptive_rank=adaptive_rank,
            erank_ema_beta=erank_ema_beta,
            rho=rho,
            rank_min=rank_min,
            rank_max=rank_max,
            rank_quantize=rank_quantize,
            rank_warmup_steps=rank_warmup_steps,
            warmup_rank=warmup_rank,
            rank_step_up=rank_step_up,
            rank_step_down=rank_step_down,
            use_quality_control=use_quality_control,
            aerr_ema_beta=aerr_ema_beta,
            aerr_target=aerr_target,
            aerr_up_margin=aerr_up_margin,
            aerr_down_margin=aerr_down_margin,
            qbuf_max_cols=qbuf_max_cols,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

        self._device_mesh = device_mesh
        self._process_group = None
        if device_mesh is not None:
            self._process_group = device_mesh.get_group()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_matrix_param(self, p: Tensor) -> bool:
        return p.ndim == 2 and p.shape[0] >= 2 and p.shape[1] >= 2

    def _get_local_tensor(self, p: Tensor) -> Tensor:
        if isinstance(p, DTensor):
            return p._local_tensor
        return p

    def _get_full_shape(self, p: Tensor) -> tuple[int, ...]:
        if isinstance(p, DTensor):
            return tuple(p.shape)  # DTensor.shape returns the global (unsharded) shape
        return tuple(p.shape)

    def _get_world_size(self) -> int:
        if self._process_group is not None:
            return dist.get_world_size(self._process_group)
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _all_reduce(self, t: Tensor) -> None:
        if self._process_group is not None:
            dist.all_reduce(t, group=self._process_group)
        elif self._device_mesh is None and dist.is_initialized():
            dist.all_reduce(t)

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                self._step_adamw(group)
            else:
                self._step_adadion(group)

        return loss

    # ------------------------------------------------------------------
    # AdamW update (for scalar parameters)
    # ------------------------------------------------------------------

    def _step_adamw(self, group: dict) -> None:
        betas = group.get("betas", group.get("adamw_betas", (0.9, 0.95)))
        eps = group.get("eps", group.get("adamw_eps", 1e-8))
        lr = group.get("lr", self.defaults["lr"])
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            if wd != 0:
                p.data.mul_(1 - lr * wd)

            exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
            exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

            bc1 = 1 - betas[0] ** step
            bc2 = 1 - betas[1] ** step
            step_size = lr / bc1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

    # ------------------------------------------------------------------
    # AdaDion update (matching working DionSim reference)
    # ------------------------------------------------------------------

    def _step_adadion(self, group: dict) -> None:
        lr = group["lr"]
        mu = group["mu"]
        init_rank = group["init_rank"]
        adaptive = group["adaptive_rank"]
        erank_ema_beta = group["erank_ema_beta"]
        rho = group["rho"]
        rank_min = group["rank_min"]
        rank_max = group["rank_max"]
        rank_quantize = group["rank_quantize"]
        rank_warmup_steps = group["rank_warmup_steps"]
        warmup_rank = group["warmup_rank"]
        step_up = group["rank_step_up"]
        step_down = group["rank_step_down"]
        use_quality = group["use_quality_control"]
        aerr_ema_beta = group["aerr_ema_beta"]
        aerr_target = group["aerr_target"]
        aerr_up_margin = group["aerr_up_margin"]
        aerr_down_margin = group["aerr_down_margin"]
        qbuf_max_cols = group["qbuf_max_cols"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = self._get_local_tensor(p.grad)
            p_local = self._get_local_tensor(p)
            full_shape = self._get_full_shape(p)

            if not self._is_matrix_param(p):
                if weight_decay != 0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(grad, alpha=-lr)
                continue

            state = self.state[p]
            m_full, n = full_shape[0], full_shape[1]
            m_local = p_local.shape[0]
            dmin = min(m_full, n)
            dmax = max(m_full, n)
            shape_scale = math.sqrt(float(dmin) / float(dmax))
            r_cap = min(dmin, rank_max)

            if r_cap <= 0:
                continue

            # --- State initialization ---
            if len(state) == 0:
                transpose = (m_full > n)
                buf_cols = min(r_cap, qbuf_max_cols if adaptive else init_rank)
                if adaptive:
                    init_r = min(init_rank, r_cap, buf_cols)
                else:
                    init_r = min(init_rank, r_cap, buf_cols)

                # FSDP2 validation: sharded dimension must be evenly divisible
                world_size = self._get_world_size()
                if world_size > 1 and isinstance(p, DTensor):
                    if transpose and m_full % world_size != 0:
                        raise ValueError(
                            f"AdaDion FSDP2 error: transpose case requires "
                            f"m_full ({m_full}) to be divisible by "
                            f"world_size ({world_size}). "
                            f"Parameter shape: ({m_full}, {n})"
                        )
                    if not transpose and m_full % world_size != 0:
                        raise ValueError(
                            f"AdaDion FSDP2 error: non-transpose case requires "
                            f"m_full ({m_full}) to be divisible by "
                            f"world_size ({world_size}). "
                            f"Parameter shape: ({m_full}, {n})"
                        )
                    assert m_local == m_full // world_size, (
                        f"Unexpected local shard size: m_local={m_local}, "
                        f"expected m_full/world_size={m_full // world_size}"
                    )

                # Deterministic seed for FSDP consistency
                gen = torch.Generator(device=p_local.device)
                gen.manual_seed(0x4469_6F6E + n * 1000 + buf_cols)

                if transpose:
                    # m > n: Qbuf is (m_local, buf_cols), sharded per rank
                    Qbuf = F.normalize(
                        torch.randn(m_local, buf_cols, device=p_local.device,
                                    dtype=p_local.dtype, generator=gen),
                        dim=0,
                    )
                else:
                    # m <= n: Qbuf is (n, buf_cols), replicated across ranks
                    Qbuf = F.normalize(
                        torch.randn(n, buf_cols, device=p_local.device,
                                    dtype=p_local.dtype, generator=gen),
                        dim=0,
                    )

                rmin_eff = min(rank_min, r_cap, buf_cols)
                r_init = int(max(1, min(max(rmin_eff, init_r), r_cap, buf_cols)))

                state["M"] = torch.zeros_like(p_local)
                state["Qbuf"] = Qbuf
                state["r"] = r_init
                state["transpose"] = transpose
                state["erank_ema"] = None
                state["aerr_ema"] = None
                state["step"] = 0

            state["step"] += 1
            M = state["M"]
            Qbuf = state["Qbuf"]
            r = state["r"]
            transpose = state["transpose"]
            orig_dtype = M.dtype

            # Effective rank bounds
            rmin_eff = int(max(1, min(rank_min, r_cap, Qbuf.shape[1])))

            if not adaptive:
                r = int(min(max(rmin_eff, init_rank), Qbuf.shape[1], r_cap))
            else:
                r = int(min(max(rmin_eff, r), Qbuf.shape[1], r_cap))

            # --- B = M + grad (don't mutate M yet) ---
            B = M + grad

            # --- Power iteration with transpose handling ---
            if not transpose:
                # m <= n branch
                Q_prev = Qbuf[:n, :r]        # (n, r) replicated
                P_hat = B @ Q_prev            # (m_local, r) sharded

                # Distributed Cholesky QR on P_hat
                P_f32 = P_hat.float()
                gram = P_f32.T @ P_f32        # (r, r) partial sum
                self._all_reduce(gram)         # ALL-REDUCE #1
                gram.diagonal().add_(1e-6)
                try:
                    L = torch.linalg.cholesky(gram)
                    P_orth_f32 = torch.linalg.solve_triangular(
                        L, P_f32.T, upper=False
                    ).T
                except torch.linalg.LinAlgError:
                    P_orth_f32, _ = torch.linalg.qr(P_f32)
                P_orth = P_orth_f32.to(orig_dtype)

                # R = B^T @ P_orth: (n, r) partial sum
                R = (B.float().T @ P_orth_f32).to(orig_dtype)
                self._all_reduce(R)            # ALL-REDUCE #2

                approx = P_orth @ R.T          # (m_local, n)
                Q_new = _col_normalize(R)      # (n, r) replicated
                update = P_orth @ Q_new.T      # (m_local, n)
                Qbuf[:n, :r].copy_(Q_new)

            else:
                # m > n branch (transpose)
                Q_prev = Qbuf[:m_local, :r]   # (m_local, r) sharded

                # P_hat = B^T @ Q_prev: (n, r) partial sum
                P_hat_f32 = (B.float().T @ Q_prev.float())
                self._all_reduce(P_hat_f32)    # ALL-REDUCE #1

                # QR on small replicated matrix
                P_orth_f32, _ = torch.linalg.qr(P_hat_f32, mode="reduced")
                P_orth = P_orth_f32.to(orig_dtype)

                R = B @ P_orth                 # (m_local, r) sharded
                approx = R @ P_orth.T          # (m_local, n) sharded
                # R is sharded along dim=0: need global column norms
                Q_new = _col_normalize_distributed(R, self._process_group)
                update = Q_new @ P_orth.T      # (m_local, n) sharded
                Qbuf[:m_local, :r].copy_(Q_new)

            # --- NaN/Inf recovery ---
            # Check update for NaN (covers both branches)
            if torch.isnan(update).any() or torch.isinf(update).any():
                update = torch.zeros_like(update)
                # Don't update Qbuf (already copied, but that's ok — next step re-computes)

            # --- Error feedback: M = B - (1 - mu) * approx ---
            M.copy_(B - (1.0 - mu) * approx)

            # --- Weight update ---
            if weight_decay != 0:
                p_local.mul_(1 - lr * weight_decay)
            p_local.add_(update, alpha=-(lr * shape_scale))

            # --- Adaptive rank selection ---
            if adaptive:
                if state["step"] <= rank_warmup_steps:
                    # During warmup, ramp to warmup_rank
                    desired = int(min(warmup_rank, r_cap, Qbuf.shape[1]))
                    desired = max(rmin_eff, desired)
                    if desired > r:
                        qbuf_dim = Qbuf.shape[0]
                        new_cols = desired - r
                        Qbuf[:, r:desired].copy_(F.normalize(
                            torch.randn(qbuf_dim, new_cols,
                                        device=p_local.device, dtype=orig_dtype),
                            dim=0,
                        ))
                    state["r"] = desired
                else:
                    # Effective rank from R column norms
                    sig = torch.linalg.vector_norm(R.float(), dim=0)
                    if transpose:
                        # R is sharded: need all-reduce of squared norms
                        sig_sq = sig.pow(2)
                        self._all_reduce(sig_sq)
                        sig = sig_sq.sqrt()
                    sig = sig + 1e-12
                    probs = sig / (sig.sum() + 1e-12)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum()
                    erank = torch.exp(entropy).item()

                    # Standard EMA: beta * old + (1-beta) * new
                    old_ema = state["erank_ema"]
                    if old_ema is None:
                        state["erank_ema"] = erank
                    else:
                        state["erank_ema"] = (
                            erank_ema_beta * old_ema
                            + (1.0 - erank_ema_beta) * erank
                        )

                    desired = int(round(rho * state["erank_ema"]))

                    # Quality control via approximation error
                    if use_quality:
                        # Compute relative Frobenius error (distributed)
                        num_sq = (B - approx).float().pow(2).sum()
                        den_sq = B.float().pow(2).sum()
                        self._all_reduce(num_sq)
                        self._all_reduce(den_sq)
                        aerr = (num_sq.sqrt() / (den_sq.sqrt() + 1e-12)).item()

                        old_aerr = state["aerr_ema"]
                        if old_aerr is None:
                            state["aerr_ema"] = aerr
                        else:
                            state["aerr_ema"] = (
                                aerr_ema_beta * old_aerr
                                + (1.0 - aerr_ema_beta) * aerr
                            )

                        if aerr > max(aerr_target,
                                      state["aerr_ema"] * (1.0 + aerr_up_margin)):
                            desired = max(desired, r + step_up)
                        if aerr < min(0.5 * aerr_target,
                                      state["aerr_ema"] * (1.0 - aerr_down_margin)):
                            desired = min(desired, r - step_down)

                    # Clamp and quantize
                    desired = min(max(rmin_eff, desired), r_cap, Qbuf.shape[1])
                    desired = _quantize_int(desired, rank_quantize)
                    desired = min(max(rmin_eff, desired), r_cap, Qbuf.shape[1])

                    # Rate-limit rank changes
                    if desired > r:
                        r_next = min(desired, r + step_up)
                    elif desired < r:
                        r_next = max(desired, r - step_down)
                    else:
                        r_next = r
                    r_next = int(min(max(rmin_eff, r_next), r_cap, Qbuf.shape[1]))

                    # Fill new Qbuf columns with random init
                    if r_next > r:
                        qbuf_dim = Qbuf.shape[0]
                        new_cols = r_next - r
                        Qbuf[:, r:r_next].copy_(F.normalize(
                            torch.randn(qbuf_dim, new_cols,
                                        device=p_local.device, dtype=orig_dtype),
                            dim=0,
                        ))

                    state["r"] = r_next

    # ------------------------------------------------------------------
    # Checkpoint safety
    # ------------------------------------------------------------------

    def state_dict(self):
        """Return state dict with a warning about world_size constraints."""
        sd = super().state_dict()
        world_size = self._get_world_size()
        if world_size > 1:
            sd["_adadion_world_size"] = world_size
            logger.info(
                "AdaDion state_dict saved with world_size=%d. "
                "Resuming with a different world_size is NOT supported "
                "(optimizer states M, Qbuf are stored as local shards).",
                world_size,
            )
        return sd

    def load_state_dict(self, state_dict):
        """Load state dict with world_size mismatch check."""
        saved_ws = state_dict.pop("_adadion_world_size", None)
        current_ws = self._get_world_size()
        if saved_ws is not None and saved_ws != current_ws:
            raise RuntimeError(
                f"AdaDion checkpoint was saved with world_size={saved_ws}, "
                f"but current world_size={current_ws}. "
                f"Resuming with a different world_size is not supported. "
                f"Optimizer states (M, Qbuf) are stored as local shards."
            )
        super().load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_rank(self) -> dict[str, int]:
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "r" in state:
                    result[f"param_{idx}"] = state["r"]
                idx += 1
        return result

    def get_effective_rank(self) -> dict[str, float]:
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "erank_ema" in state and state["erank_ema"] is not None:
                    result[f"param_{idx}"] = state["erank_ema"]
                idx += 1
        return result

    def get_aerr(self) -> dict[str, float]:
        """Return EMA of approximation error per matrix parameter."""
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "aerr_ema" in state and state["aerr_ema"] is not None:
                    result[f"param_{idx}"] = state["aerr_ema"]
                idx += 1
        return result
