"""
Extended metrics collector for optimizer-internal telemetry.

Collects metrics beyond what TorchTitan's MetricsProcessor provides:
  - Dion: rank, residual norm, orthogonality error, error-feedback magnitude
  - Dion2: alpha, selected fraction, sparsity
  - Muon: momentum norms
  - All: gradient norms, update norms per param group

Writes to JSON lines file and optionally to TensorBoard/W&B.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

from ..optim.muon import Muon
from ..optim.dion import Dion
from ..optim.dion2 import Dion2


class OptimizerMetricsCollector:
    """
    Collects optimizer-internal metrics at each step.

    Usage:
        collector = OptimizerMetricsCollector(
            matrix_optimizers=hybrid_container.get_matrix_optimizers(),
            output_dir="./metrics",
        )
        # In training loop:
        collector.collect(step=step)
        # After training:
        collector.close()
    """

    def __init__(
        self,
        matrix_optimizers: list,
        output_dir: str = "./metrics",
        tb_writer=None,
        wandb_run=None,
    ):
        self.matrix_optimizers = matrix_optimizers
        self.output_dir = output_dir
        self.tb_writer = tb_writer
        self.wandb_run = wandb_run

        self._history: dict[str, list] = defaultdict(list)
        self._jsonl_path = os.path.join(output_dir, "optimizer_metrics.jsonl")
        os.makedirs(output_dir, exist_ok=True)
        self._jsonl_file = open(self._jsonl_path, "a")

    def collect(self, step: int) -> dict:
        """
        Collect metrics from all matrix optimizers at current step.

        Returns dict of metric_name -> value.
        """
        metrics = {"step": step, "timestamp": time.time()}

        for i, opt in enumerate(self.matrix_optimizers):
            prefix = f"opt_{i}"

            if isinstance(opt, Dion):
                # Dion-specific metrics
                ortho_errs = opt.get_orthogonality_errors()
                residual_norms = opt.get_residual_norms()
                ranks = opt.get_ranks()

                if ortho_errs:
                    metrics[f"{prefix}/dion_ortho_error_mean"] = (
                        sum(ortho_errs) / len(ortho_errs)
                    )
                    metrics[f"{prefix}/dion_ortho_error_max"] = max(ortho_errs)
                if residual_norms:
                    metrics[f"{prefix}/dion_residual_norm_mean"] = (
                        sum(residual_norms) / len(residual_norms)
                    )
                if ranks:
                    metrics[f"{prefix}/dion_rank_mean"] = (
                        sum(ranks) / len(ranks)
                    )

            elif isinstance(opt, Dion2):
                # Dion2-specific metrics
                fracs = opt.get_selected_fractions()
                counts = opt.get_selected_counts()
                sparsity = opt.get_sparsity()
                mom_norms = opt.get_momentum_norms()

                metrics[f"{prefix}/dion2_alpha"] = opt.get_alpha()
                metrics[f"{prefix}/dion2_sparsity"] = sparsity
                if fracs:
                    metrics[f"{prefix}/dion2_selected_frac_mean"] = (
                        sum(fracs) / len(fracs)
                    )
                if counts:
                    metrics[f"{prefix}/dion2_selected_count_mean"] = (
                        sum(counts) / len(counts)
                    )
                if mom_norms:
                    metrics[f"{prefix}/dion2_momentum_norm_mean"] = (
                        sum(mom_norms) / len(mom_norms)
                    )

            elif isinstance(opt, Muon):
                # Muon-specific metrics
                mom_norms = opt.get_momentum_norms()
                if mom_norms:
                    metrics[f"{prefix}/muon_momentum_norm_mean"] = (
                        sum(mom_norms) / len(mom_norms)
                    )

            # Generic gradient/param metrics
            total_grad_norm = 0.0
            total_param_norm = 0.0
            n_params = 0
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total_grad_norm += p.grad.norm().item() ** 2
                        total_param_norm += p.data.norm().item() ** 2
                        n_params += 1

            if n_params > 0:
                metrics[f"{prefix}/grad_norm"] = total_grad_norm ** 0.5
                metrics[f"{prefix}/param_norm"] = total_param_norm ** 0.5

        # Store and write
        for k, v in metrics.items():
            if k not in ("step", "timestamp"):
                self._history[k].append(v)

        # Write to JSONL
        self._jsonl_file.write(json.dumps(metrics, default=str) + "\n")
        self._jsonl_file.flush()

        # Write to TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if k not in ("step", "timestamp") and isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"optimizer/{k}", v, step)

        # Write to W&B
        if self.wandb_run is not None:
            log_dict = {
                f"optimizer/{k}": v
                for k, v in metrics.items()
                if k not in ("step", "timestamp") and isinstance(v, (int, float))
            }
            self.wandb_run.log(log_dict, step=step)

        return metrics

    def get_history(self) -> dict[str, list]:
        """Return the full history of collected metrics."""
        return dict(self._history)

    def close(self):
        """Flush and close output files."""
        if self._jsonl_file:
            self._jsonl_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
