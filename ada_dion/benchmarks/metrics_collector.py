"""
Extended metrics collector for optimizer telemetry.

Collects generic gradient/param norm metrics that work with any optimizer.
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

from dion import Muon, Dion, Dion2


class OptimizerMetricsCollector:
    """
    Collects optimizer metrics at each step.

    Usage:
        collector = OptimizerMetricsCollector(
            matrix_optimizers=hybrid_container.get_optimizers(),
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
        Collect metrics from all optimizers at current step.

        Returns dict of metric_name -> value.
        """
        metrics = {"step": step, "timestamp": time.time()}

        for i, opt in enumerate(self.matrix_optimizers):
            prefix = f"opt_{i}"

            # Identify optimizer type for logging
            if isinstance(opt, Dion2):
                metrics[f"{prefix}/type"] = "Dion2"
            elif isinstance(opt, Dion):
                metrics[f"{prefix}/type"] = "Dion"
            elif isinstance(opt, Muon):
                metrics[f"{prefix}/type"] = "Muon"

            # Generic gradient/param metrics (work with any optimizer)
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
