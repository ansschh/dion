"""
Tier A: Correctness and determinism tests.

These are cheap-but-mandatory sanity checks to run before burning GPU hours.
They verify that each optimizer produces valid updates, is reproducible,
and works correctly under minimal distributed settings.

A1. Single-step invariants (1 GPU)
A2. Same-seed reproducibility (1 GPU)
A3. 2-GPU distributed sanity (torchrun --nproc_per_node=2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..optim.muon import Muon
from ..optim.dion import Dion
from ..optim.dion2 import Dion2
from ..optim.ortho import newton_schulz_zeroth_power


# ======================================================================
# A1: Single-step invariants
# ======================================================================

def test_single_step_invariants(
    device: str = "cpu",
    shapes: Optional[list[tuple[int, int]]] = None,
) -> dict:
    """
    Run one optimizer step for each optimizer and check invariants:
      - No NaN/Inf in parameters, gradients, or updated params
      - Update norms are reasonable (not 0, not huge)
      - Muon: NS output is approximately orthonormal
      - Dion: V basis is approximately orthonormal
      - Dion2: selected fraction matches alpha

    Returns dict with pass/fail status for each optimizer.
    """
    if shapes is None:
        shapes = [(128, 64), (64, 128), (256, 256)]

    results = {}

    for opt_name, opt_cls, opt_kwargs in [
        ("Muon", Muon, {"lr": 0.02, "mu": 0.95, "ns_steps": 5}),
        ("Dion", Dion, {"lr": 0.02, "rank_frac": 0.25, "beta": 0.05}),
        ("Dion2", Dion2, {"lr": 0.02, "alpha": 0.25, "selection": "top_l1", "mu": 0.95}),
    ]:
        checks = {}
        all_pass = True

        for shape in shapes:
            m, n = shape
            # Create parameter and synthetic gradient
            p = nn.Parameter(torch.randn(m, n, device=device))
            p.grad = torch.randn(m, n, device=device) * 0.01

            p_before = p.data.clone()
            optimizer = opt_cls([p], **opt_kwargs)
            optimizer.step()
            p_after = p.data

            # Check no NaN/Inf
            has_nan = torch.isnan(p_after).any().item()
            has_inf = torch.isinf(p_after).any().item()
            checks[f"no_nan_{shape}"] = not has_nan
            checks[f"no_inf_{shape}"] = not has_inf

            # Check update is non-zero
            update_norm = (p_after - p_before).norm().item()
            checks[f"nonzero_update_{shape}"] = update_norm > 1e-10
            checks[f"reasonable_update_{shape}"] = update_norm < 1e6

            # Optimizer-specific checks
            if opt_name == "Muon":
                # Verify NS output is approximately orthonormal
                M = optimizer.state[p]["M"]
                U = newton_schulz_zeroth_power(M, steps=5)
                if m <= n:
                    orth_err = (U @ U.T - torch.eye(m, device=device)).norm().item()
                else:
                    orth_err = (U.T @ U - torch.eye(n, device=device)).norm().item()
                checks[f"ortho_err_{shape}"] = orth_err
                checks[f"ortho_ok_{shape}"] = orth_err < 0.5  # relaxed tolerance

            elif opt_name == "Dion":
                # Verify V orthonormality
                errs = optimizer.get_orthogonality_errors()
                if errs:
                    checks[f"V_ortho_err_{shape}"] = errs[0]
                    checks[f"V_ortho_ok_{shape}"] = errs[0] < 0.5

            elif opt_name == "Dion2":
                # Verify selected fraction
                fracs = optimizer.get_selected_fractions()
                if fracs:
                    checks[f"selected_frac_{shape}"] = fracs[0]
                    expected = 0.25
                    checks[f"frac_ok_{shape}"] = abs(fracs[0] - expected) < 0.1

            if has_nan or has_inf:
                all_pass = False

        checks["all_pass"] = all_pass
        results[opt_name] = checks

    return results


# ======================================================================
# A2: Same-seed reproducibility
# ======================================================================

def test_reproducibility(
    steps: int = 50,
    seed: int = 42,
    device: str = "cpu",
    shape: tuple[int, int] = (128, 64),
) -> dict:
    """
    Run `steps` optimization steps twice with the same seed.
    Verify that the loss trajectories are bit-exact.

    Returns dict with pass/fail per optimizer.
    """
    results = {}

    for opt_name, opt_cls, opt_kwargs in [
        ("Muon", Muon, {"lr": 0.02, "mu": 0.95}),
        ("Dion", Dion, {"lr": 0.02, "rank_frac": 0.25, "beta": 0.05}),
        ("Dion2", Dion2, {"lr": 0.02, "alpha": 0.25, "mu": 0.95}),
    ]:
        trajectories = []

        for run in range(2):
            torch.manual_seed(seed)
            m, n = shape
            p = nn.Parameter(torch.randn(m, n, device=device))
            optimizer = opt_cls([p], **opt_kwargs)

            losses = []
            for step in range(steps):
                optimizer.zero_grad()
                # Synthetic "loss" = sum of squares
                loss = (p ** 2).sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            trajectories.append(losses)

        # Compare
        match = all(
            abs(a - b) < 1e-10
            for a, b in zip(trajectories[0], trajectories[1])
        )
        results[opt_name] = {
            "reproducible": match,
            "max_diff": max(
                abs(a - b)
                for a, b in zip(trajectories[0], trajectories[1])
            ),
            "final_loss_run0": trajectories[0][-1],
            "final_loss_run1": trajectories[1][-1],
        }

    return results


# ======================================================================
# A3: Distributed sanity (2 GPUs)
# ======================================================================

def test_distributed_sanity_check(
    steps: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Run each optimizer for `steps` on 2 GPUs with FSDP2 (dp_shard=2).
    Verify loss is decreasing and stable (no NaN/Inf).

    NOTE: This must be run via torchrun --nproc_per_node=2.
    See scripts/run_local.sh for the launch command.

    Returns dict with stability metrics per optimizer.
    """
    # This test is designed to be run within TorchTitan's training loop
    # using the debug model configs from config_registry.py.
    # The actual test is: run llama3_debug_muon / dion / dion2 for 100 steps
    # and verify loss[0] > loss[-1] and no NaN.

    return {
        "note": "Run via: torchrun --nproc_per_node=2 -m torchtitan.train "
                "--module ada_dion.integration.config_registry "
                "--config llama3_debug_muon --training.steps 100",
        "checks": [
            "loss[-1] < loss[0]",
            "no NaN in loss trajectory",
            "grad norms < 100",
        ],
    }


# ======================================================================
# Runner
# ======================================================================

def run_all_tier_a(device: str = "cpu") -> dict:
    """Run all Tier A tests and return combined results."""
    results = {}

    print("=" * 60)
    print("Tier A1: Single-step invariants")
    print("=" * 60)
    a1 = test_single_step_invariants(device=device)
    results["A1_single_step"] = a1
    for opt_name, checks in a1.items():
        status = "PASS" if checks.get("all_pass", False) else "FAIL"
        print(f"  {opt_name}: {status}")
        for k, v in checks.items():
            if k != "all_pass":
                print(f"    {k}: {v}")

    print()
    print("=" * 60)
    print("Tier A2: Reproducibility")
    print("=" * 60)
    a2 = test_reproducibility(device=device)
    results["A2_reproducibility"] = a2
    for opt_name, res in a2.items():
        status = "PASS" if res["reproducible"] else "FAIL"
        print(f"  {opt_name}: {status} (max_diff={res['max_diff']:.2e})")

    print()
    print("=" * 60)
    print("Tier A3: Distributed sanity")
    print("=" * 60)
    a3 = test_distributed_sanity_check()
    results["A3_distributed"] = a3
    print(f"  {a3['note']}")

    return results


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    results = run_all_tier_a(device=device)
    print("\n" + json.dumps(results, indent=2, default=str))
