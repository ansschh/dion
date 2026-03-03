"""Tests for the AdaDion optimizer."""
import pytest
import torch
import torch.nn as nn

from ada_dion.optim import AdaDion


class TestAdaDionBasics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_different_rank_fractions(self):
        for rf in [0.1, 0.25, 0.5, 1.0]:
            p = nn.Parameter(torch.randn(32, 16))
            opt = AdaDion([p], lr=0.02, rank_fraction=rf)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with rank_fraction={rf}"

    def test_weight_decay(self):
        p = nn.Parameter(torch.ones(16, 8))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()
        assert norm_after < norm_before


class TestAdaDionConvergence:
    def test_loss_decreases(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.01, rank_fraction=0.25)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"


class TestAdaDionAnchor:
    def test_anchor_initialized_after_first_step(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        state = opt.state[p]
        assert state["Q_anc"] is not None

    def test_anchor_drift_positive(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        # Run a few steps to let anchor and Q diverge
        for _ in range(5):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
        drift = opt.get_anchor_drift()
        assert len(drift) > 0
        for v in drift.values():
            assert v >= 0.0


class TestAdaDionTailRatio:
    def test_tau_in_range(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        ratios = opt.get_tail_ratio()
        assert len(ratios) > 0
        for v in ratios.values():
            assert 0.0 <= v <= 1.0, f"tau={v} not in [0, 1]"


class TestAdaDionDiagnostics:
    def test_all_diagnostics_return_non_empty(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()

        assert len(opt.get_anchor_drift()) > 0
        assert len(opt.get_tail_ratio()) > 0
        assert len(opt.get_rank()) > 0
        assert len(opt.get_energy_captured()) > 0


class TestAdaDionAdamWRouting:
    def test_adamw_param_group(self):
        """Verify algorithm='adamw' param groups use AdamW logic."""
        # Matrix param (spectral)
        p_matrix = nn.Parameter(torch.randn(32, 16))
        # Scalar param (adamw)
        p_scalar = nn.Parameter(torch.randn(64))

        param_groups = [
            {"params": [p_matrix]},
            {
                "params": [p_scalar],
                "algorithm": "adamw",
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ]
        opt = AdaDion(param_groups, lr=0.02, rank_fraction=0.25)

        p_matrix_before = p_matrix.data.clone()
        p_scalar_before = p_scalar.data.clone()

        p_matrix.grad = torch.randn_like(p_matrix) * 0.01
        p_scalar.grad = torch.randn_like(p_scalar) * 0.01

        opt.step()

        # Both should be updated
        assert not torch.equal(p_matrix.data, p_matrix_before)
        assert not torch.equal(p_scalar.data, p_scalar_before)
        # Neither should have NaN
        assert not torch.isnan(p_matrix).any()
        assert not torch.isnan(p_scalar).any()
