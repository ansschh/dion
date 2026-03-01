"""Tests for the Dion2 optimizer (official microsoft/dion package)."""
import pytest
import torch
import torch.nn as nn

from dion import Dion2


class TestDion2Basics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion2([p], lr=0.02, fraction=0.25, ef_decay=0.95)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Dion2([p], lr=0.02, fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_weight_decay_applies_to_all(self):
        """Weight decay should affect the entire parameter tensor."""
        p = nn.Parameter(torch.ones(32, 64))
        opt = Dion2([p], lr=0.02, fraction=0.25, weight_decay=0.5)

        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()

        assert norm_after < norm_before, "Weight decay should shrink all params"

    def test_different_fractions(self):
        for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            p = nn.Parameter(torch.randn(40, 80))
            opt = Dion2([p], lr=0.02, fraction=frac)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with fraction={frac}"


class TestDion2Convergence:
    def test_loss_decreases(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 64))
        opt = Dion2([p], lr=0.01, fraction=0.5, ef_decay=0.9)

        losses = []
        for _ in range(100):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"
