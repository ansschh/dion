"""Tests for the Dion optimizer (official microsoft/dion package)."""
import pytest
import torch
import torch.nn as nn

from dion import Dion


class TestDionBasics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Dion([p], lr=0.02, rank_fraction=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_different_rank_fractions(self):
        for rf in [0.1, 0.25, 0.5, 1.0]:
            p = nn.Parameter(torch.randn(32, 16))
            opt = Dion([p], lr=0.02, rank_fraction=rf)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with rank_fraction={rf}"

    def test_weight_decay(self):
        p = nn.Parameter(torch.ones(16, 8))
        opt = Dion([p], lr=0.02, rank_fraction=0.25, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()
        assert norm_after < norm_before


class TestDionConvergence:
    def test_loss_decreases(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.01, rank_fraction=0.25)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"
