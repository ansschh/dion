"""Tests for the vanilla Muon optimizer."""
import pytest
import torch
import torch.nn as nn

from ada_dion.optim.muon import Muon


class TestMuonBasics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.02, mu=0.95, ns_steps=5)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Muon([p], lr=0.02, mu=0.95, ns_steps=5)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_ema_momentum(self):
        """Momentum should use EMA: M = mu*M + (1-mu)*G."""
        p = nn.Parameter(torch.randn(16, 8))
        opt = Muon([p], lr=0.02, mu=0.9, nesterov=False, ns_steps=5)

        g = torch.randn_like(p)
        p.grad = g.clone()
        opt.step()

        M = opt.state[p]["M"]
        # After first step with M=0: M = 0.9*0 + 0.1*G = 0.1*G
        expected = 0.1 * g.to(torch.float32)
        assert torch.allclose(M, expected, atol=1e-5), \
            "First step EMA: M should be (1-mu)*G"

    def test_nesterov_enabled_by_default(self):
        """Nesterov should be enabled by default."""
        p = nn.Parameter(torch.randn(16, 8))
        opt = Muon([p], lr=0.02)
        assert opt.param_groups[0]["nesterov"] is True

    def test_nesterov_vs_plain_differ(self):
        """Nesterov and plain momentum should produce different updates."""
        torch.manual_seed(42)
        g = torch.randn(32, 16)

        # First do 2 steps with Nesterov
        p1 = nn.Parameter(torch.randn(32, 16))
        opt1 = Muon([p1], lr=0.02, mu=0.9, nesterov=True)
        p1.grad = g.clone()
        opt1.step()
        p1.grad = g.clone()
        opt1.step()

        # Then 2 steps without Nesterov
        p2 = nn.Parameter(p1.data.clone())  # start from same initial
        # Actually need same initial — re-init
        torch.manual_seed(42)
        p2 = nn.Parameter(torch.randn(32, 16))
        opt2 = Muon([p2], lr=0.02, mu=0.9, nesterov=False)
        p2.grad = g.clone()
        opt2.step()
        p2.grad = g.clone()
        opt2.step()

        # After 2 steps, they should differ
        assert not torch.equal(p1.data, p2.data), \
            "Nesterov and plain momentum should produce different parameters"

    def test_momentum_accumulation(self):
        """Momentum should accumulate over steps."""
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.02, mu=0.95, ns_steps=5)

        for _ in range(5):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()

        M = opt.state[p]["M"]
        assert M.norm().item() > 0

    def test_weight_decay(self):
        """Weight decay should shrink parameters."""
        p = nn.Parameter(torch.ones(16, 8))
        opt = Muon([p], lr=0.02, mu=0.95, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()
        assert norm_after < norm_before

    def test_rejects_1d(self):
        p = nn.Parameter(torch.randn(32))
        opt = Muon([p], lr=0.02)
        p.grad = torch.randn_like(p)
        with pytest.raises(ValueError, match="ndim>=2"):
            opt.step()

    def test_multiple_shapes(self):
        """Test with different matrix shapes."""
        for shape in [(64, 32), (32, 64), (128, 128)]:
            p = nn.Parameter(torch.randn(*shape))
            opt = Muon([p], lr=0.02, mu=0.95, ns_steps=5)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN for shape {shape}"

    def test_diagnostic_momentum_norms(self):
        p1 = nn.Parameter(torch.randn(16, 8))
        p2 = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p1, p2], lr=0.02)

        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        opt.step()

        norms = opt.get_momentum_norms()
        assert len(norms) == 2
        assert all(n > 0 for n in norms)


class TestMuonConvergence:
    def test_loss_decreases(self):
        """Verify loss decreases over multiple steps."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.01, mu=0.9, ns_steps=5)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"

    def test_loss_decreases_without_nesterov(self):
        """Verify loss also decreases without Nesterov."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.01, mu=0.9, nesterov=False, ns_steps=5)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"
