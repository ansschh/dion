"""
Comprehensive pre-push validation for AdaDion.

Tests every failure mode: bf16/fp16/fp32 dtypes, all matrix shapes,
rank fractions, anchor mixing, gap monitoring, AdamW routing, mixed
precision groups, edge cases, long-run stability, simulated FSDP sharding.

Run: python -m pytest ada_dion/tests/test_adadion_prepush.py -v
"""
import sys
import types
import copy
import math

import torch
import torch.nn as nn
import pytest

# Mock dion if triton unavailable
try:
    from dion import Muon, Dion, Dion2
except (ImportError, ModuleNotFoundError):
    mock = types.ModuleType("dion")
    mock.__path__ = []

    class _MockOpt(torch.optim.Optimizer):
        def __init__(self, params, **kw):
            super().__init__(params, kw)
        def step(self, closure=None):
            pass

    mock.Muon = type("Muon", (_MockOpt,), {})
    mock.Dion = type("Dion", (_MockOpt,), {})
    mock.Dion2 = type("Dion2", (_MockOpt,), {})
    sys.modules["dion"] = mock
    for sub in ["dion.muon", "dion.dion", "dion.dion2",
                 "dion.newton_schulz_triton", "dion.utils"]:
        sys.modules[sub] = types.ModuleType(sub)

from ada_dion.optim.adadion import AdaDion


# ================================================================
# Section 1: Dtype Compatibility
# ================================================================

class TestDtypeCompat:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_10_steps(self, dtype):
        p = nn.Parameter(torch.randn(128, 64, dtype=dtype))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        for i in range(10):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
            assert not torch.isnan(p).any(), f"NaN at step {i}"
            assert not torch.isinf(p).any(), f"Inf at step {i}"


# ================================================================
# Section 2: Matrix Shapes
# ================================================================

class TestShapes:
    @pytest.mark.parametrize("shape", [
        (8, 8), (16, 4), (4, 16), (64, 64), (128, 256),
        (256, 128), (512, 512), (1024, 768), (2, 2),
    ])
    def test_shape_bf16(self, shape):
        p = nn.Parameter(torch.randn(*shape, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p) * 0.01
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        opt.step()
        assert not torch.isnan(p).any()

    def test_shape_1x1_bf16(self):
        """1x1 matrix — rank=1, edge case for eigh."""
        p = nn.Parameter(torch.randn(1, 1, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p) * 0.01
        opt = AdaDion([p], lr=0.02, rank_fraction=1.0)
        opt.step()
        assert not torch.isnan(p).any()


# ================================================================
# Section 3: Rank Fractions
# ================================================================

class TestRankFractions:
    @pytest.mark.parametrize("rf", [0.01, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_rank_fraction_bf16(self, rf):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=rf)
        for i in range(10):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
            assert not torch.isnan(p).any()


# ================================================================
# Section 4: Anchor & Gap Monitoring (bf16)
# ================================================================

class TestAnchorGap:
    def test_anchor_mixing_bf16(self):
        p = nn.Parameter(torch.randn(64, 128, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.5,
                      anchor_lambda=0.2, anchor_rho=0.95)
        for i in range(30):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()
        d = opt.get_anchor_drift()
        assert len(d) > 0 and all(v >= 0 for v in d.values())

    def test_gap_refresh_every_step_bf16(self):
        """tau_lo=0.99 forces Q refresh nearly every step."""
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25, tau_lo=0.99)
        for i in range(20):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()

    def test_periodic_refresh_bf16(self):
        """refresh_period=5 triggers refresh at steps 5,10,15,20,25."""
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25, refresh_period=5)
        for i in range(25):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()

    def test_no_refresh_bf16(self):
        """tau_lo=0.0 and large refresh_period disables refresh."""
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25,
                      tau_lo=0.0, refresh_period=99999)
        for i in range(50):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()


# ================================================================
# Section 5: AdamW Routing (mixed precision)
# ================================================================

class TestAdamWRouting:
    def test_mixed_bf16_fp32_groups(self):
        p_mat = nn.Parameter(torch.randn(128, 64, dtype=torch.bfloat16))
        p_embed = nn.Parameter(torch.randn(100, 64, dtype=torch.float32))
        p_norm = nn.Parameter(torch.randn(64, dtype=torch.float32))
        p_bias = nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
        groups = [
            {"params": [p_mat]},
            {"params": [p_embed], "algorithm": "adamw", "lr": 3e-4,
             "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01},
            {"params": [p_norm], "algorithm": "adamw", "lr": 3e-4,
             "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0},
            {"params": [p_bias], "algorithm": "adamw", "lr": 3e-4,
             "betas": (0.9, 0.95), "eps": 1e-8},
        ]
        opt = AdaDion(groups, lr=0.02, rank_fraction=0.25)
        for i in range(20):
            for p in [p_mat, p_embed, p_norm, p_bias]:
                p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        for p, name in [(p_mat, "mat"), (p_embed, "embed"),
                        (p_norm, "norm"), (p_bias, "bias")]:
            assert not torch.isnan(p).any(), f"NaN in {name}"


# ================================================================
# Section 6: Convergence
# ================================================================

class TestConvergence:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_convergence_100steps(self, dtype):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(64, 64, dtype=dtype))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        initial_loss = (p.float() ** 2).sum().item()
        for i in range(100):
            opt.zero_grad()
            loss = (p.float() ** 2).sum()
            loss.backward()
            opt.step()
        final_loss = (p.float() ** 2).sum().item()
        assert final_loss < initial_loss, \
            f"No convergence: {initial_loss:.2f} -> {final_loss:.2f}"


# ================================================================
# Section 7: Diagnostics
# ================================================================

class TestDiagnostics:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_all_diagnostics(self, dtype):
        p = nn.Parameter(torch.randn(128, 64, dtype=dtype))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        for i in range(5):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        d = opt.get_anchor_drift()
        t = opt.get_tail_ratio()
        r = opt.get_rank()
        e = opt.get_energy_captured()
        assert len(d) > 0
        assert len(t) > 0
        assert len(r) > 0
        assert len(e) > 0
        for v in t.values():
            assert 0 <= v <= 1, f"tau={v}"
        for v in e.values():
            assert 0 <= v <= 1.01, f"energy={v}"
        for v in r.values():
            assert v >= 1, f"rank={v}"


# ================================================================
# Section 8: State Dict Save/Load
# ================================================================

class TestStateDict:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_state_dict_roundtrip(self, dtype):
        torch.manual_seed(42)
        p1 = nn.Parameter(torch.randn(64, 64, dtype=dtype))
        opt1 = AdaDion([p1], lr=0.02, rank_fraction=0.25,
                       tau_lo=0.0, refresh_period=9999)
        for i in range(10):
            p1.grad = torch.randn(64, 64, dtype=dtype) * 0.01
            opt1.step()
            opt1.zero_grad()
        sd = copy.deepcopy(opt1.state_dict())

        torch.manual_seed(42)
        p2 = nn.Parameter(torch.randn(64, 64, dtype=dtype))
        opt2 = AdaDion([p2], lr=0.02, rank_fraction=0.25,
                       tau_lo=0.0, refresh_period=9999)
        for i in range(10):
            p2.grad = torch.randn(64, 64, dtype=dtype) * 0.01
            opt2.step()
            opt2.zero_grad()
        opt2.load_state_dict(sd)
        p2.grad = torch.randn(64, 64, dtype=dtype) * 0.01
        opt2.step()
        assert not torch.isnan(p2).any()


# ================================================================
# Section 9: Edge Cases
# ================================================================

class TestEdgeCases:
    def test_zero_gradient_bf16(self):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        p.grad = torch.zeros_like(p)
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        opt.step()
        assert not torch.isnan(p).any()

    def test_large_gradient_bf16(self):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p) * 1000
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_tiny_gradient_bf16(self):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        p.grad = torch.randn_like(p) * 1e-10
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        opt.step()
        assert not torch.isnan(p).any()

    def test_weight_decay_bf16(self):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25, weight_decay=0.1)
        for i in range(20):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()

    def test_some_params_no_grad(self):
        p1 = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        p2 = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p1, p2], lr=0.02, rank_fraction=0.25)
        p1.grad = torch.randn_like(p1) * 0.01
        # p2 has no grad
        opt.step()
        assert not torch.isnan(p1).any()

    def test_1d_param_in_default_group(self):
        p_mat = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        p_1d = nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
        opt = AdaDion([p_mat, p_1d], lr=0.02, rank_fraction=0.25)
        p_mat.grad = torch.randn_like(p_mat) * 0.01
        p_1d.grad = torch.randn_like(p_1d) * 0.01
        opt.step()
        assert not torch.isnan(p_mat).any()
        assert not torch.isnan(p_1d).any()

    def test_closure(self):
        p = nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)

        def closure():
            opt.zero_grad()
            loss = (p.float() ** 2).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert not torch.isnan(p).any()


# ================================================================
# Section 10: Long Run Stability
# ================================================================

class TestLongRun:
    def test_200_steps_bf16_all_features(self):
        p = nn.Parameter(torch.randn(256, 256, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25,
                      anchor_lambda=0.1, anchor_rho=0.99,
                      tau_lo=0.3, refresh_period=50)
        for i in range(200):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
            if i % 50 == 0:
                assert not torch.isnan(p).any(), f"NaN at step {i}"
                assert not torch.isinf(p).any(), f"Inf at step {i}"
        d = opt.get_anchor_drift()
        t = opt.get_tail_ratio()
        assert len(d) > 0
        assert len(t) > 0

    def test_500_steps_large_bf16(self):
        """Closest simulation to real HPC: 1024x768 bf16, 500 steps."""
        p = nn.Parameter(torch.randn(1024, 768, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25,
                      anchor_lambda=0.1, anchor_rho=0.99,
                      tau_lo=0.3, refresh_period=100)
        for i in range(500):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()


# ================================================================
# Section 11: Simulated FSDP Sharding
# ================================================================

class TestSimulatedFSDP:
    @pytest.mark.parametrize("shard_m", [32, 64, 128, 256])
    def test_fsdp_shard_bf16(self, shard_m):
        """Simulate FSDP: sharded dim-0, full dim-1."""
        n = 256
        p = nn.Parameter(torch.randn(shard_m, n, dtype=torch.bfloat16))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        for i in range(20):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()
        assert not torch.isnan(p).any()


# ================================================================
# Section 12: Full Model Simulation (SimpleModel + param_grouper)
# ================================================================

class TestFullModelSimulation:
    def test_simple_model_bf16_training(self):
        """Train a SimpleModel with bf16 weights — closest to real HPC."""

        class SimpleModel(nn.Module):
            def __init__(self, dim=64, hidden_dim=128, vocab_size=200):
                super().__init__()
                self.tok_embeddings = nn.Embedding(vocab_size, dim)
                self.layers = nn.ModuleList()
                for _ in range(2):
                    layer = nn.Module()
                    attn = nn.Module()
                    attn.wq = nn.Linear(dim, dim, bias=False)
                    attn.wk = nn.Linear(dim, dim, bias=False)
                    attn.wv = nn.Linear(dim, dim, bias=False)
                    attn.wo = nn.Linear(dim, dim, bias=False)
                    layer.attention = attn
                    layer.attention_norm = nn.LayerNorm(dim)
                    ff = nn.Module()
                    ff.w1 = nn.Linear(dim, hidden_dim, bias=False)
                    ff.w2 = nn.Linear(hidden_dim, dim, bias=False)
                    ff.w3 = nn.Linear(dim, hidden_dim, bias=False)
                    layer.feed_forward = ff
                    layer.ffn_norm = nn.LayerNorm(dim)
                    self.layers.append(layer)
                self.norm = nn.LayerNorm(dim)
                self.output = nn.Linear(dim, vocab_size, bias=False)

            def forward(self, x):
                h = self.tok_embeddings(x)
                for layer in self.layers:
                    q = layer.attention.wq(h)
                    v = layer.attention.wv(h)
                    h = h + layer.attention.wo(v)
                    h = layer.attention_norm(h)
                    h = h + layer.feed_forward.w2(
                        torch.relu(layer.feed_forward.w1(h))
                        * layer.feed_forward.w3(h)
                    )
                    h = layer.ffn_norm(h)
                h = self.norm(h)
                return self.output(h)

        from ada_dion.integration.param_grouper import group_params_for_hybrid

        torch.manual_seed(42)
        model = SimpleModel()
        # Convert entire model to bf16 to simulate FSDP2
        model = model.to(torch.bfloat16)

        groups = group_params_for_hybrid(model)
        param_groups = [{"params": groups.matrix_params}]
        if groups.embed_params:
            param_groups.append({
                "params": groups.embed_params,
                "algorithm": "adamw", "lr": 3e-4,
                "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01,
            })
        if groups.output_params:
            param_groups.append({
                "params": groups.output_params,
                "algorithm": "adamw", "lr": 3e-4 / math.sqrt(64),
                "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01,
            })
        if groups.norm_params:
            param_groups.append({
                "params": groups.norm_params,
                "algorithm": "adamw", "lr": 3e-4,
                "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            })

        opt = AdaDion(param_groups, lr=0.02, rank_fraction=0.25)

        vocab_size = 200
        losses = []
        for step in range(50):
            x = torch.randint(0, vocab_size, (4, 32))
            target = torch.randint(0, vocab_size, (4, 32))
            opt.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), target.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())

        # Verify no NaN in any parameter
        for name, p in model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in {name}"

        # Verify diagnostics work
        d = opt.get_anchor_drift()
        t = opt.get_tail_ratio()
        r = opt.get_rank()
        e = opt.get_energy_captured()
        assert len(d) > 0
        assert len(t) > 0
