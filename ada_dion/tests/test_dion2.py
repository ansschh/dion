"""Tests for the Dion2 optimizer."""
import pytest
import torch
import torch.nn as nn

from ada_dion.optim.dion2 import Dion2


class TestDion2Basics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion2([p], lr=0.02, alpha=0.25, selection="top_l1", mu=0.95)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Dion2([p], lr=0.02, alpha=0.25, mu=0.95)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_selected_fraction_wide_matrix(self):
        """Wide matrix (m <= n): should select rows (dim=0)."""
        # 50 x 100 -> m=50 <= n=100, select_dim=0, select 25% of 50 = 13
        p = nn.Parameter(torch.randn(50, 100))
        opt = Dion2([p], lr=0.02, alpha=0.25, selection="top_l1", mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()

        counts = opt.get_selected_counts()
        assert counts[0] == round(50 * 0.25)  # 12 or 13 depending on rounding
        dims = opt.get_select_dims()
        assert dims[0] == 0  # selected rows

    def test_selected_fraction_tall_matrix(self):
        """Tall matrix (m > n): should select columns (dim=1)."""
        # 100 x 50 -> m=100 > n=50, select_dim=1, select 25% of 50
        p = nn.Parameter(torch.randn(100, 50))
        opt = Dion2([p], lr=0.02, alpha=0.25, selection="top_l1", mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()

        counts = opt.get_selected_counts()
        assert counts[0] == round(50 * 0.25)
        dims = opt.get_select_dims()
        assert dims[0] == 1  # selected columns

    def test_selected_fraction_random(self):
        """Random selection should also select alpha fraction."""
        p = nn.Parameter(torch.randn(50, 100))
        opt = Dion2([p], lr=0.02, alpha=0.5, selection="random", mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()

        counts = opt.get_selected_counts()
        assert counts[0] == 25  # round(50 * 0.5)

    def test_selective_decay(self):
        """Only selected slices of M should be decayed."""
        p = nn.Parameter(torch.randn(32, 64))  # wide: selects rows
        opt = Dion2([p], lr=0.02, alpha=0.5, selection="top_l1", mu=0.5)

        # First step: accumulate gradient
        p.grad = torch.ones_like(p)
        opt.step()

        M = opt.state[p]["M"]
        # After selective decay with mu=0.5, selected rows should be scaled
        # Unselected rows should still have gradient accumulated
        row_norms = M.norm(dim=1)
        assert row_norms.std().item() > 0, "Row norms should vary (selective decay)"

    def test_weight_decay_applies_to_all(self):
        """Weight decay should affect the entire parameter tensor."""
        p = nn.Parameter(torch.ones(32, 64))  # wide: selects rows
        opt = Dion2([p], lr=0.02, alpha=0.25, mu=0.95, weight_decay=0.5)

        p.grad = torch.zeros_like(p)  # zero grad so only weight decay acts
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()

        # Weight decay should reduce the entire tensor norm
        assert norm_after < norm_before, "Weight decay should shrink all params"

        # ALL elements should be reduced (not just selected rows)
        # Since grad is zero and alpha is small, the NS update on zero momentum
        # doesn't contribute much. The weight decay ratio should apply uniformly.
        ratio = p.data / torch.ones(32, 64)
        # All elements should be approximately the same (all decayed equally)
        assert ratio.std().item() < 0.1, "Weight decay should be uniform across all elements"

    def test_sparsity(self):
        """Sparsity should be 1 - alpha (for wide/square matrices)."""
        p = nn.Parameter(torch.randn(64, 128))  # wide: selects along dim=0 (64 rows)
        opt = Dion2([p], lr=0.02, alpha=0.25, mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()
        assert abs(opt.get_sparsity() - 0.75) < 0.05

    def test_alpha_1_updates_all(self):
        """With alpha=1.0, all slices should be selected."""
        p = nn.Parameter(torch.randn(32, 64))  # wide: selects 32 rows
        opt = Dion2([p], lr=0.02, alpha=1.0, mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()
        assert opt.get_selected_counts()[0] == 32

    def test_different_alphas(self):
        for alpha in [0.1, 0.25, 0.5, 0.75, 1.0]:
            p = nn.Parameter(torch.randn(40, 80))
            opt = Dion2([p], lr=0.02, alpha=alpha, mu=0.95)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with alpha={alpha}"

    def test_both_selection_methods(self):
        for sel in ["top_l1", "random"]:
            p = nn.Parameter(torch.randn(32, 64))
            opt = Dion2([p], lr=0.02, alpha=0.25, selection=sel, mu=0.95)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with selection={sel}"

    def test_square_matrix_selects_rows(self):
        """Square matrix (m == n) should select rows (dim=0)."""
        p = nn.Parameter(torch.randn(64, 64))
        opt = Dion2([p], lr=0.02, alpha=0.25, mu=0.95)
        p.grad = torch.randn_like(p)
        opt.step()
        assert opt.get_select_dims()[0] == 0
        assert opt.get_selected_counts()[0] == 16  # round(64 * 0.25)

    def test_uses_dion2_ns_coefficients(self):
        """Verify Dion2 uses the microsoft/dion NS coefficients, not Muon's."""
        from ada_dion.optim.ortho import DION2_NS_COEFFS
        # First coeff should be the microsoft/dion one, not (3.4445, ...)
        assert DION2_NS_COEFFS[0][0] == pytest.approx(4.0848)


class TestDion2Convergence:
    def test_loss_decreases(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 64))
        opt = Dion2([p], lr=0.01, alpha=0.5, mu=0.9, ns_steps=5)

        losses = []
        for _ in range(100):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"

    def test_random_and_l1_converge_similarly(self):
        """Both selection methods should converge (may differ in speed)."""
        results = {}

        for sel in ["top_l1", "random"]:
            torch.manual_seed(42)
            p = nn.Parameter(torch.randn(32, 64))
            opt = Dion2([p], lr=0.01, alpha=0.5, selection=sel, mu=0.9)

            initial_loss = None
            for _ in range(100):
                opt.zero_grad()
                loss = (p ** 2).sum()
                loss.backward()
                opt.step()
                if initial_loss is None:
                    initial_loss = loss.item()

            results[sel] = (initial_loss, loss.item())

        # Both should have reduced loss relative to their starting point
        for sel, (init, final) in results.items():
            assert final < init, f"Selection {sel} didn't converge: {init} -> {final}"
