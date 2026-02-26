"""Tests for orthogonalization utilities."""
import pytest
import torch

from ada_dion.optim.ortho import (
    newton_schulz_zeroth_power,
    col_norm,
    orthonormalize_qr,
    shape_scale,
    MUON_NS_COEFFS,
    DION2_NS_COEFFS,
)


class TestNewtonSchulzZerothPower:
    def test_output_shape(self):
        x = torch.randn(32, 16)
        out = newton_schulz_zeroth_power(x, steps=5)
        assert out.shape == (32, 16)

    def test_output_shape_tall(self):
        """Tall matrix (rows > cols) should auto-transpose and produce same shape."""
        x = torch.randn(16, 32)
        out = newton_schulz_zeroth_power(x, steps=5)
        assert out.shape == (16, 32)

    def test_approximate_orthonormality_wide(self):
        """For wide matrix (m <= n), U @ U^T should approximate I."""
        x = torch.randn(64, 128)
        out = newton_schulz_zeroth_power(x, steps=5)
        orth = out @ out.T  # (64, 64)
        eye = torch.eye(64)
        err = (orth - eye).norm().item() / (64 ** 0.5)
        assert err < 0.5, f"Orthonormality error too large: {err}"

    def test_approximate_orthonormality_tall(self):
        """For tall matrix (m > n), U^T @ U should approximate I."""
        x = torch.randn(128, 64)
        out = newton_schulz_zeroth_power(x, steps=5)
        orth = out.T @ out  # (64, 64)
        eye = torch.eye(64)
        err = (orth - eye).norm().item() / (64 ** 0.5)
        assert err < 0.5, f"Orthonormality error too large: {err}"

    def test_no_nan(self):
        x = torch.randn(64, 32)
        out = newton_schulz_zeroth_power(x, steps=5)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D"):
            newton_schulz_zeroth_power(torch.randn(32))

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="2D"):
            newton_schulz_zeroth_power(torch.randn(4, 8, 16))

    def test_different_step_counts(self):
        x = torch.randn(16, 16)
        out1 = newton_schulz_zeroth_power(x, steps=1)
        out5 = newton_schulz_zeroth_power(x, steps=5)
        # More steps should give better orthonormality
        err1 = (out1 @ out1.T - torch.eye(16)).norm().item()
        err5 = (out5 @ out5.T - torch.eye(16)).norm().item()
        assert err5 <= err1 + 0.1  # 5 steps should be at least as good

    def test_muon_coeffs_single_tuple(self):
        """MUON_NS_COEFFS should be a single tuple, used for all iterations."""
        assert len(MUON_NS_COEFFS) == 1
        assert MUON_NS_COEFFS[0] == (3.4445, -4.7750, 2.0315)

    def test_dion2_coeffs_five_tuples(self):
        """DION2_NS_COEFFS should have 5 per-iteration tuples."""
        assert len(DION2_NS_COEFFS) == 5
        assert DION2_NS_COEFFS[0] == (4.0848, -6.8946, 2.9270)

    def test_dion2_coeffs_produce_valid_output(self):
        """NS with Dion2 coefficients should also produce reasonable output."""
        x = torch.randn(64, 128)
        out = newton_schulz_zeroth_power(x, steps=5, coeffs=DION2_NS_COEFFS)
        assert out.shape == (64, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestColNorm:
    def test_columns_unit_norm(self):
        w = torch.randn(10, 5)
        out = col_norm(w)
        norms = out.norm(dim=0)
        assert torch.allclose(norms, torch.ones(5), atol=1e-6)

    def test_shape_preserved(self):
        w = torch.randn(8, 3)
        out = col_norm(w)
        assert out.shape == (8, 3)


class TestOrthonormalizeQR:
    def test_orthonormal_columns(self):
        p = torch.randn(10, 5)
        q = orthonormalize_qr(p)
        # Q^T Q should be identity
        qtq = q.T @ q
        eye = torch.eye(5)
        assert torch.allclose(qtq, eye, atol=1e-5)

    def test_shape(self):
        p = torch.randn(20, 8)
        q = orthonormalize_qr(p)
        assert q.shape == (20, 8)


class TestShapeScale:
    def test_square(self):
        assert shape_scale(100, 100) == pytest.approx(1.0)

    def test_wide(self):
        # max(1, 64/256) = 1 -> sqrt(1) = 1.0
        # Wide matrices should NOT be scaled down (reference behavior)
        assert shape_scale(64, 256) == pytest.approx(1.0)

    def test_tall(self):
        # max(1, 256/64) = 4 -> sqrt(4) = 2.0
        assert shape_scale(256, 64) == pytest.approx(2.0)
