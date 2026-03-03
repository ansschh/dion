"""
Pytest conftest: mock the dion package if triton is unavailable.

On Linux/HPC with CUDA + triton, the real dion package works fine.
On Windows/macOS dev machines, triton is unavailable, so we install a
minimal mock so that `from ada_dion.optim import AdaDion` works.
"""
import sys
import types

import torch


def _install_dion_mock():
    """Install a mock dion module if the real one can't import."""
    try:
        from dion import Muon, Dion, Dion2  # noqa: F401
        return  # Real dion works — no mock needed
    except (ImportError, ModuleNotFoundError):
        pass

    mock = types.ModuleType("dion")
    mock.__path__ = []

    class _MockOptimizer(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            super().__init__(params, kwargs)

        def step(self, closure=None):
            pass

    mock.Muon = type("Muon", (_MockOptimizer,), {})
    mock.Dion = type("Dion", (_MockOptimizer,), {})
    mock.Dion2 = type("Dion2", (_MockOptimizer,), {})
    sys.modules["dion"] = mock

    for sub in ["dion.muon", "dion.dion", "dion.dion2",
                 "dion.newton_schulz_triton", "dion.utils"]:
        sys.modules[sub] = types.ModuleType(sub)


# Run before any test module imports
_install_dion_mock()
