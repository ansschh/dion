from .param_grouper import group_params_for_hybrid


def __getattr__(name):
    """Lazy import for HybridOptimizersContainer (requires TorchTitan)."""
    if name == "HybridOptimizersContainer":
        from .hybrid_optimizer import HybridOptimizersContainer
        return HybridOptimizersContainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HybridOptimizersContainer",
    "group_params_for_hybrid",
]
