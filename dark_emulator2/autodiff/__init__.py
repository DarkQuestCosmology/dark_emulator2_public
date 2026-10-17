from ._torch import TorchEmulator

__all__ = ["TorchEmulator", "JaxEmulator"]


def __getattr__(name):
    # import JAX only when it is used
    if name == "JaxEmulator":
        from ._jax import JaxEmulator
        return JaxEmulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
