import jax
import jax.numpy as jnp
import torch

from .. import reg_lin_pk as rlp
from ._core import _Emulator


def _load_mlp(path, dtype, device):
    # the networks are plain MLPs: rebuild them in JAX from the TorchScript weights
    net = torch.jit.load(path, map_location="cpu")
    sd = net.state_dict()
    layers, act = [], None
    for name, m in net.seq_fc.named_children():
        if m.original_name == "Linear":
            layers.append(jax.device_put((jnp.asarray(sd[f"seq_fc.{name}.weight"].numpy().T, dtype=dtype),
                                          jnp.asarray(sd[f"seq_fc.{name}.bias"].numpy(), dtype=dtype)), device))
        elif m.original_name == "GELU":
            act = lambda x, tanh="tanh" in m.code: jax.nn.gelu(x, approximate=tanh)
        elif m.original_name == "ELU":
            act = jax.nn.elu

    def mlp(x):
        # keep XLA from constant-folding the constant input columns (e.g. GRF = 1) through the weights at compile time
        x = jax.lax.optimization_barrier(x)
        for W, b in layers[:-1]:
            x = act(x @ W + b)
        W, b = layers[-1]
        return x @ W + b
    return mlp


class JaxEmulator(_Emulator):
    """
    Differentiable Dark Emulator 2 for JAX (e.g. NumPyro, BlackJAX).

    Outputs are JAX arrays; all methods can be used inside ``jax.jit``, ``jax.grad`` and ``jax.vmap``
    as long as the grids (zred, klist, llist, ...) are concrete numpy arrays.

    Parameters
    ----------
    de : DarkEmulator2, optional
        Emulator whose settings and tables are used (created if None).
    device : jax.Device, optional
        Device of the weights and tables (default: jax.devices()[0]).
    dtype : dtype, optional
        Floating point type (default: float64 if jax_enable_x64 is set, otherwise float32).
    """

    def __init__(self, de=None, device=None, dtype=None):
        super().__init__(de)
        self.xp = jnp
        self.device = jax.devices()[0] if device is None else device
        self.dtype = jnp.result_type(float) if dtype is None else dtype
        self._amp, self._lin, self._ratio, self._nlin = (_load_mlp(path, self.dtype, self.device) for path in (
            rlp.def_amp_torch, rlp.def_lin_pk_cb_torch, rlp.def_lin_ratio_torch, self.de.pk_emu.input_file))

    def _asarray(self, a):
        # concrete even when first called inside jit, since the tables are cached
        with jax.ensure_compile_time_eval():
            return jax.device_put(jnp.asarray(a, dtype=self.dtype), self.device)

    def _param(self, v):
        return jnp.asarray(v, dtype=self.dtype)
