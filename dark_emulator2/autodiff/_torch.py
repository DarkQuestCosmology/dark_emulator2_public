import torch
import array_api_compat.torch as xp

from .. import reg_lin_pk as rlp
from ..network import load_net
from ._core import _Emulator


class TorchEmulator(_Emulator):
    """
    Differentiable Dark Emulator 2 for PyTorch (e.g. Pyro).

    Outputs are torch tensors connected to the autograd graph of the input parameters.

    Parameters
    ----------
    de : DarkEmulator2, optional
        Emulator whose settings and tables are used (created if None).
    device : str or torch.device, optional
        "cpu" (default), "cuda" or "cuda:N".
    dtype : torch.dtype, optional
        Floating point type (default: torch.get_default_dtype()).
    """

    def __init__(self, de=None, device=None, dtype=None):
        super().__init__(de)
        self.xp = xp
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self._amp, self._lin, self._ratio, self._nlin = (self._load(path) for path in (
            rlp.def_amp_torch, rlp.def_lin_pk_cb_torch, rlp.def_lin_ratio_torch, self.de.pk_emu.input_file))

    def _load(self, path):
        return load_net(path)[0].to(self.device, self.dtype).eval().seq_fc

    def _asarray(self, a):
        return torch.as_tensor(a, dtype=self.dtype, device=self.device)

    _param = _asarray
