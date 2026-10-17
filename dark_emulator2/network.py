import argparse
from collections import OrderedDict

import torch


class FCNet(torch.nn.Module):
    """
    Fully connected network of the emulators.

    Parameters
    ----------
    layers : list
        ``[name, kind, *args]`` for each layer of ``seq_fc``, where ``kind`` is a ``torch.nn`` class
        (Linear, GELU, ELU, Dropout) and ``args`` its positional arguments.
    list_input : bool, optional
        The input is a list ``[npart, boxsize, z, ...]``; the first three entries are stacked
        and concatenated with the others (non-linear P(k) networks).
    """

    def __init__(self, layers, list_input=False):
        super().__init__()
        self.seq_fc = torch.nn.Sequential(OrderedDict((name, getattr(torch.nn, kind)(*args)) for name, kind, *args in layers))
        self.list_input = list_input

    def forward(self, x):
        if self.list_input:
            x = torch.cat([torch.stack(x[:3], 1), torch.cat(x[3:], 1)], 1)
        return self.seq_fc(x)


def load_net(path):
    """
    Load a network file saved as ``{"layers", "list_input", "state_dict", "netargs"}``.

    Returns
    -------
    net : FCNet
        Network on the CPU, without gradients for the weights.
    netargs : argparse.Namespace
        Arguments of the network training.
    """

    data = torch.load(path, map_location="cpu", weights_only=True)
    net = FCNet(data["layers"], data["list_input"])
    net.load_state_dict(data["state_dict"])
    return net.requires_grad_(False), argparse.Namespace(**data["netargs"])
