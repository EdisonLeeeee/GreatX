import torch.nn as nn

from graphwar.utils import wrapper
from graphwar.nn.layers import SSGConv, Sequential, activations


class SSGC(nn.Module):
    """Simple Spectra Graph Convolution Network from paper `Simple Spectral Graph Convolution 
     <https://openreview.net/forum?id=CYO5T-YjWZV>`__.

    Example
    -------
    # SSGC model without hidden layers (by default)
    >>> model = SSGC(100, 10)
    # SSGC with one hidden layers
    >>> model = SSGC(100, 10, hids=[16], acts=['relu'])    
    """

    @wrapper
    def __init__(self,
                 in_channels,
                 out_channels,
                 hids: list = [],
                 acts: list = [],
                 dropout: float = 0.,
                 K: int = 5,
                 alpha: float = 0.1,
                 bn: bool = False,
                 bias: bool = True,
                 cached: bool = True):
        super().__init__()

        conv = []
        for i, (hid, act) in enumerate(zip(hids, acts)):
            if i == 0:
                conv.append(SSGConv(in_channels,
                                    hid,
                                    bias=bias,
                                    K=K,
                                    alpha=alpha,
                                    cached=cached))
            else:
                conv.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid

        if not hids:
            conv.append(SSGConv(in_channels,
                                out_channels,
                                bias=bias,
                                K=K,
                                alpha=alpha,
                                cached=cached))
        else:
            conv.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def cache_clear(self):
        for layer in self.conv:
            if hasattr(layer, 'cache_clear'):
                layer.cache_clear()
        return self

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
