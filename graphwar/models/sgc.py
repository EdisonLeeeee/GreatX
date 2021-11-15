import torch.nn as nn
from dgl import DGLError

from graphwar.nn import SGConv


class SGC(nn.Module):
    """Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    Example
    -------
    # SGC model
    >>> model = SGC(100, 10)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 k=2,
                 bias=True,
                 cached=True):
        super().__init__()

        conv = SGConv(in_features,
                      out_features,
                      bias=bias,
                      k=k,
                      cached=cached)
        self.conv = conv

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, g, feat):
        return self.conv(g, feat)

    def cache_clear(self):
        self.conv._cached_h = None
        return self
