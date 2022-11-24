from typing import List

import torch.nn as nn
from torch_geometric.nn import APPNP as APPNPConv

from greatx.nn.layers import Sequential, activations
from greatx.utils import wrapper


class APPNP(nn.Module):
    r"""Implementation of Approximated personalized
    propagation of neural predictions (APPNP) from
    the `"Predict then Propagate: Graph Neural
    Networks meet Personalized PageRank"
    <https://arxiv.org/abs/1810.05997>`_ paper (ICLR'19)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer, by default [16]
    acts : List[str], optional
        the activation function for each hidden layer, by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.8
    K : int, optional
        the number of propagation steps, by default 10
    alpha : float
        Teleport probability :math:`\alpha`, by default 0.1
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False
    cached : bool, optional
        whether the layer will cache the computation of propagation
        on first execution, and will use the
        cached version for further executions, by default False

    Note
    ----
    To accept a different graph as inputs, please call
    :meth:`cache_clear` first to clear cached results.

    Examples
    --------
    >>> # APPNP without hidden layer
    >>> model = APPNP(100, 10)

    >>> # APPNP with two hidden layers
    >>> model = APPNP(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # APPNP with two hidden layers, without first activation
    >>> model = APPNP(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # APPNP with deep architectures, each layer has elu activation
    >>> model = APPNP(100, 10, hids=[16]*8, acts=['elu'])

    """
    @wrapper
    def __init__(self, in_channels: int, out_channels: int,
                 hids: List[int] = [16], acts: List[str] = ['relu'],
                 dropout: float = 0.8, K: int = 10, alpha: float = 0.1,
                 bn: bool = False, bias: bool = True, cached: bool = False):

        super().__init__()
        assert len(hids) > 0

        lin = []
        for hid, act in zip(hids, acts):
            lin.append(nn.Dropout(dropout))
            lin.append(nn.Linear(in_channels, hid, bias=bias))
            if bn:
                lin.append(nn.BatchNorm1d(hid))
            in_channels = hid
            lin.append(activations.get(act))

        lin.append(nn.Dropout(dropout))
        lin.append(nn.Linear(in_channels, out_channels, bias=bias))

        self.prop = APPNPConv(K, alpha, cached=cached)
        self.lin = Sequential(*lin)

    def reset_parameters(self):
        self.prop.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = self.lin(x)
        return self.prop(x, edge_index, edge_weight)
