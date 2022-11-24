from typing import List

import torch
import torch.nn as nn

from greatx.nn.layers import RobustConv, activations
from greatx.utils import wrapper


class RobustGCN(nn.Module):
    r"""Robust graph convolutional network (RobustGCN)
    from the `"Robust Graph Convolutional Networks
    Against Adversarial Attacks"
    <http://pengcui.thumedialab.com/papers/RGCN.pdf>`_ paper (KDD'19)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : List[int], optional
        the number of hidden units for each hidden layer,
        by default [32]
    acts : List[str], optional
        the activation function for each hidden layer,
        by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    bias : bool, optional
        whether to use bias in the layers, by default True
    gamma : float, optional
        the scale of attention on the variances, by default 1.0
    gamma : float, optional
        the scale of attention on the variances, by default 1.0
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False

    Examples
    --------
    >>> # RobustGCN with one hidden layer
    >>> model = RobustGCN(100, 10)

    >>> # RobustGCN with two hidden layers
    >>> model = RobustGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # RobustGCN with two hidden layers, without first activation
    >>> model = RobustGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # RobustGCN with deep architectures, each layer has elu activation
    >>> model = RobustGCN(100, 10, hids=[16]*8, acts=['elu'])

    See also
    --------
    :class:`greatx.nn.layers.RobustConv`

    """
    @wrapper
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hids: List[int] = [32],
        acts: List[str] = ['relu'],
        dropout: float = 0.5,
        bias: bool = True,
        gamma: float = 1.0,
        kl: float = 5e-4,
        bn: bool = False,
    ):

        super().__init__()

        assert len(hids) > 0
        self.conv1 = RobustConv(in_channels, hids[0], bias=bias)
        self.act1 = activations.get(acts[0])

        conv2 = nn.ModuleList()

        in_channels = hids[0]
        for hid, act in zip(hids[1:], acts[1:]):
            conv2.append(RobustConv(in_channels, hid, bias=bias, gamma=gamma))
            if bn:
                conv2.append(nn.BatchNorm1d(hid))
            conv2.append(activations.get(act))
            in_channels = hid

        conv2.append(
            RobustConv(in_channels, out_channels, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.kl = kl
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()
        self.cache_clear()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self.mean = self.var = None
        return self

    def forward(self, x, edge_index, edge_weight=None):
        """"""

        x = self.dropout(x)
        mean, var = self.conv1(x, edge_index, edge_weight)
        mean, var = self.act1(mean), self.act1(var)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            if isinstance(conv, RobustConv):
                mean, var = self.dropout(mean), self.dropout(var)
                mean, var = conv((mean, var), edge_index, edge_weight)
            else:
                mean, var = conv(mean), conv(var)

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def loss(self, *args):
        mean, var = self.mean, self.var
        kl_loss = -0.5 * torch.sum(
            torch.mean(1 + torch.log(var + 1e-8) - mean.pow(2) + var, dim=1))
        return self.kl * kl_loss
