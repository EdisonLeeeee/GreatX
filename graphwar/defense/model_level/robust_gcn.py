import torch
import torch.nn as nn

from graphwar.config import Config
from graphwar.nn import RobustConv, activations, Sequential
from graphwar.utils import wrapper

_EDGE_WEIGHT = Config.edge_weight


class RobustGCN(nn.Module):
    """Robust Graph Convolution Network (GCN). 

    Example
    -------
    # RobustGCN with one hidden layer
    >>> model = RobustGCN(100, 10)
    # RobustGCN with two hidden layers
    >>> model = RobustGCN(100, 10, hids=[32, 16], acts=['relu', 'elu'])
    # RobustGCN with two hidden layers, without activation at the first layer
    >>> model = RobustGCN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    Note
    ----
    please make sure `hids` and `acts` are both `list` or `tuple` and
    `len(hids)==len(acts)`.

    """

    @wrapper
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = True,
                 bn: bool = False,
                 gamma: float = 1.0):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False            
        gamma : float, optional
            the attention weight, by default 1.0
        """

        super().__init__()

        assert len(hids) > 0
        conv1 = []
        conv1.append(RobustConv(in_features,
                                hids[0],
                                bias=bias,
                                activation=activations.get(acts[0])))

        if bn:
            conv1.append(nn.BatchNorm1d(hids[0]))
        self.conv1 = Sequential(*conv1)

        conv2 = nn.ModuleList()
        in_features = hids[0]

        for hid, act in zip(hids[1:], acts[1:]):
            conv2.append(RobustConv(in_features,
                                    hid,
                                    bias=bias,
                                    gamma=gamma,
                                    activation=activations.get(act)))
            if bn:
                conv2.append(nn.BatchNorm1d(hid))

            in_features = hid
        if bn:
            conv2.append(nn.BatchNorm1d(in_features))
        conv2.append(RobustConv(in_features, out_features, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for conv in self.conv1:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

        for conv in self.conv2:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

    def forward(self, g, feat, edge_weight=None):
        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)

        feat = self.dropout(feat)
        mean, var = self.conv1(g, feat, edge_weight=edge_weight)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var), edge_weight=edge_weight)

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z
