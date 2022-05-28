import torch
import torch.nn as nn

from graphwar.nn.layers import RobustConv, activations, Sequential
from graphwar.utils import wrapper


class RobustGCN(nn.Module):
    """Robust Graph Convolution Network (RobustGCN). 

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
                 in_feats: int,
                 out_feats: int,
                 hids: list = [32],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bn: bool = False,
                 bias: bool = True,
                 gamma: float = 1.0):
        r"""
        Parameters
        ----------
        in_feats : int, 
            the input dimmensions of model
        out_feats : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [32]
        acts : list, optional
            the activation function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        gamma : float, optional
            the attention weight, by default 1.0
        """

        super().__init__()

        assert len(hids) > 0
        self.conv1 = RobustConv(in_feats,
                                hids[0],
                                bias=bias)
        self.act1 = activations.get(acts[0])

        conv2 = nn.ModuleList()

        in_feats = hids[0]
        for hid, act in zip(hids[1:], acts[1:]):
            conv2.append(RobustConv(in_feats,
                                    hid,
                                    bias=bias,
                                    gamma=gamma))
            if bn:
                conv.append(nn.BatchNorm1d(hid))            
            conv2.append(activations.get(act))
            in_feats = hid

        conv2.append(RobustConv(in_feats, out_feats, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()
        self.cache_clear()
        
    def cache_clear(self):
        self.mean = self.var = None
        return self

    def forward(self, x, edge_index, edge_weight=None):

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
