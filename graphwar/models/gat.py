import torch.nn as nn
from dgl.nn.pytorch import GATConv

from graphwar.nn import Sequential, activations
from graphwar.utils import wrapper


class GAT(nn.Module):
    """Graph Attention Network (GAT)

    Example
    -------
    # GAT with one hidden layer
    >>> model = GAT(100, 10)
    # GAT with two hidden layers
    >>> model = GAT(100, 10, hids=[32, 16], num_heads=[8, 8], acts=['relu', 'elu'])
    # GAT with two hidden layers, without activation at the first layer
    >>> model = GAT(100, 10, hids=[32, 16], num_heads=[8, 8], acts=[None, 'elu'])

    References
    ----------
    Paper: https://arxiv.org/abs/1710.10903
    Author's code: https://github.com/PetarV-/GAT
    Pytorch implementation: https://github.com/Diego999/pyGAT    

    """

    @wrapper
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [8],
                 num_heads: list = [8],
                 acts: list = ['elu'],
                 dropout: float = 0.6,
                 bias: bool = True,
                 bn: bool = False,
                 includes=['num_heads']):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [8]
        num_heads : list, optional
            the number of attention heads of each hidden layer, by default [8]
        acts : list, optional
            the activaction function of each hidden layer, by default ['elu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True      
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False            
        """
        super().__init__()

        conv = []
        head = 1

        for hid, num_head, act in zip(hids, num_heads, acts):
            conv.append(GATConv(in_features * head,
                                hid,
                                num_heads=num_head,
                                bias=bias,
                                feat_drop=dropout,
                                attn_drop=dropout,
                                activation=None))
            conv.append(nn.Flatten(1))
            if bn:
                conv.append(nn.BatchNorm1d(hid * num_head))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid
            head = num_head

        conv.append(GATConv(in_features * head, out_features,
                            num_heads=1, bias=bias,
                            feat_drop=dropout,
                            attn_drop=dropout))
        self.conv = Sequential(*conv, loc=1)  # `loc=1` specifies the location of features.

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, g, feat):
        g = g.add_self_loop()
        return self.conv(g, feat).mean(1)
