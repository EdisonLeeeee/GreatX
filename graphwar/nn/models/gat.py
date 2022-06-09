import torch.nn as nn
from torch_geometric.nn import GATConv

from graphwar.nn.layers import Sequential, activations
from graphwar.utils import wrapper


class GAT(nn.Module):
    r"""Graph Attention Networks (GAT) from the 
    `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper (ICLR'19)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [8]
    num_heads : list, optional
        the number of attention heads for each hidden layer, by default [8]        
    acts : list, optional
        the activation function for each hidden layer, by default ['relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.6
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False         

    Note
    ----
    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # GAT with one hidden layer
    >>> model = GAT(100, 10)

    >>> # GAT with two hidden layers
    >>> model = GAT(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    >>> # GAT with two hidden layers, without activation at the first layer
    >>> model = GAT(100, 10, hids=[32, 16], acts=[None, 'relu'])

    >>> # GAT with very deep architectures, each layer has elu as activation function
    >>> model = GAT(100, 10, hids=[16]*8, acts=['elu'])

    References
    ----------
    * Paper: https://arxiv.org/abs/1710.10903
    * Author's code: https://github.com/PetarV-/GAT
    * Pytorch implementation: https://github.com/Diego999/pyGAT    

    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [8],
                 num_heads: list = [8],
                 acts: list = ['elu'],
                 dropout: float = 0.6,
                 bias: bool = True,
                 bn: bool = False,
                 includes=['num_heads']):
        super().__init__()
        head = 1
        conv = []
        for hid, num_head, act in zip(hids, num_heads, acts):
            conv.append(GATConv(in_channels * head,
                                hid,
                                heads=num_head,
                                bias=bias,
                                dropout=dropout))
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_channels = hid
            head = num_head

        conv.append(GATConv(in_channels * head,
                            out_channels,
                            heads=1,
                            bias=bias,
                            concat=False,
                            dropout=dropout))

        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)
