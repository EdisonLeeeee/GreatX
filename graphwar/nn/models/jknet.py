import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphwar.nn.layers import GCNConv, Sequential, activations
from graphwar.nn.layers.gcn_conv import dense_gcn_norm
from graphwar.utils import wrapper
from graphwar.functional import spmm
from graphwar import is_edge_index


class JKNet(nn.Module):
    r"""Implementation of Graph Convolution Network with 
    Jumping knowledge (JKNet) from
    the `"Representation Learning on Graphs with 
    Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper (ICML'18)

    Parameters
    ----------
    in_channels : int, 
        the input dimensions of model
    out_channels : int, 
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer, by default [16, 16, 16]
    acts : list, optional
        the activation function for each hidden layer, by default ['relu', 'relu', 'relu']
    dropout : float, optional
        the dropout ratio of model, by default 0.5
    mode : str, optional
        the mode of jumping knowledge, including 'cat', 'lstm', and 'max',          
    bias : bool, optional
        whether to use bias in the layers, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer, by default False              

    Note
    ----
    To accept a different graph as inputs, please call :meth:`cache_clear` first
    to clear cached results.

    It is convenient to extend the number of layers with different or the same
    hidden units (activation functions) using :meth:`graphwar.utils.wrapper`. 

    See Examples below:

    Examples
    --------
    >>> # JKNet with five hidden layers
    >>> model = JKNet(100, 10, hids=[16]*5)



    """

    @wrapper
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hids: list = [16]*3,
                 acts: list = ['relu']*3,
                 dropout: float = 0.5,
                 mode: str = 'cat',
                 bn: bool = False,
                 bias: bool = True):

        super().__init__()
        self.mode = mode
        num_JK_layers = len(list(hids)) - 1  # number of JK layers

        assert num_JK_layers >= 1 and len(set(
            hids)) == 1, 'the number of hidden layers should be greater than 2 and the hidden units must be equal'

        conv = []
        assert len(hids) == len(acts)
        for hid, act in zip(hids, acts):
            block = []
            block.append(nn.Dropout(dropout))
            block.append(GCNConv(in_channels,
                                 hid,
                                 bias=bias))
            if bn:
                block.append(nn.BatchNorm1d(hid))
            block.append(activations.get(act))
            conv.append(Sequential(*block))
            in_channels = hid

        # `loc=1` specifies the location of features.
        self.conv = Sequential(*conv)

        assert len(conv) == num_JK_layers + 1

        if self.mode == 'lstm':
            self.jump = JumpingKnowledge(mode, hid, num_JK_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == 'cat':
            hid = hid * (num_JK_layers + 1)

        self.mlp = nn.Linear(hid, out_channels, bias=bias)

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.attn.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for conv in self.conv:
            x = conv(x, edge_index, edge_weight)
            xs.append(x)

        x = self.jump(xs)

        is_edge_like = is_edge_index(edge_index)

        if is_edge_like:
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(0), False,
                add_self_loops=True, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(0), False,
                add_self_loops=True, dtype=x.dtype)
        else:
            # N by N dense adjacency matrix
            adj = dense_gcn_norm(edge_index, add_self_loops=True)

        if is_edge_like:
            out = spmm(x, edge_index, edge_weight)
        else:
            out = adj @ x

        return self.mlp(out)
