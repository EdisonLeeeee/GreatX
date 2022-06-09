from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch_sparse import SparseTensor

from graphwar.functional import spmm
from graphwar import is_edge_index
from graphwar.nn.layers.gcn_conv import dense_gcn_norm


class RobustConv(nn.Module):
    r"""The robust graph convolutional operator
    from the `"Robust Graph Convolutional Networks 
    Against Adversarial Attacks"
    <http://pengcui.thumedialab.com/papers/RGCN.pdf>`_ paper (KDD'19)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    gamma : float, optional
        the scale of attention on the variances, by default 1.0
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    bias : bool, optional
        whether to use bias in the layers, by default True     

    Note
    ----
    Different from that in :class:`torch_geometric`, 
    For the inputs :obj:`x`, :obj:`edge_index`, and :obj:`edge_weight`,
    our implementation supports:

    * :obj:`edge_index` is :class:`torch.FloatTensor`: dense adjacency matrix with shape :obj:`[N, N]`
    * :obj:`edge_index` is :class:`torch.LongTensor`: edge indices with shape :obj:`[2, M]`
    * :obj:`edge_index` is :class:`torch_sparse.SparseTensor`: sparse matrix with sparse shape :obj:`[N, N]`           

    See also
    --------
    :class:`graphwar.nn.models.RobustGCN`       
    """

    def __init__(self, in_channels: int, out_channels: int,
                 gamma: float = 1.0,
                 add_self_loops: bool = True,
                 bias: bool = True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.gamma = gamma

        self.lin_mean = Linear(in_channels, out_channels, bias=False,
                               weight_initializer='glorot')

        self.lin_var = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')
        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_channels))
            self.bias_var = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_mean.reset_parameters()
        self.lin_var.reset_parameters()
        zeros(self.bias_mean)
        zeros(self.bias_var)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x_mean = x_var = x
        else:
            x_mean, x_var = x

        mean = self.lin_mean(x_mean)
        var = self.lin_var(x_var)

        if self.bias_mean is not None:
            mean = mean + self.bias_mean
            var = var + self.bias_var

        mean = F.relu(mean)
        var = F.relu(var)

        is_edge_like = is_edge_index(edge_index)

        if is_edge_like:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, mean.size(0),
                                               improved=False,
                                               add_self_loops=self.add_self_loops,
                                               dtype=mean.dtype)
        elif isinstance(edge_index, SparseTensor):
            adj = gcn_norm(edge_index, mean.size(0),
                           improved=False,
                           add_self_loops=self.add_self_loops, dtype=mean.dtype)

        else:
            # N by N dense adjacency matrix
            adj = dense_gcn_norm(edge_index, improved=False,
                                 add_self_loops=self.add_self_loops)

        attention = torch.exp(-self.gamma * var)
        mean = mean * attention
        var = var * attention * attention

        # TODO: actually, using .square() is not always right,
        # particularly weighted graph
        if is_edge_like:
            mean = spmm(mean, edge_index, edge_weight)
            var = spmm(var, edge_index, edge_weight.square())
        else:
            mean = adj @ mean
            var = (adj * adj) @ var

        return mean, var

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
