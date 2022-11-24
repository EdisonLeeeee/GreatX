from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


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
    normalize : bool, optional
        whether to normalize the input graph, by default True
    bias : bool, optional
        whether to use bias in the layers, by default True

    Note
    ----
    Different from that in :class:`torch_geometric`,
    for the input :obj:`edge_index`, our implementation supports
    :obj:`torch.FloatTensor`, :obj:`torch.LongTensor`
    and obj:`torch_sparse.SparseTensor`.

    See also
    --------
    :class:`greatx.nn.models.supervised.RobustGCN`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: float = 1.0,
        normalize: bool = True,
        add_self_loops: bool = True,
        bias: bool = True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.gamma = gamma
        self.normalize = normalize

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
        """"""

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

        if self.add_self_loops:
            edge_index, edge_weight = make_self_loops(edge_index, edge_weight,
                                                      num_nodes=mean.size(0))

        if self.normalize:
            edge_index, edge_weight = make_gcn_norm(edge_index, edge_weight,
                                                    num_nodes=mean.size(0),
                                                    dtype=mean.dtype,
                                                    add_self_loops=False)

        attention = torch.exp(-self.gamma * var)
        mean = mean * attention
        var = var * attention * attention

        # TODO: actually, using .square() is not always right,
        # particularly weighted graph
        if edge_weight is not None:
            mean = spmm(mean, edge_index, edge_weight)
            var = spmm(var, edge_index, edge_weight.square())
        else:
            # N by N adjacency matrix (sparse or dense)
            mean = edge_index @ mean
            var = (edge_index * edge_index) @ var

        return mean, var

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
