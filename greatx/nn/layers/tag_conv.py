import torch
from torch import Tensor, nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


class TAGConv(nn.Module):
    r"""The topological adaptive graph convolutional operator from
    the `"Topological Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper (arXiv'17)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    K : int
        the number of propagation steps, by default 2
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True
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
    :class:`greatx.nn.models.supervised.TAGCN`

    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 2,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels * (K + 1), out_channels, bias=bias,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.add_self_loops:
            edge_index, edge_weight = make_self_loops(edge_index, edge_weight,
                                                      num_nodes=x.size(0))

        if self.normalize:
            edge_index, edge_weight = make_gcn_norm(edge_index, edge_weight,
                                                    num_nodes=x.size(0),
                                                    dtype=x.dtype,
                                                    add_self_loops=False)

        xs = [x]
        for k in range(self.K):
            x = spmm(x, edge_index, edge_weight)
            xs.append(x)

        xs = torch.cat(xs, dim=-1)

        return self.lin(xs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
