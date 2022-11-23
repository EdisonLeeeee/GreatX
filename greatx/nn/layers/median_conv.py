import torch
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

from greatx.functional import spmm


class MedianConv(nn.Module):
    r"""The graph convolutional operator with median aggregation
    from the `"Understanding Structural Vulnerability
    in Graph Convolutional Networks"
    <https://www.ijcai.org/proceedings/2021/310>`_ paper (IJCAI'21)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    reduce : str
        aggregation function, including {'median', 'sample_median'},
        where :obj:`median` uses the exact median as the aggregation function,
        while :obj:`sample_median` appropriates the median with a fixed set
        of sampled nodes. :obj:`sample_median` is much faster and
        more scalable than :obj:`median`. By default, :obj:`median` is used.
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True
    bias : bool, optional
        whether to use bias in the layers, by default True

    See also
    --------
    :class:`greatx.nn.models.supervised.MedianGCN`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 reduce: str = 'median', add_self_loops: bool = True,
                 normalize: bool = False, bias: bool = True):

        super().__init__()

        assert reduce in ('median', 'sample_median')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.reduce = reduce

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        x = self.lin(x)

        # NOTE: we do not support Dense adjacency matrix here
        if isinstance(edge_index, SparseTensor):
            row, col, edge_weight = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)

        if self.add_self_loops:
            edge_index, edge_weight = add_self_loops(edge_index,
                                                     num_nodes=x.size(0))

        if self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight,
                                               x.size(0), improved=False,
                                               add_self_loops=False,
                                               dtype=x.dtype)

        out = spmm(x, edge_index, edge_weight, reduce=self.reduce)

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
