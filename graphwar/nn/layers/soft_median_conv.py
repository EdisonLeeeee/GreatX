from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops, sort_edge_index
from torch_sparse import SparseTensor

try:
    from glcore import dimmedian_idx
except (ModuleNotFoundError, ImportError):
    dimmedian_idx = None


class SoftMedianConv(nn.Module):
    r"""The graph convolutional operator with soft 
    median aggregation from the `"Robustness of Graph Neural Networks 
    at Scale" <https://arxiv.org/abs/2110.14038>`_ paper (NeurIPS'21)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    cached : bool, optional
        whether the layer will cache
        the computation of :math:`(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2})` and sorted edges on first execution, 
        and will use the cached version for further executions, by default False
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default False
    row_normalize : bool, optional
        whether to perform row-normalization on the fly, by default True        
    bias : bool, optional
        whether to use bias in the layers, by default True    

    Raises
    ------
    RuntimeWarning
        if module `"glcore" <https://github.com/EdisonLeeeee/glcore>`_ is not properly installed.

    Note
    ----
    * The input edges must be sorted for :meth:`dimmedian_idx` from :class:`glcore`

    The same as :class:`torch_geometric`, 
    for the inputs :obj:`x`, :obj:`edge_index`, and :obj:`edge_weight`,
    our implementation supports:

    * :obj:`edge_index` is :class:`torch.LongTensor`: edge indices with shape :obj:`[2, M]`
    * :obj:`edge_index` is :class:`torch_sparse.SparseTensor`: sparse matrix with sparse shape :obj:`[N, N]`      

    See also
    --------
    :class:`graphwar.nn.models.SoftMedianGCN`   
    """
    _cached_edges: Optional[Tuple[Tensor, Tensor]] = None

    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = False, row_normalize: bool = True,
                 bias: bool = True):

        super().__init__()

        if dimmedian_idx is None:
            raise RuntimeWarning("Module 'glcore' is not properly installed, please refer to "
                                 "'https://github.com/EdisonLeeeee/glcore' for more information.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.row_normalize = row_normalize

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

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._cached_edges = None
        return self

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        x = self.lin(x)

        if self._cached_edges is not None:
            edge_index, edge_weight = self._cached_edges
        else:
            # NOTE: we do not support Dense adjacency matrix here
            if isinstance(edge_index, SparseTensor):
                row, col, edge_weight = edge_index.coo()
                edge_index = torch.stack([row, col], dim=0)

            if self.add_self_loops:
                edge_index, edge_weight = add_self_loops(
                    edge_index, num_nodes=x.size(0))

            if self.normalize:
                edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(0),
                                                   improved=False,
                                                   add_self_loops=False, dtype=x.dtype)

            if edge_weight is None:
                edge_weight = x.new_ones(edge_index.size(1))

            edge_index, edge_weight = sort_edge_index(edge_index, edge_weight)

            # cache edges
            if self.cached:
                self._cached_edges = edge_index, edge_weight

        x = soft_median_reduce(x, edge_index, edge_weight)

        # Normalization and calculation of new embeddings
        if self.row_normalize:
            row_sum = edge_weight.new_zeros(x.size(0))
            row_sum.scatter_add_(0, edge_index[0], edge_weight)
            x = row_sum.view(-1, 1) * x

        if self.bias is not None:
            x = x + self.bias

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


def soft_median_reduce(x: Tensor, edge_index: Tensor,
                       edge_weight: Tensor) -> Tensor:
    """weighted dimension-wise Median aggregation"""

    assert edge_weight is not None
    row, col = edge_index
    N, D = x.size()
    median_idx = dimmedian_idx(x, row, col, edge_weight, N)
    col_idx = torch.arange(D, device=row.device).view(1, -1).expand(N, D)
    x = x[median_idx, col_idx]
    return x
