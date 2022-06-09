from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from graphwar import is_edge_index
from graphwar.functional import spmm
from graphwar.nn.layers.gcn_conv import dense_gcn_norm


class SSGConv(nn.Module):
    r"""The simple spectral graph convolutional operator from 
    the `"Simple Spectral Graph Convolution"
    <https://openreview.net/forum?id=CYO5T-YjWZV>`_ paper (ICLR'21)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    K : int
        the number of propagation steps, by default 5     
    alpha : float
        Teleport probability :math:`\alpha`, by default 0.1           
    cached : bool, optional 
        whether the layer will cache
        the K-step aggregation on first execution, and will use the
        cached version for further executions, by default False
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
    for the inputs :obj:`x`, :obj:`edge_index`, and :obj:`edge_weight`,
    our implementation supports:

    * :obj:`edge_index` is :class:`torch.FloatTensor`: dense adjacency matrix with shape :obj:`[N, N]`
    * :obj:`edge_index` is :class:`torch.LongTensor`: edge indices with shape :obj:`[2, M]`
    * :obj:`edge_index` is :class:`torch_sparse.SparseTensor`: sparse matrix with sparse shape :obj:`[N, N]`   

    See also
    --------
    :class:`graphwar.nn.models.SSGC`   
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int,
                 K: int = 5,
                 alpha: float = 0.1,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cache_clear()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._cached_x = None
        return self

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        is_edge_like = is_edge_index(edge_index)

        if cache is None:
            if self.normalize:
                if is_edge_like:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                elif isinstance(edge_index, SparseTensor):
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                else:
                    # N by N dense adjacency matrix
                    edge_index = dense_gcn_norm(
                        edge_index, add_self_loops=self.add_self_loops)

            x_out = x * self.alpha
            for k in range(self.K):
                if is_edge_like:
                    x = spmm(x, edge_index, edge_weight)
                else:
                    x = edge_index @ x
                x_out = x_out + (1 - self.alpha)/self.K * x

            if self.cached:
                self._cached_x = x_out
        else:
            x_out = cache.detach()

        return self.lin(x_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, alpha={self.alpha})')
