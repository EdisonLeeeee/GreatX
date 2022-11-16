from typing import Optional

from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import dense_gcn_norm
from greatx.utils.check import is_edge_index


class DGConv(nn.Module):
    r"""The decoupled graph convolutional operator from
    the `"Dissecting the Diffusion Process in Linear Graph
    Convolutional Networks"
    <https://arxiv.org/abs/2102.10739>`_ paper (NeurIPS'21)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    K : int
        the number of propagation steps, by default 2
    t : float
        Terminal time :math:`t`, by default 5.27
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
    for the input :obj:`edge_index`, our implementation supports
    :obj:`torch.FloatTensor`, :obj:`torch.LongTensor`
    and obj:`torch_sparse.SparseTensor`.

    See also
    --------
    :class:`~greatx.nn.models.supervised.DGC`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, t: float = 5.27,
                 K: int = 2, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.K = K
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
        """"""

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

            delta = self.t / self.K
            for k in range(self.K):
                if is_edge_like:
                    out = spmm(x, edge_index, edge_weight)
                else:
                    out = edge_index @ x
                x = (1 - delta) * x + delta * out

            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, t={self.t})')
