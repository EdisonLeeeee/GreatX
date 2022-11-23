from typing import Optional

from torch import Tensor, nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from greatx.functional import spmm
from greatx.nn.layers import LIF, PoissonEncoder
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


class SpikingGCNonv(nn.Module):
    r"""The simplified graph convolutional operator from
    the `"Spiking Graph Convolutional Networks"
    <https://arxiv.org/abs/2205.02767>`_ paper (IJCAI'22)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    K : int
        the number of propagation steps, by default 1
    tau : float
        the :math:`\tau` in LIF neuron, by default 1.0
    v_threshold : float
        the threshold :math:`V_{th}` in LIF neuron, by default 1.0
    v_reset : float
        the reset level :math:`V_{reset}` in LIF neuron, by default 0
    cached : bool, optional
        whether the layer will cache
        the computation of :math:`(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2})^K` on first execution, and will use the
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
    :class:`~greatx.nn.models.supervised.SpikingGCN`
    """

    _cached_x: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 1,
        T: int = 20,
        tau: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer='glorot')
        assert T > 0
        self.T = T
        self.encoder = PoissonEncoder()
        self.neuron = LIF(tau=tau, v_threshold=v_threshold, v_reset=v_reset)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cache_clear()
        self.neuron.reset()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._cached_x = None
        return self

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        cache = self._cached_x

        if cache is None:
            if self.add_self_loops:
                edge_index, edge_weight = make_self_loops(
                    edge_index, edge_weight, num_nodes=x.size(0))

            if self.normalize:
                edge_index, edge_weight = make_gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0),
                    dtype=x.dtype, add_self_loops=False)

            for k in range(self.K):
                x = spmm(x, edge_index, edge_weight)

            if self.cached:
                self._cached_x = x
        else:
            x = cache.detach()

        spikes = 0.
        for t in range(self.T):
            out = self.lin(self.encoder(x))
            spikes += self.neuron(out)

        self.neuron.reset()
        return spikes / self.T

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
