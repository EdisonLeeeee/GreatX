import torch
from torch import Tensor, nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


class DAGNNConv(nn.Module):
    r"""The DAGNN operator from the `"Towards Deeper Graph Neural
    Networks" <https://arxiv.org/abs/2007.09296>`_
    paper (KDD'20)

    Parameters
    ----------
    in_channels : int
        dimensions of input samples
    out_channels : int, optional
        dimensions of output samples,
        must be 1 for any cases, by default 1
    K : int, optional
        the number of propagation steps, by default 1
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
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
    :class:`~greatx.nn.models.supervised.DAGNN`
    """
    def __init__(self, in_channels: int, out_channels: int = 1, K: int = 1,
                 add_self_loops: bool = True, bias: bool = True):
        super().__init__()

        assert out_channels == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=bias,
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
        for _ in range(self.K):
            x = spmm(x, edge_index, edge_weight)
            xs.append(x)

        H = torch.stack(xs, dim=1)
        S = self.lin(H).sigmoid()
        S = S.permute(0, 2, 1)
        out = torch.matmul(S, H).squeeze()

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
