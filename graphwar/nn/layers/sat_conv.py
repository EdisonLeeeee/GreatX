from typing import Optional

import torch
from torch import nn
from torch import Tensor

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops


from graphwar import is_edge_index
from graphwar.functional import spmm
from graphwar.nn.layers.gcn_conv import dense_gcn_norm


class SATConv(nn.Module):
    r"""The spectral adversarial training operator
    from the `"Spectral Adversarial Training for Robust Graph Neural Network"
    <https://arxiv.org/>`_ paper (arXiv'22)

    Parameters
    ----------
    in_channels : int
        dimensions of int samples
    out_channels : int
        dimensions of output samples
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    normalize : bool, optional
        whether to compute symmetric normalization
        coefficients on the fly, by default True
    bias : bool, optional
        whether to use bias in the layers, by default True    

    Note
    ----
    For the inputs :obj:`x`, :obj:`U`, and :obj:`V`,
    our implementation supports:

    * :obj:`U` is :class:`torch.LongTensor`: edge indices with shape :obj:`[2, M]`
    * :obj:`U` is :class:`torch.FloatTensor` and :obj:`V` is :obj:`None`: dense matrix with shape :obj:`[N, N]`
    * :obj:`U` and :obj:`V` are :class:`torch.FloatTensor`: eigenvector and corresponding eigenvalues           

    See also
    --------
    :class:`graphwar.nn.models.SAT`       
    """

    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 bias: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, U: Tensor, V: Optional[Tensor] = None):
        # NOTE: torch_sparse.SparseTensor is not supported
        is_edge_like = is_edge_index(U)
        x = self.lin(x)

        if is_edge_like:
            edge_index, edge_weight = U, V
            if self.add_self_loops:
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                         num_nodes=x.size(0))
            if self.normalize:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(0), False,
                    add_self_loops=False, dtype=x.dtype)

            x = spmm(x, edge_index, edge_weight)

        elif V is None:
            adj = U
            if self.normalize:
                adj = dense_gcn_norm(adj, add_self_loops=self.add_self_loops)
            x = adj @ x
        else:
            x = (U * V) @ (U.t() @ x)

        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


# class SATConv(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int,
#                  K: int = 5,
#                  alpha: float = 0.1,
#                  add_self_loops: bool = True,
#                  normalize: bool = True,
#                  bias: bool = False):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         self.alpha = alpha
#         self.add_self_loops = add_self_loops
#         self.normalize = normalize

#         self.lin = Linear(in_channels, out_channels, bias=False,
#                           weight_initializer='glorot')
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin.reset_parameters()

#     def forward(self, x: Tensor, U: Tensor, V: Optional[Tensor] = None):
#         is_edge_like = is_edge_index(U)
#         x = self.lin(x)

#         if is_edge_like:
#             edge_index, edge_weight = U, V
#             if self.normalize:
#                 edge_index, edge_weight = gcn_norm(  # yapf: disable
#                     edge_index, edge_weight, x.size(0), False,
#                     self.add_self_loops, dtype=x.dtype)

#             x_out = self.alpha*x
#             for _ in range(self.K):
#                 x = spmm(x, edge_index, edge_weight)
#                 x_out = x_out + (1 - self.alpha)/self.K * x
#         elif V is None:
#             adj = U
#             if self.normalize:
#                 adj = dense_gcn_norm(adj, add_self_loops=self.add_self_loops)
#             x_in = x
#             x_out = torch.zeros_like(x)
#             for _ in range(self.K):
#                 x = adj @ x
#                 x_out += (1 - self.alpha) * x
#             x_out /= self.K
#             x_out += self.alpha * x_in
#         else:
#             V_out = 0.
#             V_pow = 1.
#             for _ in range(self.K):
#                 V_pow = V_pow * V
#                 V_out = V_out + (1 - self.alpha) / self.K * V_pow
#             x_out = (U * V_out) @ (U.t() @ x) + self.alpha * x

#         if self.bias is not None:
#             x_out += self.bias
#         return x_out

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, K={self.K}, alpha={self.alpha})')
