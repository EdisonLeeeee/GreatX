from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, mul

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


def get_inc(edge_index: Adj, num_nodes: Optional[int] = None) -> SparseTensor:
    """Compute the incident matrix
    """
    device = edge_index.device
    if torch.is_tensor(edge_index):
        row_index, col_index = edge_index
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    else:
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        num_nodes = edge_index.sizes()[1]

    mask = row_index > col_index  # remove duplicate edge and self loop

    row_index = row_index[mask]
    col_index = col_index[mask]
    num_edges = row_index.numel()

    row = torch.cat([
        torch.arange(num_edges, device=device),
        torch.arange(num_edges, device=device)
    ])
    col = torch.cat([row_index, col_index])
    value = torch.cat([
        torch.ones(num_edges, device=device),
        -torch.ones(num_edges, device=device)
    ])
    inc_mat = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(num_edges, num_nodes))
    return inc_mat


def inc_norm(inc: SparseTensor, edge_index: Adj,
             num_nodes: Optional[int] = None) -> SparseTensor:
    """Normalize the incident matrix
    """

    if torch.is_tensor(edge_index):
        deg = degree(edge_index[0], num_nodes=num_nodes,
                     dtype=torch.float).clamp(min=1)
    else:
        deg = edge_index.sum(1).clamp(min=1)

    deg_inv_sqrt = deg.pow(-0.5)
    inc = mul(inc, deg_inv_sqrt.view(1, -1))  # col-wise
    return inc


class ElasticConv(nn.Module):
    r"""
    The ElasticGNN operator from the `"Elastic Graph Neural
    Networks" <https://arxiv.org/abs/2107.06996>`_
    paper (ICML'21)

    Parameters
    ----------
    K : int, optional
        the number of propagation steps, by default 3
    lambda_amp : float, optional
        trade-off of adaptive message passing, by default 0.1
    normalize : bool, optional
        Whether to add self-loops and compute
        symmetric normalization coefficients on the fly, by default True
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    lambda1 : float, optional
        trade-off hyperparameter, by default 3
    lambda2 : float, optional
        trade-off hyperparameter, by default 3
    L21 : bool, optional
        whether to use row-wise projection
        on the l2 ball of radius λ1., by default True
    cached : bool, optional
        whether to cache the incident matrix, by default True


    See also
    --------
    :class:`~greatx.nn.models.supervised.ElasticGNN`
    """

    _cached: Optional[SparseTensor] = None  # incident matrix

    def __init__(self, K: int = 3, lambda_amp: float = 0.1,
                 normalize: bool = True, add_self_loops: bool = True,
                 lambda1: float = 3., lambda2: float = 3., L21: bool = True,
                 cached: bool = True):

        super().__init__()

        self.K = K
        self.lambda_amp = lambda_amp
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.cached = cached

    def reset_parameters(self):
        self.cache_clear()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._cached = None
        return self

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        cache = self._cached

        if cache is None:
            if self.add_self_loops:
                # NOTE: we do not support Dense adjacency matrix here
                edge_index, edge_weight = make_self_loops(
                    edge_index, edge_weight, num_nodes=x.size(0))

            if self.normalize:
                # NOTE: we do not support Dense adjacency matrix here
                edge_index, edge_weight = make_gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0),
                    dtype=x.dtype, add_self_loops=False)

            # compute incident matrix before normalizing edge_index
            inc_mat = get_inc(edge_index, num_nodes=x.size(0))
            # normalize incident matrix
            inc_mat = inc_norm(inc_mat, edge_index, num_nodes=x.size(0))

            if self.cached:
                self._cached = (inc_mat, edge_index, edge_weight)
                self.init_z = x.new_zeros((inc_mat.sizes()[0], x.size()[-1]))
        else:
            inc_mat, edge_index, edge_weight = self._cached

        return self.emp_forward(x, inc_mat, edge_index, edge_weight)

    def emp_forward(self, x: Tensor, inc_mat: SparseTensor, edge_index: Adj,
                    edge_weight: OptTensor = None) -> Tensor:
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)

        hh = x

        if lambda1:
            z = self.init_z

        for k in range(self.K):

            if lambda2:
                out = spmm(x, edge_index, edge_weight)

                y = gamma * hh + (1 - gamma) * out
            else:
                y = gamma * hh + (1 - gamma) * x  # y = x - gamma * (x - hh)

            if lambda1:
                x_bar = y - gamma * (inc_mat.t() @ z)
                z_bar = z + beta * (inc_mat @ x_bar)
                if self.L21:
                    z = self.L21_projection(z_bar, lambda_=lambda1)
                else:
                    z = self.L1_projection(z_bar, lambda_=lambda1)
                x = y - gamma * (inc_mat.t() @ z)
            else:
                x = y  # z=0

        return x

    def L1_projection(self, x: Tensor, lambda_: float) -> Tensor:
        """component-wise projection onto the l∞ ball of radius λ1."""
        return torch.clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x: Tensor, lambda_: float) -> Tensor:
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / \
            row_norm[index]  # avoid to be devided by 0
        return scale.unsqueeze(1) * x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"
