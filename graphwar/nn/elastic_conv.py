import dgl.function as fn
import dgl.ops as ops
import torch
import torch.nn as nn
from dgl import DGLError
from graphwar.functional.transform import dgl_normalize
from torch import Tensor

try:
    import torch_sparse
except ImportError:
    torch_sparse = None


def get_inc(graph):
    """
    compute the incident matrix
    """

    device = graph.device
    row_index, col_index = graph.edges()
    mask = row_index > col_index  # remove duplicate edge and self loop

    row_index = row_index[mask]
    col_index = col_index[mask]
    num_edges = row_index.numel()
    num_nodes = graph.num_nodes()

    row = torch.cat([torch.arange(num_edges, device=device), torch.arange(num_edges, device=device)])
    col = torch.cat([row_index, col_index])
    value = torch.cat([torch.ones(num_edges, device=device), -1 * torch.ones(num_edges, device=device)])
    inc_mat = torch_sparse.SparseTensor(row=row, rowptr=None, col=col, value=value,
                                        sparse_sizes=(num_edges, num_nodes))
    return inc_mat


def inc_norm(inc_mat, graph):
    """
    normalize the incident matrix
    """
    deg = graph.in_degrees()
    deg_inv_sqrt = deg.pow(-0.5)
    inc_mat = torch_sparse.mul(inc_mat, deg_inv_sqrt.view(1, -1))  # col-wise
    return inc_mat


class ElasticConv(nn.Module):
    r"""

    Description
    -----------
    The elastic message passing layer from the paper
    "Elastic Graph Neural Networks", ICML 2021

    Parameters
    ----------
    add_self_loop : bool
        whether to add self-loop edges
    norm : str
        How to apply the normalizer.  Can be one of the following values:

        * ``both``, where the messages are scaled with :math:`1/c_{ji}`, 
        where :math:`c_{ji}` is the product of the square root of node degrees
        (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

        * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
        :math:`c_{ji}` is defined as above.

        * ``right``, to divide the aggregated messages by each node's in-degrees,
        which is equivalent to averaging the received messages.

        * ``none``, where no normalization is applied.

        * ``left``, to divide the messages sent out from each node by its out-degrees,
        equivalent to random walk normalization.      


    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from graphwar.nn import ElasticConv
    >>>
    >>> graph = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 5)
    >>> conv = ElasticConv()
    >>> res = conv(graph, feat)
    >>> res
    tensor([[0.9851, 0.9851, 0.9851, 0.9851, 0.9851],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [0.9256, 0.9256, 0.9256, 0.9256, 0.9256],
            [1.2162, 1.2162, 1.2162, 1.2162, 1.2162],
            [1.1134, 1.1134, 1.1134, 1.1134, 1.1134],
            [0.8726, 0.8726, 0.8726, 0.8726, 0.8726]])
    """

    def __init__(self,
                 k: int = 3,
                 lambda1: float = 3.,
                 lambda2: float = 3.,
                 L21: bool = True,
                 cached: bool = True,
                 add_self_loop: bool = True,
                 norm='both'):

        super().__init__()
        assert torch_sparse, "'ElasticConv' requires a 'torch_sparse' installed." + \
            "See <https://github.com/rusty1s/pytorch_sparse> for more information."

        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._norm = norm
        self._add_self_loop = add_self_loop
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.cached = cached

        self._cached_inc = None  # incident matrix

    def reset_parameters(self):
        self._cached_inc = None

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D)` where :math:`D`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight : torch.Tensor, optional
            Optional edge weight for each edge.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D)`.
        """

        assert edge_weight is None or edge_weight.size(0) == graph.num_edges()

        if self._add_self_loop:
            graph = graph.add_self_loop()
            if edge_weight is not None:
                size = (graph.num_nodes(),) + edge_weight.size()[1:]
                self_loop = edge_weight.new_ones(size)
                edge_weight = torch.cat([edge_weight, self_loop])
        else:
            graph = graph.local_var()

        edge_weight = dgl_normalize(graph, self._norm, edge_weight)
        graph.edata['_edge_weight'] = edge_weight

        cache = self._cached_inc
        if cache is None:
            # compute incident matrix before normalizing edge_index
            inc_mat = get_inc(graph)
            # normalize incident matrix
            inc_mat = inc_norm(inc_mat, graph)

            if self.cached:
                self._cached_inc = inc_mat
                self.init_z = feat.new_zeros((inc_mat.sizes()[0], feat.size()[-1]))
        else:
            inc_mat = self._cached_inc

        feat = self.emp_forward(graph, feat, inc_mat, hh=feat, k=self.k)
        return feat

    def emp_forward(self, graph, feat, inc_mat, hh, k):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)

        if lambda1:
            z = self.init_z

        for k in range(k):

            if lambda2:
                graph.ndata['h'] = feat
                graph.update_all(fn.u_mul_e('h', '_edge_weight', 'm'),
                                 fn.sum('m', 'h'))

                y = gamma * hh + (1 - gamma) * graph.ndata.pop('h')
            else:
                y = gamma * hh + (1 - gamma) * feat  # y = feat - gamma * (feat - hh)

            if lambda1:
                x_bar = y - gamma * (inc_mat.t() @ z)
                z_bar = z + beta * (inc_mat @ x_bar)
                if self.L21:
                    z = self.L21_projection(z_bar, lambda_=lambda1)
                else:
                    z = self.L1_projection(z_bar, lambda_=lambda1)
                feat = y - gamma * (inc_mat.t() @ z)
            else:
                feat = y  # z=0

        return feat

    def L1_projection(self, x: Tensor, lambda_):
        """component-wise projection onto the l∞ ball of radius λ1."""
        return torch.clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x: Tensor, lambda_):
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]  # avoid to be devided by 0
        return scale.unsqueeze(1) * x
