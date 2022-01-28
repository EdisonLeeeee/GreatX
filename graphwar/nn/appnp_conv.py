import dgl.function as fn
import torch
import torch.nn as nn
from dgl import DGLError

from graphwar.functional.transform import dgl_normalize


class APPNPConv(nn.Module):
    r"""

    Description
    -----------
    Approximated personalized propagation
    of neural predictions (APPNP) in
    `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/abs/1810.05997>`

    Parameters
    ----------

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from graphwar.nn import APPNPConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = APPNPConv(k=3, alpha=0.5)
    >>> res = conv(g, feat)
    >>> res
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000],
            [1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303, 1.0303,
            1.0303],
            [0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643, 0.8643,
            0.8643],
            [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
            0.5000]])

    """

    def __init__(self,
                 k: int = 10,
                 alpha: float = 0.1,
                 edge_drop: float = 0.,
                 add_self_loop: bool = True,
                 norm: str = 'both'):

        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._k = k
        self._alpha = alpha
        self._norm = norm
        self._add_self_loop = add_self_loop
        self.edge_drop = nn.Dropout(edge_drop)

    def reset_parameters(self):
        pass

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Approximated personalized propagation layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight : torch.Tensor, optional
            Optional edge weight for each edge.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
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

        feat_0 = feat
        for _ in range(self._k):
            graph.ndata['h'] = feat
            graph.edata['_edge_weight'] = self.edge_drop(edge_weight)
            graph.update_all(fn.u_mul_e('h', '_edge_weight', 'm'),
                             fn.sum('m', 'h'))
            feat = (1 - self._alpha) * graph.ndata.pop('h') + self._alpha * feat_0

        return feat

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'k={_k}, alpha={_alpha}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
