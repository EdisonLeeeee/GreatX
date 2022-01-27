import dgl.function as fn
import torch
import torch.nn as nn
from dgl import DGLError
from graphwar.functional.transform import dgl_normalize
from torch import Tensor


class AdaptiveConv(nn.Module):
    r"""

    Description
    -----------
    The adaptive message passing layer from the paper
    "Graph Neural Networks with Adaptive Residual", NeurIPS 2021

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
    >>> from graphwar.nn import AdaptiveConv
    >>>
    >>> graph = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 5)
    >>> conv = AdaptiveConv()
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
                 lambda_amp: float = 0.1,
                 cached: bool = True,
                 add_self_loop: bool = True,
                 norm='both'):

        super().__init__()

        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._norm = norm
        self._add_self_loop = add_self_loop
        self.k = k
        self.lambda_amp = lambda_amp


    def reset_parameters(self):
        pass
    
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

        feat = self.amp_forward(graph, feat, hh=feat)
        return feat

    def amp_forward(self, graph, x: Tensor, hh: Tensor):
        lambda_amp = self.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  # or simply gamma = 1

        for k in range(self.k):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(graph, x)  # Equation (9)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp)  # Equation (11) and (12)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)  # Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index]  # score is the adaptive score in Equation (14)
        return score.unsqueeze(1) * x

    def compute_LX(self, graph, x: Tensor):
        graph.ndata['h'] = x
        graph.update_all(fn.u_mul_e('h', '_edge_weight', 'm'),
                         fn.sum('m', 'h'))
        return x - graph.ndata.pop('h')
