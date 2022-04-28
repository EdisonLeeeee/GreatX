import dgl.function as fn
import torch
import torch.nn as nn
from dgl import DGLError

from graphwar.functional.transform import dgl_normalize


class GCNConv(nn.Module):
    r"""

    Description
    -----------
    Graph convolution was introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and mathematically is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ji}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    in_feats : int
        Number of input features; i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Number of output features; i.e, the number of dimensions of :math:`H^{K}`.
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

    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.        
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.        
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from graphwar.nn import GCNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = GCNConv(10, 2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.9441, -0.9343],
            [-1.9441, -0.9343],
            [-1.9441, -0.9343],
            [-2.7709, -1.3316],
            [-1.9297, -0.9273],
            [-1.9441, -0.9343]], grad_fn=<AddmmBackward>)
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 add_self_loop=True,
                 norm='both',
                 activation=None,
                 weight=True,
                 bias=True):

        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._add_self_loop = add_self_loop
        self._activation = activation

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

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
        graph.edata['_edge_weight'] = edge_weight

        if self.weight is not None:
            feat = feat @ self.weight

        graph.ndata['h'] = feat
        graph.update_all(fn.u_mul_e('h', '_edge_weight', 'm'),
                         fn.sum('m', 'h'))
        feat = graph.ndata.pop('h')

        if self.bias is not None:
            feat = feat + self.bias

        if self._activation is not None:
            feat = self._activation(feat)
        return feat

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        summary += ', add_self_loop={_add_self_loop}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
