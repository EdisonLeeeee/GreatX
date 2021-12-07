import torch
import torch.nn as nn
import dgl.function as fn
from dgl import DGLError
from graphwar.nn import Linear
from graphwar.utils.normalize import dgl_normalize


class SGConv(nn.Module):
    r"""

    Description
    -----------
    Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    .. math::
        H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta

    where :math:`\tilde{A}` is :math:`A` + :math:`I`.
    Thus the graph input is expected to have self-loop edges added.

    Parameters
    ----------
    in_feats : int
        Number of input features; i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Number of output features; i.e, the number of dimensions of :math:`H^{K}`.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
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

    cached : bool
        If True, the module would cache

        .. math::
            (\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
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
    >>> from graphwar.nn import SGConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = SGConv(10, 2, k=2, cached=True)
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
                 k=1,
                 add_self_loop=True,
                 norm='both',
                 cached=False,
                 weight=True,
                 bias=True):
        
        super().__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))     
        self._in_feats = in_feats
        self._out_feats = out_feats            
        self._cached = cached
        self._cached_h = None
        self._k = k
        self._norm = norm
        self._add_self_loop = add_self_loop

        self.linear = Linear(in_feats, out_feats, weight=weight, bias=bias)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""

        self.linear.reset_parameters()

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

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

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """

        if self._cached and self._cached_h is not None:
            feat = self._cached_h
        else:
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

            for _ in range(self._k):
                graph.ndata['h'] = feat
                graph.update_all(fn.u_mul_e('h', '_edge_weight', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')

            # cache feature
            if self._cached:
                self._cached_h = feat

        return self.linear(feat)
    
    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)        
