import torch as th
import torch.nn as nn
import dgl.function as fn
import dgl.ops as ops
from torch.nn import init
from dgl import DGLError


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
    cached : bool
        If True, the module would cache

        .. math::
            (\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from graphwar.layers import SGConv
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
                 cached=False,
                 weight=True,
                 bias=True):
        super(SGConv, self).__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
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
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
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
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():

            if self._cached and self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5).to(feat.device).unsqueeze(1)

                # compute (D^-0.5 * A * D^-0.5)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_src('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                # cache feature
                if self._cached:
                    self._cached_h = feat

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = th.matmul(feat, weight)

            if self.bias is not None:
                feat = feat + self.bias
            return feat
