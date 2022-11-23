import torch.nn as nn

from greatx.nn.layers import Sequential, SpikingGCNonv
from greatx.utils import wrapper


class SpikingGCN(nn.Module):
    r"""The spiking graph convolutional neural network from
    the `"Spiking Graph Convolutional Networks"
    <https://arxiv.org/abs/2205.02767>`_ paper (IJCAI'22)

    Parameters
    ----------
    in_channels : int,
        the input dimensions of model
    out_channels : int,
        the output dimensions of model
    hids : list, optional
        the number of hidden units for each hidden layer,
        by default []
    acts : list, optional
        the activation function for each hidden layer,
        by default []
    K : int, optional
        the number of propagation steps, by default 2
    T : int
        the number of time steps, by default 20
    tau : float
        the :math:`\tau` in LIF neuron, by default 2.0
    v_threshold : float
        the threshold :math:`V_{th}` in LIF neuron, by default 1.0
    v_reset : float
        the reset level :math:`V_{reset}` in LIF neuron, by default 0
    dropout : float, optional
        the dropout ratio of model, by default 0.
    bias : bool, optional
        whether to use bias in the layers, by default True
    cached : bool, optional
        whether the layer will cache
        the computation of :math:`(\mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2})^K`
        on first execution, and will use the
        cached version for further executions, by default True
    bn: bool, optional
        whether to use :class:`BatchNorm1d` after the convolution layer,
        by default False
    bn_input: bool, optional
        whether to use :class:`BatchNorm1d` before input to
        the convolution layer, by default False

    Note
    ----
    To accept a different graph as inputs, please call :meth:`cache_clear`
    first to clear cached results.

    Examples
    --------
    >>> # SGC without hidden layer
    >>> model = SpikingGCN(100, 10)

    See also
    --------
    :class:`~greatx.nn.layers.SpikingGCNonv`

    """
    @wrapper
    def __init__(
        self,
        in_channels,
        out_channels,
        hids: list = [],
        acts: list = [],
        K: int = 2,
        T: int = 20,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.,
        dropout: float = 0.,
        bias: bool = True,
        cached: bool = True,
        bn: bool = False,
    ):
        super().__init__()

        assert len(hids) == len(acts) == 0

        conv = []
        if bn:
            conv.append(nn.BatchNorm1d(in_channels))
        else:
            conv.append(nn.Identity())

        conv.append(
            SpikingGCNonv(in_channels, out_channels, bias=bias, K=K, T=T,
                          cached=cached, tau=tau, v_threshold=v_threshold,
                          v_reset=v_reset))
        self.conv = Sequential(*conv)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        for layer in self.conv:
            if hasattr(layer, 'cache_clear'):
                layer.cache_clear()
        return self

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        return self.conv(x, edge_index, edge_weight)
