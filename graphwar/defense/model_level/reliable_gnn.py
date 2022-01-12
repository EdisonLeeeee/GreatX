import torch.nn as nn

from graphwar.config import Config
from graphwar.nn import DimwiseMedianConv, Sequential, SoftKConv, activations
from graphwar.utils import wrapper

_EDGE_WEIGHT = Config.edge_weight


class ReliableGNN(nn.Module):
    """ReliableGNN with `weighted dimension-wise Median` aggregation function.

    Example
    -------
    # ReliableGNN with one hidden layer
    >>> model = ReliableGNN(100, 10)

    # ReliableGNN with two hidden layers
    >>> model = ReliableGNN(100, 10, hids=[32, 16], acts=['relu', 'elu'])

    # ReliableGNN with two hidden layers, without activation at the first layer
    >>> model = ReliableGNN(100, 10, hids=[32, 16], acts=[None, 'relu'])

    # ReliableGNN with weighted dimension-wise Median aggregation
    >>> model = ReliableGNN(100, 10, method="dimmedian")

    # ReliableGNN with Soft Weighted Medoid topk aggregation
    >>> model = ReliableGNN(100, 10, method="softk")

    # ReliableGNN with Soft Weighted Medoid topk aggregation
    >>> model = ReliableGNN(100, 10, method="softk", k=64, temperature=0.5)    

    Note
    ----
    please make sure `hids` and `acts` are both `list` or `tuple` and
    `len(hids)==len(acts)`.

    """

    @wrapper
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hids: list = [16],
                 acts: list = ['relu'],
                 dropout: float = 0.5,
                 bias: bool = True,
                 bn: bool = False,
                 norm: str = 'none',
                 method: str = 'dimmedian',
                 row_normalize: bool = False,
                 **kwargs):
        r"""
        Parameters
        ----------
        in_features : int, 
            the input dimmensions of model
        out_features : int, 
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activaction function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False            
        norm : str, optional
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

        method : str, `dimmedian` or `softk` optional
            the robust aggregation function, by default `dimmedian`.
            * `dimmedian`: weighted dimension-wise Median aggregation function
            * `softk`: Soft Weighted Medoid in the top `k` 
            neighborhood aggregation function

            if method=`softk`, users can specify the `softk` parameters, including:
                * k : int, optional
                    Neighborhood size for selecting the top k elements, by default 32.
                * temperature : float, optional
                    Controlling the steepness of the softmax, by default 1.0.
                * with_weight_correction : bool, optional
                    For enabling an alternative normalisazion (see above), by default True.
        row_normalize : bool, optional
            whether to perform normalization for aggregated features, by default False.
        kwargs : dict, optional
            the `softk` parameters including `k`, `temperature`, `with_weight_correction`
        """

        super().__init__()

        conv = []
        assert method in {"dimmedian", "softk"}
        if method == "dimmedian" and kwargs:
            raise ValueError("keyword arguments were not supported for method='dimmedian'.")

        assert len(hids) == len(acts)

        for hid, act in zip(hids, acts):
            if method == "dimmedian":
                conv.append(DimwiseMedianConv(in_features,
                                              hid,
                                              bias=bias, norm=norm,
                                              row_normalize=row_normalize,
                                              activation=None))
            else:
                conv.append(SoftKConv(in_features,
                                      hid,
                                      bias=bias, norm=norm,
                                      row_normalize=row_normalize,
                                      activation=None, **kwargs))

            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        if method == "dimmedian":
            conv.append(DimwiseMedianConv(in_features, out_features,
                                          row_normalize=row_normalize,
                                          bias=bias, norm=norm))
        else:
            conv.append(SoftKConv(in_features, out_features,
                                  row_normalize=row_normalize,
                                  bias=bias, norm=norm, **kwargs))

        self.conv = Sequential(*conv, loc=1)  # `loc=1` specifies the location of features.

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, g, feat, edge_weight=None):
        if edge_weight is None:
            edge_weight = g.edata.get(_EDGE_WEIGHT, edge_weight)
        return self.conv(g, feat, edge_weight=edge_weight)
