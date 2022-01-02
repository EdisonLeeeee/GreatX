import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    r"""Applies a linear tranformation to the incoming data

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    similar to :class:`torch.nn.Linear` but more flexible.
    """

    def __init__(self, in_feats: int, out_feats: int,
                 weight: bool = True, bias: bool = True):
        """Initialization of Linear layer

        Parameters
        ----------
        in_feats : int
            number of input features dimensions
        out_feats : int
            number of output feature dimensions
        weight : bool, optional
            decide if weight matrix is necessary , by default True
        bias : bool, optional
            decide if bias term is necessary, by default True
        """

        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_feats))
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

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is not None:
            if x.is_sparse:
                x = torch.sparse.mm(x, self.weight)
            else:
                x = x @ self.weight

        if self.bias is not None:
            x = x + self.bias

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_feats}, '
                f'{self.out_feats}, weight={self.weight is not None}, bias={self.bias is not None})')
