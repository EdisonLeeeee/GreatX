import torch
from torch import Tensor, nn
from torch_geometric.typing import Adj, OptTensor

from greatx.functional import spmm
from greatx.nn.layers.gcn_conv import make_gcn_norm, make_self_loops


class AdaptiveConv(nn.Module):
    r"""The AirGNN operator from the `"Graph Neural Networks
    with Adaptive Residual" <https://openreview.net/forum?id=hfkER_KJiNw>`_
    paper (NeurIPS'21)

    Parameters
    ----------
    K : int, optional
        the number of propagation steps during message passing, by default 3
    lambda_amp : float, optional
        trade-off for adaptive message passing, by default 0.1
    normalize : bool, optional
        Whether to add self-loops and compute
        symmetric normalization coefficients on the fly, by default True
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True

    Note
    ----
    Different from that in :class:`torch_geometric`,
    for the input :obj:`edge_index`, our implementation supports
    :obj:`torch.FloatTensor`, :obj:`torch.LongTensor`
    and obj:`torch_sparse.SparseTensor`.

    See also
    --------
    :class:`greatx.nn.models.supervised.AirGNN`
    """
    def __init__(self, K: int = 3, lambda_amp: float = 0.1,
                 normalize: bool = True, add_self_loops: bool = True):
        super().__init__()

        self.K = K
        self.lambda_amp = lambda_amp
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.add_self_loops:
            edge_index, edge_weight = make_self_loops(edge_index, edge_weight,
                                                      num_nodes=x.size(0))

        if self.normalize:
            edge_index, edge_weight = make_gcn_norm(edge_index, edge_weight,
                                                    num_nodes=x.size(0),
                                                    dtype=x.dtype,
                                                    add_self_loops=False)

        return self.amp_forward(x, edge_index, edge_weight)

    def amp_forward(self, x: Tensor, edge_index: Adj,
                    edge_weight: OptTensor = None) -> Tensor:
        lambda_amp = self.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  # or simply gamma = 1
        hh = x

        for k in range(self.K):
            # Equation (9)
            y = x - gamma * 2 * \
                (1 - lambda_amp) * self.compute_LX(x, edge_index, edge_weight)
            # Equation (11) and (12)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp)
        return x

    def proximal_L21(self, x: Tensor, lambda_: float) -> Tensor:
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        # Deal with the case when the row_norm is 0
        index = torch.where(row_norm > 0)
        # score is the adaptive score in Equation (14)
        score[index] = score[index] / row_norm[index]
        return score.unsqueeze(1) * x

    def compute_LX(self, x: Tensor, edge_index: Adj,
                   edge_weight: OptTensor = None) -> Tensor:
        out = spmm(x, edge_index, edge_weight)

        return x - out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"
