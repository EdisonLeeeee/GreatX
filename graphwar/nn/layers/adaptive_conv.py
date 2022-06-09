import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor

from graphwar import is_edge_index
from graphwar.functional import spmm
from graphwar.nn.layers.gcn_conv import dense_gcn_norm


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
    for the inputs :obj:`x`, :obj:`edge_index`, and :obj:`edge_weight`,
    our implementation supports:

    * :obj:`edge_index` is :class:`torch.FloatTensor`: dense adjacency matrix with shape :obj:`[N, N]`
    * :obj:`edge_index` is :class:`torch.LongTensor`: edge indices with shape :obj:`[2, M]`
    * :obj:`edge_index` is :class:`torch_sparse.SparseTensor`: sparse matrix with sparse shape :obj:`[N, N]`     

    See also
    --------
    :class:`graphwar.nn.models.AirGNN`     
    """

    def __init__(self,
                 K: int = 3,
                 lambda_amp: float = 0.1,
                 normalize: bool = True,
                 add_self_loops: bool = True):
        super().__init__()

        self.K = K
        self.lambda_amp = lambda_amp
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        is_edge_like = is_edge_index(edge_index)

        if self.normalize:
            if is_edge_like:
                edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(0),
                                                   improved=False,
                                                   add_self_loops=self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(edge_index, x.size(0),
                                      improved=False,
                                      add_self_loops=self.add_self_loops, dtype=x.dtype)

            else:
                # N by N dense adjacency matrix
                edge_index = dense_gcn_norm(edge_index, improved=False,
                                            add_self_loops=self.add_self_loops)

        return self.amp_forward(x, edge_index, edge_weight)

    def amp_forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        lambda_amp = self.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  # or simply gamma = 1
        hh: Tensor = x

        for k in range(self.K):
            y = x - gamma * 2 * \
                (1 - lambda_amp) * self.compute_LX(x,
                                                   edge_index, edge_weight)  # Equation (9)
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

    def compute_LX(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        is_edge_like = is_edge_index(edge_index)

        if is_edge_like:
            out = spmm(x, edge_index, edge_weight)
        else:
            out = edge_index @ x

        return x - out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(lambda_amp={self.lambda_amp}, K={self.K})')
