import torch
from torch import nn
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor

from graphwar import is_edge_index
from graphwar.functional import spmm
from graphwar.nn.layers.gcn_conv import dense_gcn_norm

class DAGNNConv(nn.Module):
    """
    Deep Adaptive Graph Neural Network in
    `Towards Deeper Graph Neural Networks <https://arxiv.org/abs/2007.09296>`
    """

    def __init__(self, in_channels: int, out_channels: int = 1, K: int = 1,
                 add_self_loops: bool = True, bias: bool = True):
        super().__init__()

        assert out_channels == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=bias,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        is_edge_like = is_edge_index(edge_index)
        
        if is_edge_like:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(0), False,
                self.add_self_loops, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(0), False,
                self.add_self_loops, dtype=x.dtype)
        else:
            # N by N dense adjacency matrix
            adj = dense_gcn_norm(edge_index, add_self_loops=self.add_self_loops)                

        xs = [x]
        for _ in range(self.K):
            if is_edge_like:
                x = spmm(x, edge_index, edge_weight)
            else:
                x = adj @ x      
            xs.append(x)

        H = torch.stack(xs, dim=1)
        S = self.lin(H).sigmoid()
        S = S.permute(0, 2, 1)
        out = torch.matmul(S, H).squeeze()

        return out                

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

    

