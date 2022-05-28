import torch
from torch_sparse import SparseTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

__all__ = ['to_sparse_tensor', 'to_dense_adj']

def to_sparse_tensor(edge_index, edge_weight=None, num_nodes=None, is_sorted=False):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    return SparseTensor.from_edge_index(
        edge_index, edge_weight, is_sorted=is_sorted, 
        sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def to_dense_adj(edge_index, edge_weight=None, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    if edge_weight is None:
        adj[edge_index[0], edge_index[1]] = 1.0
    else:
        adj[edge_index[0], edge_index[1]] = edge_weight
    return adj
        