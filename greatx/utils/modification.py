import copy
import torch
from torch import Tensor
import scipy.sparse as sp
from torch_geometric.utils import sort_edge_index, to_scipy_sparse_matrix, from_scipy_sparse_matrix


def add_edges(edge_index: Tensor, edges_to_add: Tensor,
              symmetric: bool = True, sort_edges: bool = True) -> Tensor:
    """Add edges to the graph denoted as :obj:`edge_index`.

    Parameters
    ----------
    edge_index : Tensor
        the graph instance where edges will be removed from.
    edges_to_add : torch.Tensor
        shape [2, M], the edges to be added into the graph.
    symmetric : bool
        whether the output graph is symmetric, if True,
        it would add the edges into the graph by:
        :obj:`edges_to_add = torch.cat([edges_to_add, edges_to_add.flip(0)], dim=1)`

    Returns
    -------
    Tensor
        the graph instance :obj:`edge_index` with edges added.
    """
    if edges_to_add.size(1) == 0:
        return edge_index
    
    if symmetric:
        edges_to_add = torch.cat([edges_to_add, edges_to_add.flip(0)], dim=1)

    edges_to_add = edges_to_add.to(edge_index)
    edge_index = torch.cat([edge_index, edges_to_add], dim=1)
    edge_index = sort_edge_index(edge_index)
    return edge_index


def remove_edges(edge_index: Tensor, edges_to_remove: Tensor, symmetric: bool = True) -> Tensor:
    """Remove edges from the graph denoted as :obj:`edge_index`. 

    Parameters
    ----------
    edge_index : Tensor
        the graph instance where edges will be removed from.
    edges_to_remove : torch.Tensor
        shape [2, M], the edges to be removed in the graph.
    symmetric : bool
        whether the output graph is symmetric, if True,
        it would remove the edges from the graph by:
        :obj:`edges_to_remove = torch.cat([edges_to_remove, edges_to_remove.flip(0)], dim=1)`

    Returns
    -------
    Tensor
        the graph instance :obj:`edge_index` with edges removed.
    """
    
    if edges_to_remove.size(1) == 0:
        return edge_index
    
    device = edge_index.device
    if symmetric:
        edges_to_remove = torch.cat(
            [edges_to_remove, edges_to_remove.flip(0)], dim=1)
    edges_to_remove = edges_to_remove.to(edge_index)

    num_nodes = max(edge_index.max().item(), edges_to_remove.max().item()) + 1
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr(copy=False)

    row, col = edges_to_remove.cpu().numpy()
    adj_matrix = adj_matrix.tolil(copy=True)
    adj_matrix[(row, col)] = 0
    adj_matrix = adj_matrix.tocsr(copy=False)
    adj_matrix.eliminate_zeros()

    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    edge_index = sort_edge_index(edge_index)
    return edge_index.to(device)


def flip_edges(edge_index: Tensor, 
               edges_to_flip: Tensor, 
               symmetric: bool = True) -> Tensor:
    """Flip edges from the graph denoted as :obj:`edge_index`. 

    Parameters
    ----------
    edge_index : Tensor
        the graph instance where edges will be flipped from.
    edges_to_flip : torch.Tensor
        shape [2, M], the edges to be flipped in the graph.
    symmetric : bool
        whether the output graph is symmetric, if True,
        it would flip the edges from the graph by:
        :obj:`edges_to_flip = torch.cat([edges_to_flip, edges_to_flip.flip(0)], dim=1)`

    Returns
    -------
    Tensor
        the graph instance :obj:`edge_index` with edges flipped.
    """
    
    if edges_to_flip.size(1) == 0:
        return edge_index
    
    device = edge_index.device
    if symmetric:
        edges_to_flip = torch.cat(
            [edges_to_flip, edges_to_flip.flip(0)], dim=1)
        
    edges_to_flip = edges_to_flip.to(edge_index)
    
    num_nodes = max(edge_index.max().item(), edges_to_flip.max().item()) + 1
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr(copy=False)
    
    row, col = edges_to_flip.cpu().numpy()
    data = adj_matrix[(row, col)].A
    data[data > 0.] = 1.
    data[data < 0.] = 0.

    adj_matrix = adj_matrix.tolil(copy=True)
    adj_matrix[(row, col)] = 1. - data
    adj_matrix = adj_matrix.tocsr(copy=False)
    adj_matrix.eliminate_zeros()

    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    edge_index = sort_edge_index(edge_index)
    return edge_index.to(device)

def flip_graph(data, edges_to_flip, 
               symmetric: bool = True):
    
    data = copy.copy(data)
    data.edge_index = flip_edges(data.edge_index, edges_to_flip, symmetric=symmetric)
    data.edge_weight = None
    data.adj_t = None
    return data