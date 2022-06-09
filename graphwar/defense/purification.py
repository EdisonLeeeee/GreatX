import torch
import torch.nn.functional as F
import scipy.sparse as sp
from copy import copy

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix

from graphwar.utils import scipy_normalize


class JaccardPurification(BaseTransform):
    r"""Graph purification based on Jaccard similarity of
    connected nodes. 
    As in `"Adversarial Examples on Graph Data: Deep Insights 
    into Attack and Defense"  <https://arxiv.org/abs/1903.01610>`_ paper (IJCAI'19)

    Parameters
    ----------
    threshold : float, optional
        threshold to filter edges based on Jaccard similarity, by default 0.
    allow_singleton : bool, optional
        whether such defense strategy allow singleton nodes, by default False    
    """

    def __init__(self, threshold: float = 0., allow_singleton: bool = False):
        # TODO: add percentage purification
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.removed_edges = None

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)

        row, col = data.edge_index
        A = data.x[row]
        B = data.x[col]
        score = jaccard_similarity(A, B)
        deg = degree(row, num_nodes=data.num_nodes)

        if self.allow_singleton:
            mask = score <= self.threshold
        else:
            mask = torch.logical_and(
                score <= self.threshold, deg[col] > 1)

        self.removed_edges = data.edge_index[:, mask]
        data.edge_index = data.edge_index[:, ~mask]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold}, allow_singleton={self.allow_singleton})'


class CosinePurification(BaseTransform):
    r"""Graph purification based on cosine similarity of
    connected nodes. 

    Note
    ----
    :class:`CosinePurification` is an extension of
    :class:`graphwar.defense.JaccardPurification` for dealing with
    continuous node features.

    Parameters
    ----------
    threshold : float, optional
        threshold to filter edges based on cosine similarity, by default 0.
    allow_singleton : bool, optional
        whether such defense strategy allow singleton nodes, by default False    
    """

    def __init__(self, threshold: float = 0., allow_singleton: bool = False):
        # TODO: add percentage purification
        self.threshold = threshold
        self.allow_singleton = allow_singleton
        self.removed_edges = None

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)

        row, col = data.edge_index
        A = data.x[row]
        B = data.x[col]
        score = F.cosine_similarity(A, B)
        deg = degree(row, num_nodes=data.num_nodes)

        if self.allow_singleton:
            mask = score <= self.threshold
        else:
            mask = torch.logical_and(
                score <= self.threshold, deg[col] > 1)

        self.removed_edges = data.edge_index[:, mask]
        data.edge_index = data.edge_index[:, ~mask]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold}, allow_singleton={self.allow_singleton})'


class SVDPurification(BaseTransform):
    r"""Graph purification based on low-rank 
    Singular Value Decomposition (SVD) reconstruction on
    the adjacency matrix.

    Parameters
    ----------
    K : int, optional
        the top-k largest singular value for reconstruction, by default 50
    threshold : float, optional
        threshold to set elements in the reconstructed adjacency matrix as zero, by default 0.01
    binaryzation : bool, optional
        whether to binarize the reconstructed adjacency matrix, by default False
    remove_edge_index : bool, optional
        whether to remove the :obj:`edge_index` and :obj:`edge_weight`
        int the input :obj:`data` after reconstruction, by default True

    Note
    ----
    We set the reconstructed adjacency matrix as :obj:`adj_t` to be compatible with
    torch_geometric where :obj:`adj_t` denotes the :class:`torch_sparse.SparseTensor`.        
    """

    def __init__(self, K: int = 50, threshold: float = 0.01,
                 binaryzation: bool = False,
                 remove_edge_index: bool = True):
        # TODO: add percentage purification
        super().__init__()
        self.K = K
        self.threshold = threshold
        self.binaryzation = binaryzation
        self.remove_edge_index = remove_edge_index

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)

        device = data.edge_index.device
        adj_matrix = to_scipy_sparse_matrix(data.edge_index, data.edge_weight,
                                            num_nodes=data.num_nodes).tocsr()
        adj_matrix = svd(adj_matrix, K=self.K,
                         threshold=self.threshold,
                         binaryzation=self.binaryzation)
        data.adj_t = torch.as_tensor(
            adj_matrix, dtype=torch.float, device=device)
        if self.remove_edge_index:
            del data.edge_index, data.edge_weight
        else:
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
            data.edge_index, data.edge_weight = edge_index.to(
                device), edge_weight.to(device)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, threshold={self.threshold}, allow_singleton={self.allow_singleton})'


class EigenDecomposition(BaseTransform):
    r"""Graph purification based on low-rank 
    Eigen Decomposition reconstruction on
    the adjacency matrix.

    :class:`EigenDecomposition` is similar to :class:`graphwar.defense.SVDPurification`

    Parameters
    ----------
    K : int, optional
        the top-k largest singular value for reconstruction, by default 50
    normalize : bool, optional
        whether to normalize the input adjacency matrix
    remove_edge_index : bool, optional
        whether to remove the :obj:`edge_index` and :obj:`edge_weight`
        int the input :obj:`data` after reconstruction, by default True

    Note
    ----
    We set the reconstructed adjacency matrix as :obj:`adj_t` to be compatible with
    torch_geometric where :obj:`adj_t` denotes the :class:`torch_sparse.SparseTensor`.
    """

    def __init__(self, K: int = 50, normalize: bool = True,
                 remove_edge_index: bool = True):
        super().__init__()
        self.K = K
        self.normalize = normalize
        self.remove_edge_index = remove_edge_index

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)

        device = data.edge_index.device
        adj_matrix = to_scipy_sparse_matrix(data.edge_index, data.edge_weight,
                                            num_nodes=data.num_nodes).tocsr()

        if self.normalize:
            adj_matrix = scipy_normalize(adj_matrix)

        adj_matrix = adj_matrix.asfptype()
        V, U = sp.linalg.eigsh(adj_matrix, k=self.K)
        adj_matrix = (U * V) @ U.T
        # sparsification
        adj_matrix[adj_matrix < 0] = 0.

        V = torch.as_tensor(V, dtype=torch.float)
        U = torch.as_tensor(U, dtype=torch.float)

        data.V, data.U = V.to(device), U.to(device)

        data.adj_t = torch.as_tensor(
            adj_matrix, dtype=torch.float, device=device)

        if self.remove_edge_index:
            del data.edge_index, data.edge_weight
        else:
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
            data.edge_index, data.edge_weight = edge_index.to(
                device), edge_weight.to(device)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'


def jaccard_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    intersection = torch.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (torch.count_nonzero(A, dim=1) +
                              torch.count_nonzero(B, dim=1) - intersection + 1e-7)
    return J


def svd(adj_matrix: sp.csr_matrix, K: int = 50,
        threshold: float = 0.01,
        binaryzation: bool = False) -> sp.csr_matrix:

    adj_matrix = adj_matrix.asfptype()

    U, S, V = sp.linalg.svds(adj_matrix, k=K)
    adj_matrix = (U * S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binaryzation:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0

    return adj_matrix
