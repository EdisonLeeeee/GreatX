from copy import copy

import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (degree, dropout_adj,
                                   from_scipy_sparse_matrix,
                                   to_scipy_sparse_matrix)

from greatx.functional import to_dense_adj
from greatx.nn.layers.gcn_conv import dense_gcn_norm
from greatx.utils import scipy_normalize


class JaccardPurification(BaseTransform):
    r"""Graph purification based on Jaccard similarity of
    connected nodes.
    As in `"Adversarial Examples on Graph Data: Deep Insights
    into Attack and Defense"  <https://arxiv.org/abs/1903.01610>`_
    paper (IJCAI'19)

    Parameters
    ----------
    threshold : float, optional
        threshold to filter edges based on Jaccard similarity,
        by default 0.
    allow_singleton : bool, optional
        whether such defense strategy allow singleton nodes,
        by default False
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
            mask = torch.logical_and(score <= self.threshold, deg[col] > 1)

        self.removed_edges = data.edge_index[:, mask]
        data.edge_index = data.edge_index[:, ~mask]
        return data

    def __repr__(self) -> str:
        desc = f"threshold={self.threshold}, " +\
            f"allow_singleton={self.allow_singleton}"
        return f'{self.__class__.__name__}({desc})'


class CosinePurification(BaseTransform):
    r"""Graph purification based on cosine similarity of
    connected nodes.

    Note
    ----
    :class:`CosinePurification` is an extension of
    :class:`~greatx.defense.JaccardPurification` for dealing with
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
            mask = torch.logical_and(score <= self.threshold, deg[col] > 1)

        self.removed_edges = data.edge_index[:, mask]
        data.edge_index = data.edge_index[:, ~mask]
        return data

    def __repr__(self) -> str:
        desc = f"threshold={self.threshold}, " +\
            f"allow_singleton={self.allow_singleton}"
        return f'{self.__class__.__name__}({desc})'


class SVDPurification(BaseTransform):
    r"""Graph purification based on low-rank
    Singular Value Decomposition (SVD) reconstruction on
    the adjacency matrix.

    Parameters
    ----------
    K : int, optional
        the top-k largest singular value for reconstruction,
        by default 50
    threshold : float, optional
        threshold to set elements in the reconstructed adjacency
        matrix as zero, by default 0.01
    binaryzation : bool, optional
        whether to binarize the reconstructed adjacency matrix,
        by default False
    remove_edge_index : bool, optional
        whether to remove the :obj:`edge_index` and :obj:`edge_weight`
        int the input :obj:`data` after reconstruction, by default True

    Note
    ----
    We set the reconstructed adjacency matrix as :obj:`adj_t` to
    be compatible with torch_geometric whose :obj:`adj_t`
    denotes the :class:`torch_sparse.SparseTensor`.
    """
    def __init__(self, K: int = 50, threshold: float = 0.01,
                 binaryzation: bool = False, remove_edge_index: bool = True):
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
        adj_matrix = svd(adj_matrix, K=self.K, threshold=self.threshold,
                         binaryzation=self.binaryzation)

        # using transposed matrix instead
        data.adj_t = torch.as_tensor(adj_matrix.A.T, dtype=torch.float,
                                     device=device)
        if self.remove_edge_index:
            del data.edge_index, data.edge_weight
        else:
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
            data.edge_index, data.edge_weight = edge_index.to(
                device), edge_weight.to(device)

        return data

    def __repr__(self) -> str:
        desc = f"K={self.K}, threshold={self.threshold}"
        return f'{self.__class__.__name__}({desc})'


class EigenDecomposition(BaseTransform):
    r"""Graph purification based on low-rank
    Eigen Decomposition reconstruction on
    the adjacency matrix.

    :class:`EigenDecomposition` is similar to
    :class:`~greatx.defense.SVDPurification`

    Parameters
    ----------
    K : int, optional
        the top-k largest singular value for reconstruction,
        by default 50
    normalize : bool, optional
        whether to normalize the input adjacency matrix
    remove_edge_index : bool, optional
        whether to remove the :obj:`edge_index` and :obj:`edge_weight`
        int the input :obj:`data` after reconstruction, by default True

    Note
    ----
    We set the reconstructed adjacency matrix as :obj:`adj_t` to
    be compatible with torch_geometric whose :obj:`adj_t`
    denotes the :class:`torch_sparse.SparseTensor`.
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

        # using transposed matrix instead
        data.adj_t = torch.as_tensor(adj_matrix.T, dtype=torch.float,
                                     device=device)

        if self.remove_edge_index:
            del data.edge_index, data.edge_weight
        else:
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
            data.edge_index, data.edge_weight = edge_index.to(
                device), edge_weight.to(device)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'


class TSVD(BaseTransform):
    r"""Graph purification based on low-rank
    Singular Value Decomposition (SVD) reconstruction on
    the adjacency matrix.

    Parameters
    ----------
    K : int, optional
        the top-k largest singular value for reconstruction,
        by default 50
    threshold : float, optional
        threshold to set elements in the reconstructed adjacency
        matrix as zero, by default 0.01
    binaryzation : bool, optional
        whether to binarize the reconstructed adjacency matrix,
        by default False
    remove_edge_index : bool, optional
        whether to remove the :obj:`edge_index` and :obj:`edge_weight`
        int the input :obj:`data` after reconstruction, by default True

    Note
    ----
    We set the reconstructed adjacency matrix as :obj:`adj_t` to
    be compatible with torch_geometric whose :obj:`adj_t`
    denotes the :class:`torch_sparse.SparseTensor`.
    """
    def __init__(self, K: int = 50, num_channels: int = 5, p: float = 0.1,
                 normalize: bool = True):
        super().__init__()
        self.K = K
        self.p = p
        self.num_channels = num_channels
        self.normalize = normalize

    def __call__(self, data: Data, inplace: bool = True) -> Data:
        if not inplace:
            data = copy(data)

        adjs = self.augmentation(data.edge_index, data.edge_weight,
                                 num_nodes=data.num_nodes)
        adjs = t_svd(adjs, self.K)
        if self.normalize:
            for i in range(self.num_channels):
                adjs[..., i] = dense_gcn_norm(adjs[..., i])

        data.adj_t = adjs
        del data.edge_index, data.edge_weight

        return data

    def augmentation(self, edge_index, edge_weight, num_nodes):
        # using transposed matrix instead
        adj = to_dense_adj(edge_index, edge_weight, num_nodes=num_nodes).t()

        if self.normalize:
            adj = dense_gcn_norm(adj)
        adjs = [adj]

        num_edges = edge_index.size(1)
        device = edge_index.device

        for _ in range(self.num_channels - 1):
            edge_index_remain = dropout_adj(edge_index, p=self.p,
                                            force_undirected=True)[0]
            num_edges_dropped = num_edges - edge_index_remain.size(1)
            random_edges = torch.randint(num_nodes,
                                         size=(2, num_edges_dropped // 2),
                                         device=device)
            random_edges2 = random_edges
            (random_edges2[0], random_edges2[1]) = (random_edges[1],
                                                    random_edges[0])
            # Actually, `random_edges2` and `random_edges` share the
            # same memory I guess the authors of this paper intended to
            # get an undirected version of randomly sampled edges,
            # but they wrote the wrong code.
            # However, once I corrected it, the model perfomance
            # dropped dramatically. I don't know why and just leave it...
            new_edge_index = torch.cat(
                [edge_index_remain, random_edges, random_edges2], dim=1)
            # using transposed matrix instead
            adj = to_dense_adj(new_edge_index, num_nodes=num_nodes).t()
            if self.normalize:
                adj = dense_gcn_norm(adj)
            adjs.append(adj)
        # [num_nodes, num_nodes, num_channels]
        return torch.stack(adjs, dim=-1)

    def __repr__(self) -> str:
        desc = f"K={self.K}, threshold={self.threshold}"
        return f'{self.__class__.__name__}({desc})'


def jaccard_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    intersection = torch.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (torch.count_nonzero(
        A, dim=1) + torch.count_nonzero(B, dim=1) - intersection + 1e-7)
    return J


def svd(adj_matrix: sp.csr_matrix, K: int = 50, threshold: float = 0.01,
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


def t_svd(adjs: torch.Tensor, K: int = 50) -> torch.Tensor:
    print('=== t-SVD: rank={} ==='.format(K))
    adjs = adjs.unsqueeze(-1) if adjs.ndim == 2 else adjs
    n1, n2, n3 = adjs.size()
    xx = torch.complex(torch.empty_like(adjs), torch.empty_like(adjs))
    mat = torch.fft.fft(adjs)

    U, S, V = torch.svd(mat[:, :, 0])
    print("rank_before = {}".format(len(S)))
    S = S.type(torch.complex64)
    if K >= 1:
        S = torch.diag(S[:K])
        xx[:, :, 0] = torch.matmul(torch.matmul(U[:, :K], S), V[:, :K].t())

    halfn3 = round(n3 / 2)
    for i in range(1, halfn3):
        U, S, V = torch.svd(mat[:, :, i])
        S = S.type(torch.complex64)
        if K >= 1:
            S = torch.diag(S[:K])
            xx[:, :, i] = torch.matmul(torch.matmul(U[:, :K], S), V[:, :K].t())

        xx[:, :, n3 - i] = xx[:, :, i].conj()

    if n3 % 2 == 0:
        i = halfn3
        U, S, V = torch.svd(mat[:, :, i])
        S = S.type(torch.complex64)
        if K >= 1:
            S = torch.diag(S[:K])
            xx[:, :, i] = torch.matmul(torch.matmul(U[:, :K], S), V[:, :K].t())

    xx = torch.fft.ifft(xx).real
    print("rank_after = {}".format(K))
    return xx
