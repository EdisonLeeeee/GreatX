import torch
import dgl
import scipy.sparse as sp
from graphwar import Config

_FEATURE = Config.feat
_EDGE_WEIGHT = Config.edge_weight


def jaccard_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    intersection = torch.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (torch.count_nonzero(A, dim=1) + torch.count_nonzero(B, dim=1) + intersection + 1e-7)
    return J


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    inner_product = (A * B).sum(1)
    C = inner_product / (torch.norm(A, 2, 1) * torch.norm(B, 2, 1) + 1e-7)
    return C


class JaccardPurification(torch.nn.Module):

    def __init__(self, threshold=0., allow_singleton=False):
        # TODO: add percentage purification
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton

    def forward(self, g, feat=None):

        g = g.local_var()
        if feat is None:
            feat = g.ndata.get(_FEATURE, None)
            if feat is None:
                raise ValueError(f"The node feature matrix is not spefified, please add argument `feat` during forward or specify `g.ndata['{_FEATURE}']=feat`")
        row, col = g.edges()
        A = feat[row]
        B = feat[col]
        score = jaccard_similarity(A, B)
        deg = g.in_degrees()

        if self.allow_singleton:
            condition = score <= self.threshold
        else:
            condition = torch.logical_and(score <= self.threshold, deg[col] > 1)

        e_id = torch.where(condition)[0]
        g.remove_edges(e_id)

        self.edges = torch.stack([row[e_id], col[e_id]], dim=0)
        return g

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


class CosinePurification(torch.nn.Module):

    def __init__(self, threshold=0., allow_singleton=False):
        # TODO: add percentage purification
        super().__init__()
        self.threshold = threshold
        self.allow_singleton = allow_singleton

    def forward(self, g, feat=None):
        g = g.local_var()
        if feat is None:
            feat = g.ndata.get(_FEATURE, None)
            if feat is None:
                raise ValueError(f"The node feature matrix is not spefified, please add argument `feat` during forward or specify `g.ndata['{_FEATURE}']=feat`")

        row, col = g.edges()
        A = feat[row]
        B = feat[col]
        score = cosine_similarity(A, B)

        deg = g.in_degrees()

        if self.allow_singleton:
            condition = score <= self.threshold
        else:
            condition = torch.logical_and(score <= self.threshold, deg[col] > 1)

        e_id = torch.where(condition)[0]
        g.remove_edges(e_id)

        self.edges = torch.stack([row[e_id], col[e_id]], dim=0)
        return g

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"


def svd(adj_matrix, k=50, threshold=0.01, binarize=False):
    adj_matrix = adj_matrix.asfptype()

    U, S, V = sp.linalg.svds(adj_matrix, k=k)
    adj_matrix = (U * S) @ V

    if threshold is not None:
        # sparsification
        adj_matrix[adj_matrix <= threshold] = 0.

    adj_matrix = sp.csr_matrix(adj_matrix)

    if binarize:
        # TODO
        adj_matrix.data[adj_matrix.data > 0] = 1.0

    return adj_matrix


class SVDPurification(torch.nn.Module):

    def __init__(self, k=50, threshold=0.01, binarize=False):
        # TODO: add percentage purification
        super().__init__()
        self.k = k
        self.threshold = threshold
        self.binarize = binarize

    def forward(self, g):
        device = g.device
        adj_matrix = g.adjacency_matrix(scipy_fmt='csr')
        adj_matrix = svd(adj_matrix, k=self.k,
                         threshold=self.threshold,
                         binarize=self.binarize)

        row, col = adj_matrix.nonzero()

        defense_g = dgl.graph((row, col), device=device)
        defense_g.ndata.update(g.ndata)
        defense_g.edata.update(g.edata)

        if not self.binarize:
            defense_g.edata[_EDGE_WEIGHT] = torch.as_tensor(adj_matrix.data,
                                                            device=device, dtype=torch.float32)

        return defense_g
