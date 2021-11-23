import torch
from graphwar import Info
from collections import namedtuple


_FEATURE = Info.feat

namedtuple_g_edges = namedtuple('NamedTuple', ['g', 'edges'])


def jaccard_similarity(A, B):
    intersection = torch.count_nonzero(A * B, axis=1)
    J = intersection * 1.0 / (torch.count_nonzero(A, dim=1) + torch.count_nonzero(B, dim=1) + intersection + 1e-7)
    return J


def cosine_similarity(A, B):
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

        return namedtuple_g_edges(g=g, edges=torch.stack([row[e_id], col[e_id]], dim=0))

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
        return namedtuple_g_edges(g=g, edges=torch.stack([row[e_id], col[e_id]], dim=0))

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, allow_singleton={self.allow_singleton}"
