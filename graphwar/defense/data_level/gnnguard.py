import torch
import torch.nn.functional as F

from sklearn.preprocessing import normalize
from graphwar import Config

_FEATURE = Config.feat
_EDGE_WEIGHT = Config.edge_weight


class GNNGUARD(torch.nn.Module):

    def __init__(self, threshold: float = 0.1, add_self_loop: bool = False):
        super().__init__()
        self.threshold = threshold
        self.add_self_loop = add_self_loop

    def forward(self, g, feat=None):
        g = g.remove_self_loop()
        if feat is None:
            feat = g.ndata.get(_FEATURE, None)
            if feat is None:
                raise ValueError(
                    f"The node feature matrix is not specified, please add argument '{_FEATURE}' during forward or specify `g.ndata['{_FEATURE}']=feat`")

        row, col = g.edges()
        A = feat[row]
        B = feat[col]
        att_score = F.cosine_similarity(A, B)
        att_score[att_score < self.threshold] = 0.
        adj = g.adjacency_matrix(scipy_fmt='csr')
        row, col = row.cpu().tolist(), col.cpu().tolist()
        adj[row, col] = att_score.cpu().tolist()
        adj = normalize(adj, axis=1, norm='l1')
        att_score_norm = torch.tensor(adj[row, col]).to(feat).view(-1)

        if self.add_self_loop:
            degree = (adj != 0).sum(1).A1
            self_weight = torch.tensor(1.0 / (degree + 1)).to(feat)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            g = g.add_self_loop()

        att_score_norm = att_score_norm.exp()
        g.edata[_EDGE_WEIGHT] = att_score_norm

        return g

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, add_self_loop={self.add_self_loop}"
