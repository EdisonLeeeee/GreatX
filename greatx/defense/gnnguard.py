import warnings

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

EPS = 1e-10


class GNNGUARD(torch.nn.Module):
    r"""Implementation of GNNGUARD
    from the `"GNNGUARD:
    Defending Graph Neural Networks against Adversarial Attacks"
    <https://arxiv.org/abs/2006.08149>`_ paper (NeurIPS'20)

    Parameters
    ----------
    threshold : float, optional
        threshold for removing edges based on
        attention scores, by default 0.1
    add_self_loops : bool, optional
        whether to add self-loops to the input graph,
        by default False
    """
    def __init__(self, threshold: float = 0.1, add_self_loops: bool = False):
        super().__init__()
        self.threshold = threshold
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if edge_weight is not None:
            warnings.warn("`edge_weight` is supported in GNNGUARD "
                          "and will be ignored for computation.")

        row, col = edge_index
        A, B = x[row], x[col]
        att_score = F.cosine_similarity(A, B)
        mask = att_score >= self.threshold
        edge_index = edge_index[:, mask]
        att_score = att_score[mask]

        row, col = edge_index
        row_sum = scatter(att_score, col, dim_size=x.size(0))
        att_score_norm = att_score / (row_sum[row] + EPS)

        if self.add_self_loops:
            degree = scatter(torch.ones_like(att_score_norm), col,
                             dim_size=x.size(0))
            self_weight = 1.0 / (degree + 1)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            loop_index = torch.arange(0, x.size(0), dtype=torch.long,
                                      device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        att_score_norm = att_score_norm.exp()
        return edge_index, att_score_norm

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"
