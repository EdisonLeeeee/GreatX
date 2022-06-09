import torch
import torch.nn.functional as F

from sklearn.preprocessing import normalize
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops


class GNNGUARD(torch.nn.Module):
    r"""Implementation of GNNGUARD
    from the `"GNNGUARD: 
    Defending Graph Neural Networks against Adversarial Attacks"
    <https://arxiv.org/abs/2006.08149>`_ paper (NeurIPS'20)

    Parameters
    ----------
    threshold : float, optional
        threshold for removing edges based on attention scores, by default 0.1
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default False    
    """

    def __init__(self, threshold: float = 0.1, add_self_loops: bool = False):
        super().__init__()
        self.threshold = threshold
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):

        row, col = edge_index
        A = x[row]
        B = x[col]
        att_score = F.cosine_similarity(A, B)
        att_score[att_score < self.threshold] = 0.
        adj_matrix = to_scipy_sparse_matrix(edge_index, att_score.detach())
        adj_matrix = normalize(adj_matrix, axis=1, norm='l1')
        row, col = edge_index.tolist()
        att_score_norm = torch.tensor(adj_matrix[row, col]).to(x).view(-1)

        if self.add_self_loops:
            degree = adj_matrix.getnnz(axis=1)
            self_weight = torch.tensor(1.0 / (degree + 1)).to(x)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            edge_index, _ = add_self_loops(edge_index)

        att_score_norm = att_score_norm.exp()

        return edge_index, att_score_norm

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, add_self_loops={self.add_self_loops}"
