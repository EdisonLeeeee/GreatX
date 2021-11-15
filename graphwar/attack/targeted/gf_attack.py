import torch
import dgl
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from tqdm import tqdm
from torch import Tensor
from typing import Optional

from graphwar.utils import singleton_filter
from .targeted_attacker import TargetedAttacker


class GFAttack(TargetedAttacker):

    def __init__(self, graph: dgl.DGLGraph, K: int = 2, T: int = 128, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        """Initialization of GFAttack.

        Parameters
        ----------
        graph : dgl.DGLGraph
            the DGL graph
        K : int, optional
            the order of graph filter, by default 2
        T : int, optional
            top-T largest eigen-values/vectors selected, by default 128            
        device : str, optional
            the device of the attack running on, by default "cpu"
        seed : Optional[int], optional
            the random seed of reproduce the attack, by default None
        name : Optional[str], optional
            name of the attacker, if None, it would be `__class__.__name__`, by default None

        """
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_exists()

        adj = self.adjacency_matrix
        adj = adj + sp.eye(adj.shape[0], format='csr')
        deg = np.diag(adj.sum(1).A1)
        eig_vals, eig_vec = linalg.eigh(adj.A, deg)
        self.eig_vals = torch.as_tensor(eig_vals, device=self.device, dtype=torch.float32)
        self.eig_vec = torch.as_tensor(eig_vec, device=self.device, dtype=torch.float32)

        feat = self.feat
        self.x_mean = feat.sum(1)  # the author named this as `x_mean`, I don't understand why not `x_sum`

        self.K = K
        self.T = T

    def get_candidate_edges(self):
        target = self.target
        N = self.num_nodes
        nodes_set = set(range(N)) - set([target])

        if self.direct_attack:
            influencers = [target]
            row = np.repeat(influencers, N - 1)
            col = list(nodes_set)

        else:
            influencers = self.adjacency_matrix[target].indices
            row = np.repeat(influencers, N - 2)
            col = np.hstack([list(nodes_set - set([infl])) for infl in influencers])
        candidate_edges = np.stack([row, col], axis=1)

        if not self._allow_singleton:
            candidate_edges = singleton_filter(candidate_edges,
                                               self.adjacency_matrix)

        return candidate_edges

    def attack(self,
               target, *,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               ll_constraint=False,
               ll_cutoff=0.004,
               disable=False):

        super().attack(target, target_label=None, num_budgets=num_budgets,
                       direct_attack=direct_attack, structure_attack=structure_attack,
                       feature_attack=feature_attack)

        candidate_edges = self.get_candidate_edges()

        score = self._structure_score(self.adjacency_matrix,
                                      self.x_mean,
                                      self.eig_vals,
                                      self.eig_vec,
                                      candidate_edges,
                                      K=self.K,
                                      T=self.T,
                                      method="nosum")

        topk = torch.topk(score, k=self.num_budgets).indices.cpu()
        edges = candidate_edges[topk].reshape(-1, 2)
        edge_weights = self.adjacency_matrix[edges[:, 0], edges[:, 1]].A1

        for it, edge_weight in tqdm(enumerate(edge_weights),
                                    desc='Peturbing Graph',
                                    disable=disable):
            u, v = edges[it]
            if edge_weight > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)
        return self

    @staticmethod
    def _structure_score(A: sp.csr_matrix,
                         x_mean: Tensor,
                         eig_vals: Tensor,
                         eig_vec: Tensor,
                         candidate_edges: np.ndarray,
                         K: int,
                         T: int,
                         method: str = "nosum"):
        """Calculate the score of potential edges as formulated in paper.

        Parameters
        ----------
        A : sp.csr_matrix
            the graph adjacency matrix
        x_mean : Tensor
        eig_vals : Tensor
            the eigen value
        eig_vec : Tensor
            the eigen vector
        candidate_edges : np.ndarray
            the candidate_edges to be selected
        K : int
            The order of graph filter K.
        T : int
            Selecting the Top-T largest eigen-values/vectors.
        method : str, optional
            "sum" or "nosum"
            Indicates the score are calculated from which loss as in Equation (8) or Equation (12).
            "nosum" denotes Equation (8), where the loss is derived from Graph Convolutional Networks,
            "sum" denotes Equation (12), where the loss is derived from Sampling-based Graph Embedding Methods., by default "nosum"

        Returns
        -------
        Tensor
            Scores for potential edges.
        """

        assert method in ['sum', 'nosum']

        D_min = A.sum(1).A1.min() + 1  # `+1` for the added selfloop
        score = []
        for (u, v) in candidate_edges:
            eig_vals_res = (1 - 2 * A[(u, v)]) * (2 * eig_vec[u] * eig_vec[v] - eig_vals *
                                                  ((eig_vec[u]).square() + (eig_vec[v]).square()))
            eig_vals_res = eig_vals + eig_vals_res

            if method == "sum":
                if K == 1:
                    eig_vals_res = (eig_vals_res / K).abs().mul(1. / D_min)
                else:
                    for itr in range(1, K):
                        eig_vals_res = eig_vals_res + eig_vals_res.pow(itr + 1)
                    eig_vals_res = (eig_vals_res / K).abs().mul(1. / D_min)
            else:
                eig_vals_res = (eig_vals_res + 1.).square().pow(K)

            least_t = torch.topk(-eig_vals_res, k=T).indices
            eig_vals_k_sum = eig_vals_res[least_t].sum()
            u_k = eig_vec[:, least_t].t()
            u_x_mean = u_k @ x_mean
            score_u_v = eig_vals_k_sum * torch.square(torch.linalg.norm(u_x_mean))
            score.append(score_u_v.item())
        return torch.as_tensor(score)
