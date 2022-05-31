from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
from tqdm import tqdm

from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphwar.surrogate import Surrogate
from graphwar.utils import singleton_mask
from graphwar.functional import to_dense_adj


class FGAttack(UntargetedAttacker, Surrogate):
    # FGAttack can conduct feature attack
    _allow_feature_attack: bool = True

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        victim_nodes: Tensor,
                        victim_labels: Optional[Tensor] = None, *,
                        eps: float = 1.0):

        Surrogate.setup_surrogate(self, surrogate=surrogate,
                                  eps=eps, freeze=True)

        self.victim_nodes = victim_nodes.to(self.device)
        if victim_labels is None:
            victim_labels = self.label[victim_nodes]
        self.victim_labels = victim_labels.to(self.device)
        return self

    def reset(self):
        super().reset()
        self.modified_adj = to_dense_adj(self.edge_index,
                                         self.edge_weight,
                                         num_nodes=self.num_nodes).to(self.device)
        self.modified_feat = self.feat.clone()
        return self

    def attack(self,
               num_budgets=0.05, *,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        if feature_attack:
            self._check_feature_matrix_binary()

        modified_adj = self.modified_adj
        modified_feat = self.modified_feat
        modified_adj.requires_grad_(bool(structure_attack))
        modified_feat.requires_grad_(bool(feature_attack))

        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing graph...',
                       disable=disable):

            adj_grad, feat_grad = self.compute_gradients(modified_adj,
                                                         modified_feat,
                                                         self.victim_nodes,
                                                         self.victim_labels)

            adj_grad_score = modified_adj.new_zeros(1)
            feat_grad_score = modified_feat.new_zeros(1)

            with torch.no_grad():
                if structure_attack:
                    adj_grad_score = self.structure_score(
                        modified_adj, adj_grad)

                if feature_attack:
                    feat_grad_score = self.feature_score(
                        modified_feat, feat_grad)

                adj_max, adj_argmax = torch.max(adj_grad_score, dim=0)
                feat_max, feat_argmax = torch.max(feat_grad_score, dim=0)
                if adj_max >= feat_max:
                    u, v = divmod(adj_argmax.item(), num_nodes)
                    edge_weight = modified_adj[u, v].data.item()
                    modified_adj[u, v].data.fill_(1 - edge_weight)
                    modified_adj[v, u].data.fill_(1 - edge_weight)

                    if edge_weight > 0:
                        self.remove_edge(u, v, it)
                    else:
                        self.add_edge(u, v, it)
                else:
                    u, v = divmod(feat_argmax.item(), num_feats)
                    feat_weight = modified_feat[u, v].data
                    modified_feat[u, v].data.fill_(1 - feat_weight)
                    if feat_weight > 0:
                        self.remove_feat(u, v, it)
                    else:
                        self.add_feat(u, v, it)

        return self

    def structure_score(self, modified_adj, adj_grad):
        score = adj_grad * (1 - 2 * modified_adj)
        score -= score.min()
        score = torch.triu(score, diagonal=1)
        if not self._allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            score *= singleton_mask(modified_adj)
        return score.view(-1)

    def feature_score(self, modified_feat, feat_grad):
        score = feat_grad * (1 - 2 * modified_feat)
        score -= score.min()
        return score.view(-1)

    def compute_gradients(self, modified_adj, modified_feat, victim_nodes, victim_labels):

        logit = self.surrogate(modified_feat, modified_adj)[
            victim_nodes] / self.eps
        loss = F.cross_entropy(logit, victim_labels)

        if self.structure_attack and self.feature_attack:
            return grad(loss, [modified_adj, modified_feat], create_graph=False)

        if self.structure_attack:
            return grad(loss, modified_adj, create_graph=False)[0], None

        if self.feature_attack:
            return None, grad(loss, modified_feat, create_graph=False)[0]
