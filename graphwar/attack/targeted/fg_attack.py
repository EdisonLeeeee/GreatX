from typing import Optional

import dgl
import torch
from torch.autograd import grad
from tqdm import tqdm

from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.surrogater import Surrogater
from graphwar.functional import normalize
from graphwar.utils import singleton_mask


class FGAttack(TargetedAttacker, Surrogater):
    # FGAttack can conduct feature attack
    _allow_feature_attack = True
    # FGAttack can not ensure there are no singleton nodes
    _allow_singleton: bool = True

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_exists()

    def reset(self):
        super().reset()
        self.modified_adj = self.graph.add_self_loop().adjacency_matrix().to_dense().to(self.device)
        self.modified_feat = self.feat.clone()
        return self

    def attack(self,
               target, *,
               target_label=None,
               num_budgets=None,
               direct_attack=True,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(target, target_label, num_budgets=num_budgets,
                       direct_attack=direct_attack, structure_attack=structure_attack,
                       feature_attack=feature_attack)

        if feature_attack:
            self._check_feature_matrix_binary()

        if target_label is None:
            assert self.target_label is not None, "please specify argument `target_label` as the node label does not exist."
            target_label = self.target_label.view(-1)
        else:
            target_label = torch.as_tensor(target_label, device=self.device, dtype=torch.long).view(-1)

        modified_adj = self.modified_adj
        modified_feat = self.modified_feat
        modified_adj.requires_grad_(bool(structure_attack))
        modified_feat.requires_grad_(bool(feature_attack))

        target = torch.as_tensor(target, device=self.device, dtype=torch.long)
        target_label = torch.as_tensor(target_label, device=self.device, dtype=torch.long).view(-1)
        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            adj_grad, feat_grad = self._compute_gradients(modified_adj,
                                                          modified_feat,
                                                          target, target_label)

            adj_grad_score = modified_adj.new_zeros(1)
            feat_grad_score = modified_feat.new_zeros(1)

            with torch.no_grad():
                if structure_attack:
                    adj_grad_score = self.structure_score(modified_adj,
                                                          adj_grad,
                                                          target)

                if feature_attack:
                    feat_grad_score = self.feature_score(modified_feat,
                                                         feat_grad,
                                                         target)

                adj_max, adj_argmax = torch.max(adj_grad_score, dim=0)
                feat_max, feat_argmax = torch.max(feat_grad_score, dim=0)
                if adj_max >= feat_max:
                    u, v = divmod(adj_argmax.item(), num_nodes)
                    if self.direct_attack:
                        u = target.item()
                    edge_weight = modified_adj[u, v].data.item()
                    modified_adj[u, v].data.fill_(1 - edge_weight)
                    modified_adj[v, u].data.fill_(1 - edge_weight)

                    assert self.is_legal_edge(u, v)
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

    def structure_score(self, modified_adj, adj_grad, target):
        if self.direct_attack:
            score = adj_grad[target] * (1 - 2 * modified_adj[target])
            score -= score.min()
            if not self._allow_singleton:
                score *= singleton_mask(modified_adj)[target]
            # make sure the targeted node would not be selected
            score[target] = -1
        else:
            score = adj_grad * (1 - 2 * modified_adj)
            score -= score.min()
            if not self._allow_singleton:
                score *= singleton_mask(modified_adj)
            score = torch.triu(score, diagonal=1)
            # make sure the targeted node and its neighbors would not be selected
            score[target] = -1
            score[:, target] = -1
        return score.view(-1)

    def feature_score(self, modified_feat, feat_grad, target):
        if self.direct_attack:
            score = feat_grad[target] * (1 - 2 * modified_feat[target])
        else:
            score = feat_grad * (1 - 2 * modified_feat)

        score -= score.min()
        # make sure the targeted node would not be selected
        score[target] = -1
        return score.view(-1)

    def _compute_gradients(self, modified_adj, modified_feat, target, target_label):

        adj_norm = normalize(modified_adj)
        logit = self.surrogate(adj_norm, modified_feat)[target].view(1, -1) / self.eps
        loss = self.loss_fn(logit, target_label)

        if self.structure_attack and self.feature_attack:
            return grad(loss, [modified_adj, modified_feat], create_graph=False)

        if self.structure_attack:
            return grad(loss, modified_adj, create_graph=False)[0], None

        if self.feature_attack:
            return None, grad(loss, modified_feat, create_graph=False)[0]
