from typing import Callable, Optional

import dgl
import torch
from torch import Tensor
from torch.autograd import grad
from tqdm import tqdm

from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphwar.surrogater import Surrogater
from graphwar.utils import normalize, singleton_mask


class IGAttack(UntargetedAttacker, Surrogater):
    # IGAttack can conduct feature attack
    _allow_feature_attack = True

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_exists()
        num_nodes, num_feats = self.num_nodes, self.num_feats
        self.nodes_set = set(range(num_nodes))
        self.feats_list = list(range(num_feats))
        self.adj = self.graph.add_self_loop().adjacency_matrix().to_dense().to(self.device)
        self.adj_norm = normalize(self.adj)

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        victim_nodes: Tensor,
                        victim_labels: Optional[Tensor] = None, *,
                        loss: Callable = torch.nn.CrossEntropyLoss(),
                        eps: float = 1.0):

        Surrogater.setup_surrogate(self, surrogate=surrogate,
                                   loss=loss, eps=eps, freeze=True)

        self.victim_nodes = victim_nodes.to(self.device)
        if victim_labels is None:
            self._check_node_label_exists()
            victim_labels = self.label[victim_nodes]
        self.victim_labels = victim_labels.to(self.device)
        return self

    def attack(self,
               num_budgets=0.05, *,
               steps=20,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        if structure_attack:
            link_importance = self.get_link_importance(steps, self.victim_nodes,
                                                       self.victim_labels, disable=disable)
            adj_score = self.structure_score(self.adj, link_importance)

        if feature_attack:
            self._check_feature_matrix_binary()
            feature_importance = self.get_feature_importance(steps, self.victim_nodes,
                                                             self.victim_labels, disable=disable)
            feat_score = self.feature_score(self.feat, feature_importance)

        if structure_attack and not feature_attack:
            indices = torch.topk(adj_score, k=self.num_budgets).indices

            for it, index in enumerate(indices.tolist()):
                u, v = divmod(index, self.num_nodes)
                edge_weight = self.adj[u, v].data.item()

                if edge_weight > 0:
                    self.remove_edge(u, v, it)
                else:
                    self.add_edge(u, v, it)

        elif feature_attack and not structure_attack:
            indices = torch.topk(feat_score, k=self.num_budgets).indices

            for it, index in enumerate(indices.tolist()):
                u, v = divmod(index, self.num_feats)
                feat_weight = self.feat[u, v].data.item()

                if feat_weight > 0:
                    self.remove_feat(u, v, it)
                else:
                    self.add_feat(u, v, it)
        else:
            # both attacks are conducted
            score = torch.cat([adj_score, feat_score])
            indices = torch.topk(score, k=self.num_budgets).indices
            boundary = adj_score.size(0)

            for it, index in enumerate(indices.tolist()):
                if index < boundary:
                    u, v = divmod(index, self.num_nodes)
                    edge_weight = self.adj[u, v].data.item()

                    if edge_weight > 0:
                        self.remove_edge(u, v, it)
                    else:
                        self.add_edge(u, v, it)
                else:
                    u, v = divmod(index - boundary, self.num_feats)
                    feat_weight = self.feat[u, v].data.item()

                    if feat_weight > 0:
                        self.remove_feat(u, v, it)
                    else:
                        self.add_feat(u, v, it)

        return self

    def get_link_importance(self, steps, victim_nodes, victim_labels, disable=False):

        adj = self.adj
        feat = self.feat

        baseline_add = torch.ones_like(adj)
        baseline_remove = torch.zeros_like(adj)

        gradients = torch.zeros_like(adj)

        for alpha in tqdm(torch.linspace(0., 1.0, steps + 1),
                          desc='Computing link importance',
                          disable=disable):
            ###### Compute integrated gradients for removing edges ######
            adj_diff = adj - baseline_remove
            adj_step = baseline_remove + alpha * adj_diff
            adj_step.requires_grad_()

            gradients += self._compute_structure_gradients(adj_step, feat, victim_nodes, victim_labels)

            ###### Compute integrated gradients for adding edges ######
            adj_diff = baseline_add - adj
            adj_step = baseline_add - alpha * adj_diff
            adj_step.requires_grad_()

            gradients += self._compute_structure_gradients(adj_step, feat, victim_nodes, victim_labels)

        return gradients

    def get_feature_importance(self, steps, victim_nodes, victim_labels, disable=False):

        adj = self.adj_norm
        feat = self.feat

        baseline_add = torch.ones_like(feat)
        baseline_remove = torch.zeros_like(feat)

        gradients = torch.zeros_like(feat)

        for alpha in tqdm(torch.linspace(0., 1.0, steps + 1),
                          desc='Computing feature importance',
                          disable=disable):
            ###### Compute integrated gradients for removing features ######
            feat_diff = feat - baseline_remove
            feat_step = baseline_remove + alpha * feat_diff
            feat_step.requires_grad_()

            gradients += self._compute_feature_gradients(adj, feat_step, victim_nodes, victim_labels)

            ###### Compute integrated gradients for adding features ######
            feat_diff = baseline_add - feat
            feat_step = baseline_add - alpha * feat_diff
            feat_step.requires_grad_()

            gradients += self._compute_feature_gradients(adj, feat_step, victim_nodes, victim_labels)

        return gradients

    def structure_score(self, adj, adj_grad):
        adj_grad = adj_grad + adj_grad.t()
        score = adj_grad * (1 - 2 * adj)
        score -= score.min()
        score = torch.triu(score, diagonal=1)
        if not self._allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            score *= singleton_mask(adj)
        return score.view(-1)

    def feature_score(self, feat, feat_grad):
        score = feat_grad * (1 - 2 * feat)
        score -= score.min()
        return score.view(-1)

    def _compute_structure_gradients(self, adj_step, feat, victim_nodes, victim_labels):

        adj_norm = normalize(adj_step)
        logit = self.surrogate(adj_norm, feat)[victim_nodes] / self.eps
        loss = self.loss_fn(logit, victim_labels)
        return grad(loss, adj_step, create_graph=False)[0]

    def _compute_feature_gradients(self, adj_norm, feat_step, victim_nodes, victim_labels):

        logit = self.surrogate(adj_norm, feat_step)[victim_nodes] / self.eps
        loss = self.loss_fn(logit, victim_labels)
        return grad(loss, feat_step, create_graph=False)[0]
