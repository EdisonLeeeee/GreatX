import torch
import dgl
from torch import Tensor
from torch.autograd import grad
from torch.nn import init
from tqdm import tqdm
from typing import Optional, Callable

from graphwar.utils import normalize, singleton_mask
from .untargeted_attacker import UntargetedAttacker
from ..surrogate_attacker import SurrogateAttacker


class Metattack(UntargetedAttacker, SurrogateAttacker):
    # Metattack can also conduct feature attack
    _allow_feature_attack = True

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_exists()

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor,
                        unlabeled_nodes: Tensor,
                        lr: float = 0.1,
                        epochs: int = 100,
                        momentum: float = 0.9,
                        lambda_: float = 0.,
                        *,
                        loss: Callable = torch.nn.CrossEntropyLoss(),
                        eps: float = 1.0):

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError(
                "Invalid argument `lambda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)]."
            )

        SurrogateAttacker.setup_surrogate(self, surrogate=surrogate,
                                          loss=loss, eps=eps, freeze=False)

        labeled_nodes = torch.LongTensor(labeled_nodes).to(self.device)
        unlabeled_nodes = torch.LongTensor(unlabeled_nodes).to(self.device)

        self.labeled_nodes = labeled_nodes
        self.unlabeled_nodes = unlabeled_nodes

        self.y_train = self.label[labeled_nodes]
        self.y_self_train = self.estimate_self_training_labels(unlabeled_nodes)
        self.adj = self.graph.adjacency_matrix().to_dense().to(self.device)

        weights = []
        w_velocities = []
        
        for para in self.surrogate.parameters():
            if para.ndim == 2:
                weights.append(torch.zeros_like(para, requires_grad=True))
                w_velocities.append(torch.zeros_like(para))
            else:
                # we do not consider bias terms for simplicity
                pass

        self.weights, self.w_velocities = weights, w_velocities

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_

    def reset(self):
        super().reset()
        self.adj_changes = torch.zeros_like(self.adj)
        self.feat_changes = torch.zeros_like(self.feat)
        return self

    def get_perturbed_adj(self, adj_changes=None):
        adj_changes = self.adj_changes if adj_changes is None else adj_changes
        adj_changes_triu = torch.triu(adj_changes, diagonal=1)
        adj_changes_symm = self.clip(adj_changes_triu + adj_changes_triu.t())
        modified_adj = adj_changes_symm + self.adj
        return modified_adj

    def get_perturbed_feat(self, feat_changes=None):
        feat_changes = self.feat_changes if feat_changes is None else feat_changes
        return self.feat + self.clip(feat_changes)

    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, -1., 1.)
        return clipped_matrix

    def reset_parameters(self):
        for w, wv in zip(self.weights, self.w_velocities):
            init.xavier_uniform_(w)
            init.zeros_(wv)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().requires_grad_()
            self.w_velocities[i] = self.w_velocities[i].detach()

    def forward(self, adj, feat):
        h = feat
        for w in self.weights[:-1]:
            h = adj @ (h @ w)
            h = h.relu()

        return adj @ (h @ self.weights[-1])

    def inner_train(self, adj, feat):
        self.reset_parameters()

        for _ in range(self.epochs):
            out = self(adj, feat)
            loss = self.loss_fn(out[self.labeled_nodes], self.y_train)
            grads = torch.autograd.grad(loss,
                                        self.weights,
                                        create_graph=True)

            self.w_velocities = [
                self.momentum * v + g
                for v, g in zip(self.w_velocities, grads)
            ]

            self.weights = [
                w - self.lr * v for w, v in zip(self.weights, self.w_velocities)
            ]

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

        adj_changes = self.adj_changes
        feat_changes = self.feat_changes
        modified_adj = self.adj
        modified_feat = self.feat

        adj_changes.requires_grad_(bool(structure_attack))
        feat_changes.requires_grad_(bool(feature_attack))

        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            if structure_attack:
                modified_adj = self.get_perturbed_adj(adj_changes)

            if feature_attack:
                modified_feat = self.get_perturbed_feat(feat_changes)

            adj_norm = normalize(modified_adj)
            self.inner_train(adj_norm, modified_feat)

            adj_grad, feat_grad = self._compute_gradients(adj_norm,
                                                          modified_feat)

            adj_grad_score = modified_adj.new_zeros(1)
            feat_grad_score = modified_feat.new_zeros(1)

            with torch.no_grad():
                if structure_attack:
                    adj_grad_score = self.structure_score(modified_adj, adj_grad)

                if feature_attack:
                    feat_grad_score = self.feature_score(modified_feat, feat_grad)

                adj_max, adj_argmax = torch.max(adj_grad_score, dim=0)
                feat_max, feat_argmax = torch.max(feat_grad_score, dim=0)
                
                if adj_max >= feat_max:
                    u, v = divmod(adj_argmax.item(), num_nodes)
                    edge_weight = modified_adj[u, v].data.item()
                    adj_changes[u, v].data.fill_(1 - 2 * edge_weight)
                    adj_changes[v, u].data.fill_(1 - 2 * edge_weight)

                    if edge_weight > 0:
                        self.remove_edge(u, v, it)
                    else:
                        self.add_edge(u, v, it)
                else:
                    u, v = divmod(feat_argmax.item(), num_feats)
                    feat_weight = modified_feat[u, v].data.item()
                    feat_changes[u, v].data.fill_(1 - 2 * feat_weight)
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

    def _compute_gradients(self, modified_adj, modified_feat):

        logit = self(modified_adj, modified_feat) / self.eps

        if self.lambda_ == 1:
            loss = self.loss_fn(logit[self.labeled_nodes], self.y_train)
        elif self.lambda_ == 0.:
            loss = self.loss_fn(logit[self.unlabeled_nodes], self.y_self_train)
        else:
            loss_labeled = self.loss_fn(logit[self.labeled_nodes], self.y_train)
            loss_unlabeled = self.loss_fn(logit[self.unlabeled_nodes], self.y_self_train)
            loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        if self.structure_attack and self.feature_attack:
            return grad(loss, [self.adj_changes, self.feat_changes])

        if self.structure_attack:
            return grad(loss, self.adj_changes)[0], None

        if self.feature_attack:
            return None, grad(loss, self.feat_changes)[0]
