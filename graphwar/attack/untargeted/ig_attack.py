from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch_geometric.data import Data
from tqdm import tqdm

from graphwar.surrogate import Surrogate
from graphwar.utils import singleton_mask
from graphwar.functional import to_dense_adj
from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker


class IGAttack(UntargetedAttacker, Surrogate):
    r"""Implementation of `IG-FGSM` attack from the: 
    `"Adversarial Examples on Graph Data: Deep Insights 
    into Attack and Defense" 
    <https://arxiv.org/abs/1903.01610>`_ paper (IJCAI'19)

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be :obj:`__class__.__name__`, 
        by default None
    kwargs : additional arguments of :class:`graphwar.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`       

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.untargeted import IGAttack
    >>> attacker = IGAttack(data)
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker.reset()
    >>> attacker.attack(0.05) # attack with 0.05% of edge perturbations
    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack   

    Note
    ----
    * In the paper, `IG-FGSM` attack was implemented for targeted attack, we adapt the codes for the non-targeted attack here.    
    * Please remember to call :meth:`reset` before each attack. 
    """
    # IGAttack can conduct feature attack
    _allow_feature_attack: bool = True

    def __init__(self, data: Data, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(data=data, device=device, seed=seed, name=name, **kwargs)

        num_nodes, num_feats = self.num_nodes, self.num_feats
        self.nodes_set = set(range(num_nodes))
        self.feats_list = list(range(num_feats))
        self.adj = to_dense_adj(self.edge_index,
                                self.edge_weight,
                                num_nodes=self.num_nodes).to(self.device)

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
                          desc='Computing link importance...',
                          disable=disable):
            ###### Compute integrated gradients for removing edges ######
            adj_diff = adj - baseline_remove
            adj_step = baseline_remove + alpha * adj_diff
            adj_step.requires_grad_()

            gradients += self.compute_structure_gradients(
                adj_step, feat, victim_nodes, victim_labels)

            ###### Compute integrated gradients for adding edges ######
            adj_diff = baseline_add - adj
            adj_step = baseline_add - alpha * adj_diff
            adj_step.requires_grad_()

            gradients += self.compute_structure_gradients(
                adj_step, feat, victim_nodes, victim_labels)

        return gradients

    def get_feature_importance(self, steps, victim_nodes, victim_labels, disable=False):

        adj = self.adj
        feat = self.feat

        baseline_add = torch.ones_like(feat)
        baseline_remove = torch.zeros_like(feat)

        gradients = torch.zeros_like(feat)

        for alpha in tqdm(torch.linspace(0., 1.0, steps + 1),
                          desc='Computing feature importance...',
                          disable=disable):
            ###### Compute integrated gradients for removing features ######
            feat_diff = feat - baseline_remove
            feat_step = baseline_remove + alpha * feat_diff
            feat_step.requires_grad_()

            gradients += self.compute_feature_gradients(
                adj, feat_step, victim_nodes, victim_labels)

            ###### Compute integrated gradients for adding features ######
            feat_diff = baseline_add - feat
            feat_step = baseline_add - alpha * feat_diff
            feat_step.requires_grad_()

            gradients += self.compute_feature_gradients(
                adj, feat_step, victim_nodes, victim_labels)

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

    def compute_structure_gradients(self, adj_step, feat, victim_nodes, victim_labels):

        logit = self.surrogate(feat, adj_step)[victim_nodes] / self.eps
        loss = F.cross_entropy(logit, victim_labels)
        return grad(loss, adj_step, create_graph=False)[0]

    def compute_feature_gradients(self, adj, feat_step, victim_nodes, victim_labels):

        logit = self.surrogate(feat_step, feat_step)[victim_nodes] / self.eps
        loss = F.cross_entropy(logit, victim_labels)
        return grad(loss, feat_step, create_graph=False)[0]
