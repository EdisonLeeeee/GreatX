from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import grad
from torch_geometric.data import Data

from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.surrogate import Surrogate
from graphwar.utils import singleton_filter
from graphwar.functional import to_dense_adj


class IGAttack(TargetedAttacker, Surrogate):
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

    >>> from graphwar.attack.targeted import IGAttack
    >>> attacker = IGAttack(data)
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker.reset()
    >>> attacker.attack(target=1) # attacking target node `1` with default budget set as node degree

    >>> attacker.reset()
    >>> attacker.attack(target=1, num_budgets=1) # attacking target node `1` with budget set as 1

    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack        

    Note
    ----
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

    def attack(self,
               target, *,
               target_label=None,
               num_budgets=None,
               steps=20,
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
            target_label = torch.as_tensor(
                target_label, device=self.device, dtype=torch.long).view(-1)

        if structure_attack:
            candidate_edges = self.get_candidate_edges()
            link_importance, edge_indicator = self.get_link_importance(
                candidate_edges, steps, target, target_label, disable=disable)

        if feature_attack:
            candidate_feats = self.get_candidate_features()
            feature_importance, feat_indicator = self.get_feature_importance(
                candidate_feats, steps, target, target_label, disable=disable)

        if structure_attack and not feature_attack:
            indices = torch.topk(link_importance, k=self.num_budgets).indices
            edge_indicator = edge_indicator[indices]
            link_selected = candidate_edges[indices]

            for u, v in link_selected[~edge_indicator].tolist():
                self.add_edge(u, v)

            for u, v in link_selected[edge_indicator].tolist():
                self.remove_edge(u, v)

        elif feature_attack and not structure_attack:
            indices = torch.topk(feature_importance,
                                 k=self.num_budgets).indices
            feat_indicator = feat_indicator[indices]
            feature_selected = candidate_feats[indices]

            for u, v in feature_selected[feat_indicator].tolist():
                self.remove_feat(u, v)

            for u, v in feature_selected[~feat_indicator].tolist():
                self.add_feat(u, v)
        else:
            # both attacks are conducted
            importance = torch.cat([link_importance, feature_importance])
            indices = torch.topk(importance, k=self.num_budgets).indices

            boundary = link_importance.size(0)

            link_indices = indices[indices < boundary]
            edge_indicator = edge_indicator[link_indices]
            link_selected = candidate_edges[link_indices]

            feat_indices = indices[indices > boundary] - boundary
            feat_indicator = feat_indicator[feat_indices]
            feature_selected = candidate_feats[feat_indices]

            for u, v in link_selected[~edge_indicator].tolist():
                self.add_edge(u, v)

            for u, v in link_selected[edge_indicator].tolist():
                self.remove_edge(u, v)

            for u, v in feature_selected[feat_indicator].tolist():
                self.remove_feat(u, v)

            for u, v in feature_selected[~feat_indicator].tolist():
                self.add_feat(u, v)

        return self

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
            col = np.hstack([list(nodes_set - set([infl]))
                            for infl in influencers])

        candidate_edges = np.stack([row, col], axis=1)
        if not self._allow_singleton:
            candidate_edges = singleton_filter(candidate_edges,
                                               self.adjacency_matrix)
        candidate_edges = torch.as_tensor(
            candidate_edges, dtype=torch.long, device=self.device)
        return candidate_edges

    def get_candidate_features(self):
        num_feats = self.num_feats
        target = self.target

        if self.direct_attack:
            influencers = [target]
            candidate_feats = np.column_stack(
                (np.tile(target, num_feats), self.feats_list))
        else:
            influencers = self.adjacency_matrix[target].indices
            candidate_feats = np.row_stack([
                np.column_stack((np.tile(infl, num_feats), self.feats_list))
                for infl in influencers
            ])

        candidate_feats = torch.as_tensor(
            candidate_feats, dtype=torch.long, device=self.device)
        return candidate_feats

    def get_link_importance(self, candidates, steps, target, target_label, disable=False):

        adj = self.adj
        feat = self.feat
        mask = (candidates[:, 0], candidates[:, 1])

        baseline_add = adj.clone()
        baseline_add[mask] = 1.0
        # baseline_add[mask[::-1]] = 1.0

        baseline_remove = adj.clone()
        baseline_remove[mask] = 0.0
        # baseline_remove[mask[::-1]] = 0.0

        edge_indicator = adj[mask] > 0

        edges = candidates[edge_indicator]
        non_edges = candidates[~edge_indicator]

        edge_gradients = adj.new_zeros(edges.size(0))
        non_edge_gradients = adj.new_zeros(non_edges.size(0))

        for alpha in tqdm(torch.linspace(0., 1.0, steps + 1),
                          desc='Computing link importance...',
                          disable=disable):
            ###### Compute integrated gradients for removing edges ######
            adj_diff = adj - baseline_remove
            adj_step = baseline_remove + alpha * adj_diff
            adj_step.requires_grad_()

            gradients = self.compute_structure_gradients(
                feat, adj_step, target, target_label)
            edge_gradients += gradients[edges[:, 0], edges[:, 1]]

            ###### Compute integrated gradients for adding edges ######
            adj_diff = baseline_add - adj
            adj_step = baseline_add - alpha * adj_diff
            adj_step.requires_grad_()

            gradients = self.compute_structure_gradients(
                feat, adj_step, target, target_label)
            non_edge_gradients += gradients[non_edges[:, 0], non_edges[:, 1]]

        integrated_grads = adj.new_zeros(edge_indicator.size(0))
        integrated_grads[edge_indicator] = edge_gradients
        integrated_grads[~edge_indicator] = non_edge_gradients

        return integrated_grads, edge_indicator

    def get_feature_importance(self, candidates, steps, target, target_label, disable=False):

        adj = self.adj
        feat = self.feat
        mask = (candidates[:, 0], candidates[:, 1])

        baseline_add = feat.clone()
        baseline_add[mask] = 1.0
        baseline_remove = feat.clone()
        baseline_remove[mask] = 0.0
        feat_indicator = feat[mask] > 0

        features = candidates[feat_indicator]
        non_features = candidates[~feat_indicator]

        feat_gradients = feat.new_zeros(features.size(0))
        non_feat_gradients = feat.new_zeros(non_features.size(0))

        for alpha in tqdm(torch.linspace(0., 1.0, steps + 1),
                          desc='Computing feature importance...',
                          disable=disable):
            ###### Compute integrated gradients for removing features ######
            feat_diff = feat - baseline_remove
            feat_step = baseline_remove + alpha * feat_diff
            feat_step.requires_grad_()

            gradients = self.compute_feature_gradients(
                feat_step, adj, target, target_label)
            feat_gradients += gradients[features[:, 0], features[:, 1]]

            ###### Compute integrated gradients for adding features ######
            feat_diff = baseline_add - feat
            feat_step = baseline_add - alpha * feat_diff
            feat_step.requires_grad_()

            gradients = self.compute_feature_gradients(
                feat_step, adj, target, target_label)
            non_feat_gradients += gradients[non_features[:,
                                                         0], non_features[:, 1]]

        integrated_grads = feat.new_zeros(feat_indicator.size(0))
        integrated_grads[feat_indicator] = feat_gradients
        integrated_grads[~feat_indicator] = non_feat_gradients

        return integrated_grads, feat_indicator

    def compute_structure_gradients(self, feat, adj_step, target, target_label):

        logit = self.surrogate(feat, adj_step)[target].view(1, -1) / self.eps
        loss = F.cross_entropy(logit, target_label)
        return grad(loss, adj_step, create_graph=False)[0]

    def compute_feature_gradients(self, feat_step, adj, target, target_label):

        logit = self.surrogate(feat_step, adj)[target].view(1, -1) / self.eps
        loss = F.cross_entropy(logit, target_label)
        return grad(loss, feat_step, create_graph=False)[0]
