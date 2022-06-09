from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm

from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.surrogate import Surrogate
from graphwar.utils import singleton_mask
from graphwar.functional import to_dense_adj


class FGAttack(TargetedAttacker, Surrogate):
    r"""Implementation of `FGA` attack from the: 
    `"Fast Gradient Attack on Network Embedding" 
    <https://arxiv.org/abs/1809.02797>`_ paper (arXiv'18)

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

    >>> from graphwar.attack.targeted import FGAttack
    >>> attacker = FGAttack(data)
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
    This is a simple but effective attack that utilizing gradient information
    of the adjacency matrix. There are several work sharing the same heuristic,
    we list them as follows:
    [1] `FGSM`: `"Explaining and Harnessing Adversarial Examples" 
    <https://arxiv.org/abs/1412.6572>`_ paper (ICLR'15)
    [2] `"Link Prediction Adversarial Attack Via Iterative Gradient Attack" 
    <https://ieeexplore.ieee.org/abstract/document/9141291>`_ paper (IEEE Trans'20)
    [3] `"Adversarial Attack on Graph Structured Data" 
    <https://arxiv.org/abs/1806.02371>`_ paper (ICML'18)    

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.     
    """

    # FGAttack can conduct feature attack
    _allow_feature_attack: bool = True
    # FGAttack cannot ensure there are no singleton nodes
    _allow_singleton: bool = True

    def reset(self):
        super().reset()
        self.modified_adj = to_dense_adj(self.edge_index,
                                         self.edge_weight,
                                         num_nodes=self.num_nodes).to(self.device)
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
            target_label = torch.as_tensor(
                target_label, device=self.device, dtype=torch.long).view(-1)

        modified_adj = self.modified_adj
        modified_feat = self.modified_feat
        modified_adj.requires_grad_(bool(structure_attack))
        modified_feat.requires_grad_(bool(feature_attack))

        target = torch.as_tensor(target, device=self.device, dtype=torch.long)
        target_label = torch.as_tensor(
            target_label, device=self.device, dtype=torch.long).view(-1)
        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing graph...',
                       disable=disable):

            adj_grad, feat_grad = self.compute_gradients(modified_adj,
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

    def compute_gradients(self, modified_adj, modified_feat, target, target_label):

        logit = self.surrogate(modified_feat, modified_adj)[
            target].view(1, -1) / self.eps
        loss = F.cross_entropy(logit, target_label)

        if self.structure_attack and self.feature_attack:
            return grad(loss, [modified_adj, modified_feat], create_graph=False)

        if self.structure_attack:
            return grad(loss, modified_adj, create_graph=False)[0], None

        if self.feature_attack:
            return None, grad(loss, modified_feat, create_graph=False)[0]
