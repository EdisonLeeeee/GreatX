from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm

from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.utils import ego_graph
from graphwar.surrogate import Surrogate

SubGraph = namedtuple('SubGraph', ['edge_index', 'sub_edges', 'non_edges',
                                   'edge_weight', 'non_edge_weight', 'selfloop_weight'])


class SGAttack(TargetedAttacker, Surrogate):
    r"""Implementation of `SGA` attack from the: 
    `"Adversarial Attack on Large Scale Graph" 
    <https://arxiv.org/abs/2009.03488>`_ paper (TKDE'21)

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

    >>> from graphwar.attack.targeted import SGAttack
    >>> attacker = SGAttack(data)
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
    * `SGAttack` is a scalable attack that can be applied to large scale graphs.
    * Please remember to call :meth:`reset` before each attack.        
    """

    # SGAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton = True

    @torch.no_grad()
    def setup_surrogate(self, surrogate: torch.nn.Module,
                        eps: float = 5.0,
                        freeze: bool = True,
                        K: int = 2):

        Surrogate.setup_surrogate(self, surrogate=surrogate,
                                  eps=eps, freeze=freeze)

        self.logits = self.surrogate(
            self.feat, self.edge_index, self.edge_weight)

        self.K = K
        return self

    def set_normalize(self, state):
        for layer in self.surrogate.modules():
            if hasattr(layer, 'normalize'):
                layer.normalize = state
            if hasattr(layer, 'add_self_loops'):
                layer.add_self_loops = state

    def strongest_wrong_class(self, target, target_label):
        logit = self.logits[target].clone()
        logit[target_label] = -1e4
        return logit.argmax()

    def get_subgraph(self, target, target_label, best_wrong_label):
        sub_nodes, sub_edges = ego_graph(
            self.adjacency_matrix, int(target), self.K)
        if sub_edges.size == 0:
            raise RuntimeError(
                f"The target node {int(target)} is a singleton node.")
        sub_nodes = torch.as_tensor(
            sub_nodes, dtype=torch.long, device=self.device)
        sub_edges = torch.as_tensor(
            sub_edges, dtype=torch.long, device=self.device)
        attacker_nodes = torch.where(self.label == best_wrong_label)[
            0].cpu().numpy()
        neighbors = self.adjacency_matrix[target].indices

        influencers = [target]
        attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        subgraph = self.subgraph_processing(
            sub_nodes, sub_edges, influencers, attacker_nodes)

        if self.direct_attack:
            influencers = [target]
            num_attackers = self.num_budgets + 1
        else:
            influencers = neighbors
            num_attackers = 3
        attacker_nodes = self.get_top_attackers(subgraph, target, target_label,
                                                best_wrong_label, num_attackers=num_attackers)

        subgraph = self.subgraph_processing(
            sub_nodes, sub_edges, influencers, attacker_nodes)

        return subgraph

    def get_top_attackers(self, subgraph, target, target_label, best_wrong_label, num_attackers):
        non_edge_grad, _ = self.compute_gradients(
            subgraph, target, target_label, best_wrong_label)
        _, index = torch.topk(non_edge_grad, k=num_attackers, sorted=False)
        attacker_nodes = subgraph.non_edges[1][index]
        return attacker_nodes.tolist()

    def subgraph_processing(self, sub_nodes, sub_edges, influencers, attacker_nodes):
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if not self.direct_attack:  # indirect attack
            mask = self.adjacency_matrix[non_edges[0],
                                         non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        non_edges = torch.as_tensor(
            non_edges, dtype=torch.long, device=self.device)
        attacker_nodes = torch.as_tensor(
            attacker_nodes, dtype=torch.long, device=self.device)
        selfloop = torch.unique(torch.cat([sub_nodes, attacker_nodes]))
        edge_index = torch.cat([non_edges, sub_edges, non_edges.flip(0),
                                sub_edges.flip(0), selfloop.repeat((2, 1))], dim=1)

        edge_weight = torch.ones(sub_edges.size(
            1), device=self.device).requires_grad_()
        non_edge_weight = torch.zeros(non_edges.size(
            1), device=self.device).requires_grad_()
        selfloop_weight = torch.ones(selfloop.size(0), device=self.device)

        subgraph = SubGraph(edge_index=edge_index, sub_edges=sub_edges, non_edges=non_edges,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            selfloop_weight=selfloop_weight,)
        return subgraph

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

        self.set_normalize(False)

        if target_label is None:
            assert self.target_label is not None, "please specify argument `target_label` as the node label does not exist."
            target_label = self.target_label.view(-1)
        else:
            target_label = torch.as_tensor(
                target_label, device=self.device, dtype=torch.long).view(-1)

        best_wrong_label = self.strongest_wrong_class(
            target, target_label).view(-1)

        subgraph = self.get_subgraph(target, target_label, best_wrong_label)

        if not direct_attack:
            condition1 = subgraph.sub_edges[0] != target
            condition2 = subgraph.sub_edges[1] != target
            mask = torch.logical_and(condition1, condition2).float()

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing graph...',
                       disable=disable):

            non_edge_grad, edge_grad = self.compute_gradients(subgraph, target,
                                                              target_label, best_wrong_label)

            with torch.no_grad():
                edge_grad *= -2 * subgraph.edge_weight + 1
                if not direct_attack:
                    edge_grad *= mask
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1

            max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
            max_non_edge_grad, max_non_edge_idx = torch.max(
                non_edge_grad, dim=0)

            if max_edge_grad > max_non_edge_grad:
                # remove one edge
                subgraph.edge_weight.data[max_edge_idx].fill_(0.)
                u, v = subgraph.sub_edges[:, max_edge_idx].tolist()
                self.remove_edge(u, v, it)
            else:
                # add one edge
                subgraph.non_edge_weight.data[max_non_edge_idx].fill_(1.)
                u, v = subgraph.non_edges[:, max_non_edge_idx].tolist()
                self.add_edge(u, v, it)

        self.set_normalize(True)
        return self

    def compute_gradients(self, subgraph, target, target_label, best_wrong_label):
        edge_weight = torch.cat([subgraph.non_edge_weight, subgraph.edge_weight,
                                 subgraph.non_edge_weight, subgraph.edge_weight,
                                 subgraph.selfloop_weight], dim=0)

        row, col = subgraph.edge_index
        norm = (self.degree + 1.).pow(-0.5)
        edge_weight = norm[row] * edge_weight * norm[col]

        logit = self.surrogate(self.feat, subgraph.edge_index, edge_weight)
        logit = logit[target].view(1, -1) / self.eps
        logit = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(logit, target_label) - \
            F.nll_loss(logit, best_wrong_label)
        return grad(loss, [subgraph.non_edge_weight, subgraph.edge_weight], create_graph=False)
