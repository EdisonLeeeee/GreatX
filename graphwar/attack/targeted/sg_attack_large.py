import torch
import dgl
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
import dgl.ops as ops

from tqdm import tqdm
from functools import lru_cache
from torch.autograd import grad
from typing import Optional, Callable

from graphwar.utils import ego_graph
from graphwar.models import SGC
from graphwar.attack.targeted.targeted_attacker import TargetedAttacker
from graphwar.surrogater import Surrogater

from collections import namedtuple
SubGraph = namedtuple('SubGraph', ['edge_index', 'sub_edges', 'non_edges',
                                   'edge_weight', 'non_edge_weight', 'selfloop_weight', 'dgl_graph'])


class SGAttackLarge(TargetedAttacker, Surrogater):
    # SGAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton = True

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_exists()
        self._check_node_label_exists()

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        loss: Callable = F.nll_loss,
                        eps: float = 5.0,
                        freeze: bool = True):

        Surrogater.setup_surrogate(self, surrogate=surrogate, loss=loss,
                                   eps=eps, freeze=freeze, required=SGC)
        self.surrogate.cache_clear()
        self.compute_XW.cache_clear()
        self.logits = self.surrogate(self.graph, self.feat)
        self.k = self.surrogate.conv._k
        self.weight = self.surrogate.conv.weight.detach()
        self.bias = self.surrogate.conv.bias.detach()
        return self

    def strongest_wrong_class(self, target, target_label):
        logit = self.logits[target]
        logit[target_label] = -1e4
        return logit.argmax()

    def get_subgraph(self, target, target_label, best_wrong_label):
        sub_nodes, sub_edges = ego_graph(self.adjacency_matrix, int(target), self.k)
        if sub_edges.size== 0:
            raise RuntimeError(f"The target node {int(target)} is a singleton node.")        
        sub_nodes = torch.as_tensor(sub_nodes, dtype=torch.long, device=self.device)
        sub_edges = torch.as_tensor(sub_edges, dtype=torch.long, device=self.device)
        attacker_nodes = torch.where(self.label == best_wrong_label)[0].cpu().numpy()
        neighbors = self.adjacency_matrix[target].indices

        influencers = [target]
        attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        subgraph = self.subgraph_processing(sub_nodes, sub_edges, influencers, attacker_nodes)

        if self.direct_attack:
            influencers = [target]
            n_attackers = self.num_budgets + 1
        else:
            influencers = neighbors
            n_attackers = 3
        attacker_nodes = self.get_top_attackers(subgraph, target, target_label,
                                                best_wrong_label, n_attackers=n_attackers)

        subgraph = self.subgraph_processing(sub_nodes, sub_edges, influencers, attacker_nodes)

        return subgraph

    def get_top_attackers(self, subgraph, target, target_label, best_wrong_label, n_attackers):
        non_edge_grad, _ = self._compute_gradients(subgraph, target, target_label, best_wrong_label)
        _, index = torch.topk(non_edge_grad, k=n_attackers, sorted=False)
        attacker_nodes = subgraph.non_edges[1][index]
        return attacker_nodes.cpu().tolist()

    def subgraph_processing(self, sub_nodes, sub_edges, influencers, attacker_nodes):
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if not self.direct_attack:  # indirect attack
            mask = self.adjacency_matrix[non_edges[0],
                                         non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        non_edges = torch.as_tensor(non_edges, dtype=torch.long, device=self.device)
        attacker_nodes = torch.as_tensor(attacker_nodes, dtype=torch.long, device=self.device)
        selfloop = torch.unique(torch.cat([sub_nodes, attacker_nodes]))
        edge_index = torch.cat([non_edges, sub_edges, non_edges[[1, 0]],
                                sub_edges[[1, 0]], selfloop.repeat((2, 1))], dim=1)

        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_()
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_()
        selfloop_weight = torch.ones(selfloop.size(0), device=self.device)

        dgl_graph = dgl.graph((edge_index[0], edge_index[1]), device=self.device, num_nodes=self.num_nodes)

        subgraph = SubGraph(edge_index=edge_index, sub_edges=sub_edges, non_edges=non_edges,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            selfloop_weight=selfloop_weight, dgl_graph=dgl_graph)
        return subgraph

    @lru_cache(maxsize=1)
    def compute_XW(self):
        return self.feat @ self.weight

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

        if target_label is None:
            assert self.target_label is not None, "please specify argument `target_label` as the node label does not exist."
            target_label = self.target_label.view(-1)
        else:
            target_label = torch.as_tensor(target_label, device=self.device, dtype=torch.long).view(-1)

        best_wrong_label = self.strongest_wrong_class(target, target_label).view(-1)

        subgraph = self.get_subgraph(target, target_label, best_wrong_label)

        if not direct_attack:
            condition1 = subgraph.sub_edges[0] != target
            condition2 = subgraph.sub_edges[1] != target
            mask = torch.logical_and(condition1, condition2).float()

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing Graph',
                       disable=disable):

            non_edge_grad, edge_grad = self._compute_gradients(subgraph, target,
                                                               target_label, best_wrong_label)

            with torch.no_grad():
                edge_grad *= -2 * subgraph.edge_weight + 1
                if not direct_attack:
                    edge_grad *= mask
                non_edge_grad *= -2 * subgraph.non_edge_weight + 1

            max_edge_grad, max_edge_idx = torch.max(edge_grad, dim=0)
            max_non_edge_grad, max_non_edge_idx = torch.max(non_edge_grad, dim=0)

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

        return self

    def _compute_gradients(self, subgraph, target, target_label, best_wrong_label):
        edge_weight = torch.cat([subgraph.non_edge_weight, subgraph.edge_weight,
                                 subgraph.non_edge_weight, subgraph.edge_weight,
                                 subgraph.selfloop_weight], dim=0)

        logit = self.SGConv(subgraph, self.compute_XW(), edge_weight)
        logit = logit[target].view(1, -1) / self.eps
        logit = F.log_softmax(logit, dim=1)
        loss = self.loss_fn(logit, target_label) - self.loss_fn(logit, best_wrong_label)
        return grad(loss, [subgraph.non_edge_weight, subgraph.edge_weight], create_graph=False)

    def SGConv(self, subgraph, x, edge_weight):
        norm = (self.degree + 1.).pow(-0.5)
        graph = subgraph.dgl_graph
        
        edge_weight = ops.e_mul_u(graph, edge_weight, norm)
        edge_weight = ops.e_mul_v(graph, edge_weight, norm)

        graph.ndata['h'] = x
        graph.edata['edge_weight'] = edge_weight

        for _ in range(self.k):
            graph.update_all(fn.u_mul_e('h', 'edge_weight', 'm'),
                             fn.sum('m', 'h'))

        x = graph.ndata.pop('h')

        if self.bias is not None:
            x += self.bias
        return x
