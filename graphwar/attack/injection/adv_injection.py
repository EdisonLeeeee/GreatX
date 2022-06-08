from copy import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from tqdm import tqdm
from graphwar import Surrogate
from graphwar.attack.injection.injection_attacker import InjectionAttacker


class AdvInjection(InjectionAttacker, Surrogate):
    r"""2nd place solution of KDD CUP 2020
    "Adversarial attack and defense" challenge.

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.injection import AdvInjection
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker = AdvInjection(data)

    >>> attacker.reset()
    >>> attacker.attack(10, feat_limits=(0, 1))  # injecting 10 nodes for continuous features

    >>> attacker.reset()
    >>> attacker.attack(10, feat_budgets=10)  # injecting 10 nodes for binary features    

    >>> attacker.data() # get attacked graph

    >>> attacker.injected_nodes() # get injected nodes after attack

    >>> attacker.injected_edges() # get injected edges after attack

    >>> attacker.injected_feats() # get injected features after attack   

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.        
    """

    def attack(self, num_budgets: Union[int, float], *,
               targets: Optional[Tensor] = None,
               interconnection: bool = False,
               lr: float = 0.01,
               num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None,
               feat_limits: Optional[Union[tuple, dict]] = None,
               feat_budgets:  Optional[int] = None,
               disable: bool = False) -> "AdvInjection":
        super().attack(num_budgets, targets=targets,
                       num_edges_global=num_edges_global,
                       num_edges_local=num_edges_local,
                       feat_limits=feat_limits,
                       feat_budgets=feat_budgets)

        candidate_nodes = self.targets.tolist()
        edge_index, edge_weight, feat = self.edge_index, self.edge_weight, self.feat
        if edge_weight is None:
            edge_weight = feat.new_ones(edge_index.size(1))

        feat_min, feat_max = self.feat_limits
        feat_limits = max(abs(feat_min), feat_max)
        feat_budgets = self.feat_budgets
        injected_feats = None

        for injected_node in tqdm(range(self.num_nodes, self.num_nodes+self.num_budgets),
                                  desc="Injecting nodes...",
                                  disable=disable):
            injected_edge_index = np.stack(
                [np.tile(injected_node, len(candidate_nodes)), candidate_nodes], axis=0)
            injected_edge_index = torch.as_tensor(
                injected_edge_index).to(edge_index)
            injected_edge_weight = edge_weight.new_zeros(
                injected_edge_index.size(1)).requires_grad_()

            injected_feat = feat.new_zeros(1, self.num_feats)

            if injected_feats is None:
                injected_feats = injected_feat.requires_grad_()
            else:
                injected_feats = torch.cat(
                    [injected_feats, injected_feat], dim=0).requires_grad_()

            edge_grad, feat_grad = self.compute_gradients(
                feat, edge_index, edge_weight,
                injected_feats, injected_edge_index, injected_edge_weight,
                targets=self.targets, target_labels=self.target_labels)

            topk_edges = torch.topk(edge_grad, k=self.num_edges_local).indices
            injected_edge_index = injected_edge_index[:, topk_edges]

            self.inject_node(injected_node)
            self.inject_edges(injected_edge_index)

            with torch.no_grad():
                edge_index = torch.cat(
                    [edge_index, injected_edge_index, injected_edge_index.flip(0)], dim=1)
                edge_weight = torch.cat(
                    [edge_weight, edge_weight.new_ones(injected_edge_index.size(1)*2)], dim=0)

                if feat_budgets is not None:
                    topk = torch.topk(
                        feat_grad, k=feat_budgets, dim=1)
                    injected_feats.data.fill_(0.)
                    injected_feats.data.scatter_(
                        1, topk.indices, 1.0)
                else:
                    injected_feats.data = (
                        feat_limits * feat_grad.sign()).clamp(min=feat_min, max=feat_max)

            if interconnection:
                candidate_nodes.append(injected_node)

        self._injected_feats = injected_feats.data
        return self

    def compute_gradients(self, x, edge_index, edge_weight,
                          injected_feats, injected_edge_index,
                          injected_edge_weight,
                          targets, target_labels):

        x = torch.cat([x, injected_feats], dim=0)
        edge_index = torch.cat(
            [edge_index, injected_edge_index, injected_edge_index.flip(0)], dim=1)
        edge_weight = torch.cat(
            [edge_weight, injected_edge_weight.repeat(2)], dim=0)
        logit = self.surrogate(x, edge_index, edge_weight)[targets] / self.eps
        loss = F.cross_entropy(logit, target_labels)

        return grad(loss, [injected_edge_weight, injected_feats], create_graph=False)
