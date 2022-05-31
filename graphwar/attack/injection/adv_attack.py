from copy import copy
from typing import Optional, Union

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from tqdm import tqdm
from graphwar.attack.injection.injection_attacker import InjectionAttacker
from graphwar.attack import Surrogate


class AdvInjection(InjectionAttacker, Surrogate):
    """2nd place solution of KDD CUP 2020
    "Adversarial attack and defense" challenge.


    """

    def attack(self, num_budgets: Union[int, float], *,
               targets: Optional[Tensor] = None,
               interconnection: bool = False,
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

        candidate_nodes = copy(self.targets)
        targets = torch.as_tensor(
            candidate_nodes, device=self.device, dtype=torch.long)
        edge_index, edge_weight, feat = self.edge_index, self.edge_weight, self.feat
        feat_min, feat_max = self.feat_limits
        injected_feats = None
        for injected_node in tqdm(range(self.num_nodes, self.num_nodes+self.num_budgets),
                                  desc="Injecting nodes...",
                                  disable=disable):
            injected_edge_index = np.stack(
                [np.tile(injected_node, len(candidate_nodes)), candidate_nodes], axis=1)

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
                feat, edge_index, edge_weight, targets, self.label[targets])

            topk_edges = torch.topk(edge_grad, k=self.num_edges_local).indices

            injected_edge_index = injected_edge_index[:, topk_edges]

            edge_index = torch.cat(
                [edge_index, injected_edge_index], dim=1)
            edge_weight = torch.cat(
                [edge_weight, injected_edge_weight[topk_edges]], dim=0)
            injected_feat.add_(lr*feat_grad).clamp(min=feat_min, max=feat_max)

            if interconnection:
                candidate_nodes.append(injected_node)

    def compute_gradients(self, x, edge_index, edge_weight,
                          injected_feats, injected_edge_index,
                          injected_edge_weight,
                          targets, target_labels):

        x = torch.cat([x, injected_feats], dim=0)
        edge_index = torch.cat(
            [edge_index, injected_edge_index, injected_edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, injected_edge_weight], dim=0)
        logit = self.surrogate(x, edge_index, edge_weight)[
            targets].view(1, -1) / self.eps
        loss = F.cross_entropy(logit, target_labels)

        return grad(loss, [injected_feats, injected_edge_weight], create_graph=False)
