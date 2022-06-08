import warnings
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from graphwar.attack.backdoor.backdoor_attacker import BackdoorAttacker
from graphwar.functional import spmm
from graphwar import Surrogate


class LGCBackdoor(BackdoorAttacker):
    r"""Implementation of `LGCB` attack from the: 
    `"Neighboring Backdoor Attacks on Graph Convolutional Network" 
    <https://arxiv.org/abs/2201.06202>`_ paper (arXiv'22)

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.backdoor import LGCBackdoor
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker = LGCBackdoor(data)

    >>> attacker.reset()
    >>> attacker.attack(num_budgets=50, target_class=0)

    >>> attacker.data() # get attacked graph

    >>> attacker.trigger() # get trigger node

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.      

    """

    @torch.no_grad()
    def setup_surrogate(self, surrogate: nn.Module) -> "LGCBackdoor":
        W = None
        for para in surrogate.parameters():
            if para.ndim == 1:
                warnings.warn(f"The surrogate model has `bias` term, which is ignored and the "
                              f"model itself may not be a perfect choice for {self.name}.")
                continue
            if W is None:
                W = para.detach()
            else:
                W = para.detach() @ W

        assert W is not None
        self.W = W.t()
        self.num_classes = self.W.size(-1)
        return self

    def attack(self, num_budgets: Union[int, float], target_class: int,
               disable: bool = False) -> "LGCBackdoor":
        super().attack(num_budgets, target_class)
        assert target_class < self.num_classes

        feat_perturbations = self.get_feat_perturbations(
            self.W, target_class, self.num_budgets)

        trigger = self.feat.new_zeros(self.num_feats)
        trigger[feat_perturbations] = 1.

        self._trigger = trigger

        return self

    @staticmethod
    def get_feat_perturbations(W: Tensor, target_class: int, num_budgets: int) -> Tensor:
        D = W - W[:, target_class].view(-1, 1)
        D = D.sum(1)
        _, indices = torch.topk(D, k=num_budgets, largest=False)
        return indices


class FGBackdoor(BackdoorAttacker, Surrogate):
    r"""Implementation of `GB-FGSM` attack from the: 
    `"Neighboring Backdoor Attacks on Graph Convolutional Network" 
    <https://arxiv.org/abs/2201.06202>`_ paper (arXiv'22)

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.backdoor import FGBackdoor
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker = FGBackdoor(data)

    >>> attacker.reset()
    >>> attacker.attack(num_budgets=50, target_class=0)

    >>> attacker.data() # get attacked graph

    >>> attacker.trigger() # get trigger node

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.         

    """

    def setup_surrogate(self, surrogate: nn.Module, *,
                        eps: float = 1.0) -> "FGBackdoor":

        Surrogate.setup_surrogate(self, surrogate=surrogate,
                                  eps=eps, freeze=True)
        W = []
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                warnings.warn(f"The surrogate model has `bias` term, which is ignored and the "
                              f"model itself may not be a perfect choice for {self.name}.")
            else:
                W.append(para.detach().t())

        assert len(W) == 2
        self.w1, self.w2 = W
        self.num_classes = W[-1].size(-1)
        return self

    def attack(self, num_budgets: Union[int, float], target_class: int, disable: bool = False) -> "FGBackdoor":
        super().attack(num_budgets, target_class)
        assert target_class < self.num_classes

        N = self.num_nodes
        feat = self.feat

        trigger = feat.new_zeros(self.num_feats).requires_grad_()
        target_labels = torch.LongTensor(
            [target_class]).to(self.device).repeat(N)

        (edge_index, edge_weight_with_trigger,
            edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight,
            augmented_edge_index, augmented_edge_weight) = get_backdoor_edges(self.edge_index, N)

        for _ in tqdm(range(self.num_budgets), desc="Updating trigger using gradients...", disable=disable):
            aug_feat = torch.cat([feat, trigger.repeat(N, 1)], dim=0)
            feat1 = aug_feat @ self.w1
            h1 = spmm(feat1, edge_index_with_self_loop, edge_weight)
            h1_aug = spmm(feat1, augmented_edge_index,
                          augmented_edge_weight).relu()
            h = spmm(h1_aug @ self.w2, trigger_edge_index, trigger_edge_weight) + \
                spmm(h1 @ self.w2, edge_index, edge_weight_with_trigger)
            h = h[:N] / self.eps
            loss = F.cross_entropy(h, target_labels)
            gradients = torch.autograd.grad(-loss, trigger)[0] * (1. - trigger)
            trigger.data[gradients.argmax()].fill_(1.0)

        self._trigger = trigger.detach()

        return self


def get_backdoor_edges(edge_index: Tensor, N: int) -> Tuple:
    device = edge_index.device
    influence_nodes = torch.arange(N, device=device)

    N_all = N + influence_nodes.size(0)
    trigger_nodes = torch.arange(N, N_all, device=device)

    # 1. edge index of original graph (without selfloops)
    edge_index, _ = remove_self_loops(edge_index)

    # 2. edge index of original graph (with selfloops)
    edge_index_with_self_loop, _ = add_self_loops(edge_index)

    # 3. edge index of trigger nodes conneted to victim nodes with selfloops (with self-loop)
    trigger_edge_index = torch.stack([trigger_nodes, influence_nodes], dim=0)
    diag_index = torch.arange(N_all, device=device).repeat(2, 1)
    trigger_edge_index = torch.cat(
        [trigger_edge_index, trigger_edge_index[[1, 0]], diag_index], dim=1)

    # 4. all edge index with trigger nodes
    augmented_edge_index = torch.cat([edge_index, trigger_edge_index], dim=1)

    d = degree(edge_index[0], num_nodes=N, dtype=torch.float)
    d_augmented = d.clone()
    d_augmented[influence_nodes] += 1.
    d_augmented = torch.cat([d_augmented, torch.full(
        trigger_nodes.size(), 2, device=device)])

    d_pow = d.pow(-0.5)
    d_augmented_pow = d_augmented.pow(-0.5)

    edge_weight = d_pow[edge_index_with_self_loop[0]] * \
        d_pow[edge_index_with_self_loop[1]]
    edge_weight_with_trigger = d_augmented_pow[edge_index[0]
                                               ] * d_pow[edge_index[1]]
    trigger_edge_weight = d_augmented_pow[trigger_edge_index[0]
                                          ] * d_augmented_pow[trigger_edge_index[1]]
    augmented_edge_weight = torch.cat(
        [edge_weight_with_trigger, trigger_edge_weight], dim=0)

    return (edge_index, edge_weight_with_trigger,
            edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight,
            augmented_edge_index, augmented_edge_weight)
