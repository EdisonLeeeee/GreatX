import dgl
import torch
import warnings
from tqdm import tqdm
from typing import Optional, Union, Callable
from graphwar.attack.backdoor.backdoor_attacker import BackdoorAttacker
from graphwar.attack.backdoor.backdoor_utils import backdoor_edges, conv
from graphwar.attack.surrogate_attacker import SurrogateAttacker


class FGBackdoor(BackdoorAttacker, SurrogateAttacker):

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_binary()

    def setup_surrogate(self, surrogate: torch.nn.Module, *,
                        loss: Callable = torch.nn.CrossEntropyLoss(),
                        eps: float = 1.0):

        SurrogateAttacker.setup_surrogate(self, surrogate=surrogate,
                                          loss=loss, eps=eps, freeze=True)
        W = []
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                warnings.warn(f"The surrogate model has `bias` term, which is ignored and the "
                              "model itself may not be a perfect choice for Nettack.")
            else:
                W.append(para.detach())

        assert len(W) == 2
        self.w1, self.w2 = W
        self.num_classes = W[-1].size(-1)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        return self

    def attack(self, num_budgets: Union[int, float], target_class: int, disable: bool = False):
        super().attack(num_budgets, target_class)
        assert target_class < self.num_classes

        N = self.num_nodes
        feat = self.feat

        trigger = feat.new_zeros(self.num_feats).requires_grad_()
        target_labels = torch.LongTensor([target_class]).to(self.device).repeat(N)

        (edge_index, edge_weight_with_trigger,
            edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight,
            augmented_edge_index, augmented_edge_weight) = backdoor_edges(self.graph)

        for _ in tqdm(range(self.num_budgets), desc="Updating trigger using gradients", disable=disable):
            aug_feat = torch.cat([feat, trigger.repeat(N, 1)], dim=0)
            feat1 = aug_feat @ self.w1
            h1 = conv(edge_index_with_self_loop, feat1, edge_weight)
            h1_aug = conv(augmented_edge_index, feat1, augmented_edge_weight).relu()
            h = conv(trigger_edge_index, h1_aug @ self.w2, trigger_edge_weight) + conv(edge_index, h1 @ self.w2, edge_weight_with_trigger)
            h = h[:N] / self.eps
            loss = self.loss_fn(h, target_labels)
            gradients = torch.autograd.grad(-loss, trigger)[0] * (1. - trigger)
            trigger.data[gradients.argmax()].fill_(1.0)

        self._trigger = trigger.detach()

        return self
