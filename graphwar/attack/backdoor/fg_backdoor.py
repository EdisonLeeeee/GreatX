import dgl
import torch
import warnings
from tqdm import tqdm
from typing import Optional, Union, Callable
from graphwar.attack.backdoor.backdoor_attacker import BackdoorAttacker
from graphwar.attack.surrogate_attacker import SurrogateAttacker
from graphwar.functional.scatter import scatter_add


def conv(edges, x, edge_weight):
    row, col = edges
    src = x[row] * edge_weight.view(-1, 1)
    x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
    return x


def backdoor_edges(g):
    N = g.num_nodes()
    device = g.device
    influence_nodes = torch.arange(N, device=device)

    N_all = N + influence_nodes.size(0)
    trigger_nodes = torch.arange(N, N_all, device=device)

    # 1. edge index of original graph (without selfloops)
    edge_index = torch.stack(g.remove_self_loop().edges(), dim=0)

    # 2. edge index of original graph (with selfloops)
    edge_index_with_self_loop = torch.stack(g.edges(), dim=0)

    # 3. edge index of trigger nodes conneted to victim nodes with selfloops (with self-loop)
    trigger_edge_index = torch.stack([trigger_nodes, influence_nodes], dim=0)
    diag_index = torch.arange(N_all, device=device).repeat(2, 1)
    trigger_edge_index = torch.cat([trigger_edge_index, trigger_edge_index[[1, 0]], diag_index], dim=1)

    # 4. all edge index with trigger nodes
    augmented_edge_index = torch.cat([edge_index, trigger_edge_index], dim=1)

    d = g.in_degrees().float()
    d_augmented = d.clone()
    d_augmented[influence_nodes] += 1.
    d_augmented = torch.cat([d_augmented, torch.full(trigger_nodes.size(), 2, device=device)])

    d_pow = d.pow(-0.5)
    d_augmented_pow = d_augmented.pow(-0.5)

    edge_weight = d_pow[edge_index_with_self_loop[0]] * d_pow[edge_index_with_self_loop[1]]
    edge_weight_with_trigger = d_augmented_pow[edge_index[0]] * d_pow[edge_index[1]]
    trigger_edge_weight = d_augmented_pow[trigger_edge_index[0]] * d_augmented_pow[trigger_edge_index[1]]
    augmented_edge_weight = torch.cat([edge_weight_with_trigger, trigger_edge_weight], dim=0)

    return (edge_index, edge_weight_with_trigger,
            edge_index_with_self_loop, edge_weight,
            trigger_edge_index, trigger_edge_weight,
            augmented_edge_index, augmented_edge_weight)


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

    def attack(self, num_budgets: Union[int, float], target_class: int, disable: bool=False):
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
