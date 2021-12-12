import dgl
import torch
import warnings
from typing import Optional, Union
from graphwar.attack.backdoor.backdoor_attacker import BackdoorAttacker

class LGCBackdoor(BackdoorAttacker):

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        super().__init__(graph=graph, device=device, seed=seed, name=name, **kwargs)
        self._check_feature_matrix_binary()

    def setup_surrogate(self, surrogate):
        W = None
        for para in surrogate.parameters():
            if para.ndim == 1:
                warnings.warn(f"The surrogate model has `bias` term, which is ignored and the "
                              "model itself may not be a perfect choice for Nettack.")
                continue
            if W is None:
                W = para.detach()
            else:
                W = W @ para.detach()
        assert W is not None
        self.W = W
        self.num_classes = self.W.shape[-1]
        return self

    def attack(self, num_budgets: Union[int, float], target_class: int, disable: bool=False):
        super().attack(num_budgets, target_class)
        assert target_class < self.num_classes

        feat_perturbations = self.get_feat_perturbations(self.W, target_class, self.num_budgets)

        trigger = self.feat.new_zeros(self.num_feats)
        trigger[feat_perturbations] = 1.

        self._trigger = trigger

        return self

    @staticmethod
    def get_feat_perturbations(W, target_class, num_budgets):
        D = W - W[:, target_class].view(-1, 1)
        D = D.sum(1)
#         _, indices = torch.topk(-D, k=num_budgets)
        _, indices = torch.topk(D, k=num_budgets, largest=False)
        return indices
