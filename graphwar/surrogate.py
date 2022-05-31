from typing import Callable

import torch


class Surrogate(torch.nn.Module):
    _is_setup = False  # flags to denote the surrogate model is properly set

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

    def setup_surrogate(self, surrogate: torch.nn.Module, *,
                        eps: float = 1.0,
                        freeze: bool = True,
                        required: torch.nn.Module = None):

        if not isinstance(surrogate, torch.nn.Module):
            raise RuntimeError(
                "The surrogate model must be an instance of `torch.nn.Module`.")

        if required is not None and not isinstance(surrogate, required):
            raise RuntimeError(
                f"The surrogate model is required to be `{required}`, but got `{surrogate.__class__.__name__}`.")

        surrogate.eval()
        if hasattr(surrogate, 'cache_clear'):
            surrogate.cache_clear()

        for layer in surrogate.modules():
            if hasattr(layer, 'cached'):
                layer.cached = False

        self.surrogate = surrogate.to(self.device)
        self.eps = eps

        if freeze:
            self.freeze_surrogate()

        self._is_setup = True

        return self

    def estimate_self_training_labels(self, nodes=None):
        self_training_labels = self.surrogate(
            self.feat, self.edge_index, self.edge_weight)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)

    def freeze_surrogate(self):
        for para in self.surrogate.parameters():
            para.requires_grad_(False)

    def defrozen_surrogate(self):
        for para in self.surrogate.parameters():
            para.requires_grad_(True)
