from typing import Optional, Union, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class Surrogate(Module):
    """Base class for attacker or defenders that require
    a surrogate model for estimating labels or computing
    gradient information.

    Parameters
    ----------
    device : str, optional
        the device of a model to use for, by default "cpu"    

    """
    _is_setup = False  # flags to denote the surrogate model is properly set

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

    def setup_surrogate(self, surrogate: Module, *,
                        eps: float = 1.0,
                        freeze: bool = True,
                        required: Union[Module, Tuple[Module]] = None) -> "Surrogate":
        """Method used to initialize the (trained) surrogate model.

        Parameters
        ----------
        surrogate : Module
            the input surrogate module
        eps : float, optional
            temperature used for softmax activation, by default 1.0
        freeze : bool, optional
            whether to freeze the model's parameters to save time, by default True
        required : Union[Module, Tuple[Module]], optional
            which class(es) of the surrogate model are required, by default None

        Returns
        -------
        Surrogate
            the class itself

        Raises
        ------
        RuntimeError
            if the surrogate model is not an instance of :class:`torch.nn.Module`
        RuntimeError
            if the surrogate model is not an instance of :obj:`required`
        """

        if not isinstance(surrogate, Module):
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

    def estimate_self_training_labels(self,
                                      nodes: Optional[Tensor] = None) -> Tensor:
        """Estimate the labels of nodes using the trained surrogate model.

        Parameters
        ----------
        nodes : Optional[Tensor], optional
            the input nodes, if None, it would be all nodes in the graph,
            by default None

        Returns
        -------
        Tensor
            the labels of the input nodes.
        """
        self_training_labels = self.surrogate(
            self.feat, self.edge_index, self.edge_weight)
        if nodes is not None:
            self_training_labels = self_training_labels[nodes]
        return self_training_labels.argmax(-1)

    def freeze_surrogate(self) -> "Surrogate":
        """Freezie the parameters of the surrogate model.

        Returns
        -------
        Surrogate
            the class itself
        """
        for para in self.surrogate.parameters():
            para.requires_grad_(False)
        return self

    def defrozen_surrogate(self) -> "Surrogate":
        """Defrozen the parameters of the surrogate model

        Returns
        -------
        Surrogate
            the class itself
        """
        for para in self.surrogate.parameters():
            para.requires_grad_(True)
        return self
