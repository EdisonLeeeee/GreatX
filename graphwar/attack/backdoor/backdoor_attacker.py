from typing import Optional, Union
from copy import copy

import torch
from torch import Tensor
from torch_geometric.data import Data

from graphwar.attack.attacker import Attacker
from graphwar.utils import add_edges


class BackdoorAttacker(Attacker):
    """Base class for backdoor attacks.

    """

    def reset(self) -> "BackdoorAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        BackdoorAttacker
            the attacker itself
        """
        self.num_budgets = None
        self._trigger = None
        self._is_reset = True

        return self

    def attack(self, num_budgets: Union[int, float], targets_class: int) -> "BackdoorAttacker":
        """Base method that describes the adversarial backdoor attack
        """

        _is_setup = getattr(self, "_is_setup", True)

        if not _is_setup:
            raise RuntimeError(
                f'{self.__class__.__name__} requires a surrogate model to conduct attack. '
                'Use `attacker.setup_surrogate(surrogate_model)`.')

        if not self._is_reset:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        num_budgets = self._check_budget(
            num_budgets, max_perturbations=self.num_feats)

        self.num_budgets = num_budgets
        self.targets_class = torch.LongTensor(
            targets_class).view(-1).to(self.device)
        self._is_reset = False

        return self

    def trigger(self) -> Tensor:
        return self._trigger

    def data(self, target_node: int, symmetric: bool = True) -> Data:
        """return the attacked graph

        Parameters
        ----------
        target_node : int
            the target node that the attack performed
        symmetric : bool
            determine whether the resulting graph is forcibly symmetric,
            by default True

        Returns
        -------
        Data
            the attacked graph with backdoor attack performed on the target node
        """
        data = copy(self.ori_data)
        num_nodes = self.num_nodes
        feat = self.trigger().view(1, -1)
        edges_to_add = torch.tensor([num_nodes, target_node]).view(
            2, 1).to(data.edge_index)
        data.x = torch.cat([data.x, feat], dim=0)
        data.edge_index = add_edges(
            data.edge_index, edges_to_add, symmetric=symmetric)

        return data
