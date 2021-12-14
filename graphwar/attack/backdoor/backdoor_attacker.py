import dgl
import torch
from torch import Tensor
from functools import lru_cache
from typing import Optional, Union
from graphwar import Config
from ..attacker import Attacker
_FEATURE = Config.feat


class BackdoorAttacker(Attacker):

    def reset(self) -> "BackdoorAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        BackdoorAttacker
            the attacker itself
        """
        self.num_budgets = None
        self._trigger = None
        self.is_reseted = True

        return self

    def attack(self, num_budgets: Union[int, float], targets_class: int) -> "BackdoorAttacker":
        """Base method that describes the adversarial backdoor attack
        """

        if not self.is_reseted:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        num_budgets = self._check_budget(
            num_budgets, max_perturbations=self.num_feats)

        self.num_budgets = num_budgets
        self.targets_class = torch.LongTensor([targets_class]).view(-1).to(self.device)
        self.is_reseted = False

        return self

    def trigger(self,):
        return self._trigger

    def g(self, target_node: int, symmetric: bool = True) -> dgl.DGLGraph:
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
        dgl.DGLGraph
            the attacked graph with backdoor attack performed on the target node
        """
        graph = self.graph.local_var()
        num_nodes = self.num_nodes
        data = self.trigger().view(1, -1)

        graph.add_nodes(1, data={_FEATURE: data})
        graph.add_edges(num_nodes, target_node)

        if symmetric:
            graph.add_edges(target_node, num_nodes)

        return graph
