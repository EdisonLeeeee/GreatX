from copy import copy
from typing import Optional, Union
import numpy as np
from graphwar.attack.injection.injection_attacker import InjectionAttacker
from torch import Tensor


class RandomInjection(InjectionAttacker):
    """Injection nodes into a graph randomly."""

    def attack(self, num_budgets: Union[int, float], *, targets: Optional[Tensor] = None,
               interconnection: bool = False,
               num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None,
               feat_limits: Optional[Union[tuple, dict]] = None) -> "RandomInjection":
        super().attack(num_budgets, targets=targets,
                       num_edges_global=num_edges_global,
                       num_edges_local=num_edges_local,
                       feat_limits=feat_limits)

        candidate_nodes = copy(self.targets)

        for injected_node in range(self.num_nodes, self.num_nodes+self.num_budgets):
            sampled = np.random.choice(
                candidate_nodes, self.num_edges_local, replace=False)

            self.inject_node(injected_node)

            for target in sampled:
                self.inject_edge(injected_node, target)

            if interconnection:
                candidate_nodes.append(injected_node)

        return self
