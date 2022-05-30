from copy import copy
from typing import Optional, Union

import numpy as np
from torch import Tensor
from tqdm import tqdm
from graphwar.attack.injection.injection_attacker import InjectionAttacker


class RandomInjection(InjectionAttacker):
    """Injection nodes into a graph randomly.

    Example
    -------
    >>> attacker = RandomInjection(data)
    >>> attacker.reset()
    # inject 10 nodes, where each nodes has 2 edges
    >>> attacker.attack(num_budgets=10, num_edges_local=2) 
    # inject 10 nodes, with 100 edges in total
    >>> attacker.attack(num_budgets=10, num_edges_global=100) 
    # inject 10 nodes, where each nodes has 2 edges, 
    # the features of injected nodes lies in [0,1]
    >>> attacker.attack(num_budgets=10, num_edges_local=2, feat_limits=(0,1)) 
    >>> attacker.attack(num_budgets=10, num_edges_local=2, feat_limits={'min': 0, 'max':1}) 
    # inject 10 nodes, where each nodes has 2 edges, 
    # the features of injected each node has 10 nonzero elements
    >>> attacker.attack(num_budgets=10, num_edges_local=2, feat_budgets=10) 

    # get injected nodes
    >>> attacker.injected_nodes()
    # get injected edges
    >>> attacker.injected_edges()
    # get injected nodes' features
    >>> attacker.injected_feats()
    # get perturbed graph
    >>> attacker.data()
    """

    def attack(self, num_budgets: Union[int, float], *, targets: Optional[Tensor] = None,
               interconnection: bool = False,
               num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None,
               feat_limits: Optional[Union[tuple, dict]] = None,
               feat_budgets:  Optional[int] = None,
               disable: bool = False) -> "RandomInjection":
        """Method to conduct adversarial injection attack with given budges.

        Parameters
        ----------
        num_budgets : Union[int, float]
            the number/percentage of nodes allowed to inject
        targets : Optional[Tensor], optional
            the targeted nodes where injected nodes perturb,
            if None, it will be all nodes in the graph, by default None
        interconnection : bool, optional
            whether the injected nodes can connect to each other, by default False
        num_edges_global : Optional[int], optional
            the number of total edges to be injected for all injected nodes, by default None
        num_edges_local : Optional[int], optional
            the number of edges allowed to inject for each injected nodes, by default None
        feat_limits : Optional[Union[tuple, dict]], optional
            the limitation or allowed budgets of injected node features,
            it can be a tuple, e.g., `(0, 1)` or 
            a dict, e.g., `{'min':0, 'max': 1}`,
        feat_budgets :  Optional[int], optional
            the number of features can be flipped for each node,
            e.g., `10`, denoting 10 features can be flipped, by default None
        disable : bool, optional
            whether the tqdm progbar is to disabled, by default False

        Returns
        -------
        RandomInjection
            the attacker itself
        """
        super().attack(num_budgets, targets=targets,
                       num_edges_global=num_edges_global,
                       num_edges_local=num_edges_local,
                       feat_limits=feat_limits,
                       feat_budgets=feat_budgets)

        candidate_nodes = copy(self.targets)

        for injected_node in tqdm(range(self.num_nodes, self.num_nodes+self.num_budgets),
                                  desc="Injecting nodes...",
                                  disable=disable):
            sampled = np.random.choice(
                candidate_nodes, self.num_edges_local, replace=False)

            self.inject_node(injected_node)

            for target in sampled:
                self.inject_edge(injected_node, target)

            if interconnection:
                candidate_nodes.append(injected_node)

        return self
