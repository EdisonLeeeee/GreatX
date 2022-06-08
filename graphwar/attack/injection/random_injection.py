from copy import copy
from typing import Optional, Union

import numpy as np
from torch import Tensor
from tqdm import tqdm
from graphwar.attack.injection.injection_attacker import InjectionAttacker


class RandomInjection(InjectionAttacker):
    r"""Injection nodes into a graph randomly.

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]


    >>> from graphwar.attack.injection import RandomInjection
    >>> attacker = RandomInjection(data)

    >>> attacker.reset()
    >>> attacker.attack(10, feat_limits=(0, 1))  # injecting 10 nodes for continuous features

    >>> attacker.reset()
    >>> attacker.attack(10, feat_budgets=10)  # injecting 10 nodes for binary features    

    >>> attacker.data() # get attacked graph

    >>> attacker.injected_nodes() # get injected nodes after attack

    >>> attacker.injected_edges() # get injected edges after attack

    >>> attacker.injected_feats() # get injected features after attack   

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.       
    """

    def attack(self, num_budgets: Union[int, float], *,
               targets: Optional[Tensor] = None,
               interconnection: bool = False,
               num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None,
               feat_limits: Optional[Union[tuple, dict]] = None,
               feat_budgets:  Optional[int] = None,
               disable: bool = False) -> "RandomInjection":
        """Base method that describes the adversarial injection attack

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
            the number of total edges in the graph to be injected for 
            all injected nodes, by default None
        num_edges_local : Optional[int], optional
            the number of edges allowed to inject for each injected nodes, by default None
        feat_limits : Optional[Union[tuple, dict]], optional
            the limitation or allowed budgets of injected node features,
            it can be a tuple, e.g., `(0, 1)` or 
            a dict, e.g., `{'min':0, 'max': 1}`.
            if None, it is set as (self.feat.min(), self.feat.max()), by default None
        feat_budgets :  Optional[int], optional
            the number of nonzero features can be injected for each node,
            e.g., `10`, denoting 10 nonzero features can be injected, by default None
        disable : bool, optional
            whether the tqdm progbar is to disabled, by default False

        Returns
        -------
        the attacker itself

        Note
        ----
        * Both `num_edges_local` and `num_edges_global` cannot be used simultaneously.
        * Both `feat_limits` and `feat_budgets` cannot be used simultaneously.
        """
        super().attack(num_budgets, targets=targets,
                       num_edges_global=num_edges_global,
                       num_edges_local=num_edges_local,
                       feat_limits=feat_limits,
                       feat_budgets=feat_budgets)

        candidate_nodes = self.targets.tolist()

        for injected_node in tqdm(range(self.num_nodes, self.num_nodes+self.num_budgets),
                                  desc="Injecting nodes...",
                                  disable=disable):
            sampled = np.random.choice(
                candidate_nodes, self.num_edges_local, replace=False)

            self.inject_node(injected_node)
            self.inject_feat()

            for target in sampled:
                self.inject_edge(injected_node, target)

            if interconnection:
                candidate_nodes.append(injected_node)

        return self
