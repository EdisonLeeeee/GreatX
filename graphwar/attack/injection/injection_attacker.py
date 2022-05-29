from functools import lru_cache
from typing import Optional, Union

import torch
from torch import Tensor

from graphwar.attack.attacker import Attacker
from torch_geometric.data import Data


class InjectionAttacker(Attacker):

    def reset(self) -> "InjectionAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        InjectionAttacker
            the attacker itself
        """
        self.num_budgets = None
        self._added_edges = {}
        self._added_nodes = {}
        self.data.cache_clear()

        return self

    def attack(self, num_budgets: Union[int, float], *, targets: Optional[Tensor] = None, num_edges_global: Optional[int] = None,
               num_edges_local: Optional[int] = None, feat_limits: Optional[Union[tuple, dict]] = None) -> "InjectionAttacker":
        """Base method that describes the adversarial injection attack
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
            num_budgets, max_perturbations=self.num_nodes)

        self.num_budgets = num_budgets
        self.num_edges_global = num_edges_global
        self.num_edges_local = num_edges_local

        # ============== get feature limitation of injected node ==============
        min_limits = max_limits = None

        if feat_limits is not None:
            if isinstance(feat_limits, tuple):
                min_limits, max_limits = feat_limits
            elif isinstance(feat_limits, dict):
                min_limits = feat_limits.pop('min', None)
                max_limits = feat_limits.pop('max', None)
                if feat_limits:
                    raise ValueError(
                        f"Unrecognized key {next(iter(feat_limits.keys()))}.")
            else:
                raise TypeError(
                    f"`feat_limits` should be an instance of tuple and dict, but got {feat_limits}.")

        feat = self.feat

        if min_limits is None and feat is not None:
            min_limits = feat.min()

        if max_limits is None and feat is not None:
            max_limits = feat.max()
        # ======================================================================

        self.feat_limits = min_limits, max_limits
        if targets is None:
            self.targets = torch.arange(self.num_nodes, device=self.device)
        else:
            self.targets = torch.LongTensor([targets]).view(-1).to(self.device)
        self._is_reset = False

        return self

    def added_nodes(self,):
        ...

    def added_edges(self):
        ...

    def injections(self):
        ...

    @lru_cache(maxsize=1)
    def data(self, symmetric: bool = True) -> Data:
        """return the attacked graph

        Parameters
        ----------
        symmetric : bool
            Determine whether the resulting graph is forcibly symmetric

        Returns
        -------
        Data
            the attacked graph
        """
        graph = self.graph.local_var()
        added_nodes = self.added_nodes()
        added_edges = self.added_edges()
        graph.add_nodes(added_nodes)
        if symmetric:
            added_edges = ...
        graph.add_edges(added_edges[0], added_edges[1])

        return graph
