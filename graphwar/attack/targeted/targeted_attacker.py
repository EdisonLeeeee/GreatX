from numbers import Number

from graphwar.attack.flip_attacker import FlipAttacker


class TargetedAttacker(FlipAttacker):

    def reset(self) -> "TargetedAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        TargetedAttacker
            the attacker itself
        """
        super().reset()
        self.target = None
        self.target_label = None
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None
        self.direct_attack = None

        return self

    def attack(self, target, target_label, num_budgets, direct_attack,
               structure_attack, feature_attack) -> "TargetedAttacker":
        """Base method that describes the adversarial targeted attack

        Parameters
        ----------
        target : int
            the target node to be attacked
        target_label: int
            the label of the target node.
        num_budgets : int (0<`num_budgets`<=:attr:max_perturbations) or float (0<`num_budgets`<=1)
            Case 1:
            `int` : the number of attack budgets, 
            i.e., how many edges can be perturbed.

            Case 2:
            `float`: the number of attack budgets is 
            the ratio of :attr:max_perturbations

            See `:attr:max_perturbations`

        direct_attack : bool
            whether to conduct direct attack or indirect attack.
        structure_attack : bool
            whether to conduct structure attack, i.e., modify the graph structure (edges)
        feature_attack : bool
            whether to conduct feature attack, i.e., modify the node features

        """

        if not self.is_reseted:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        if not isinstance(target, Number):
            raise ValueError(target)

        if target_label is not None and not isinstance(target_label, Number):
            raise ValueError(target_label)

        if not (structure_attack or feature_attack):
            raise RuntimeError(
                'Either `structure_attack` or `feature_attack` must be True.')

        if feature_attack and not self._allow_feature_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking features."
                " If the model can conduct feature attack, please call `attacker.set_allow_feature_attack(True)`."
            )

        if structure_attack and not self._allow_structure_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking structures."
                " If the model can conduct structure attack, please call `attacker.set_allow_structure_attack(True)`."
            )

        max_perturbations = int(self._degree[target].item())
        if num_budgets is None:
            num_budgets = max_perturbations
        else:
            num_budgets = self._check_budget(
                num_budgets, max_perturbations=max_perturbations)

        self.target = target

        if target_label is None and self.label is not None:
            self.target_label = self.label[target]
        else:
            self.target_label = target_label

        self.num_budgets = num_budgets
        self.direct_attack = direct_attack
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack

        self.is_reseted = False

        return self

    def is_legal_edge(self, u: int, v: int) -> bool:
        """Check whether the edge (u,v) is legal.

        For targeted attacker, an edge (u,v) is legal
        if u!=v and edge (u,v) is not selected before.

        In addition, if the setting is `indirect attack`,
        the targeted node is not allowed to be u or v.

        Parameters
        ----------
        u : int
            src node id
        v : int
            dst node id

        Returns
        -------
        bool
            True if the u!=v and edge (u,v) is not selected, otherwise False.
        """

        condition = super().is_legal_edge(u, v)
        if self.direct_attack:
            return condition
        else:
            return (condition and self.target not in (u, v))
