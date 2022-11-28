from greatx.attack.flip_attacker import FlipAttacker


class UntargetedAttacker(FlipAttacker):
    r"""Base class for adversarial non-targeted attack.

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be
        :obj:`__class__.__name__`, by default None
    kwargs : additional arguments of :class:`greatx.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`

    Note
    ----
    :class:`greatx.attack.targeted.UntargetedAttacker` is a subclass of
    :class:`greatx.attack.FlipAttacker`.
    It belongs to graph modification attack (GMA).
    """
    def reset(self) -> "UntargetedAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        UntargetedAttacker
            the attacker itself
        """
        super().reset()
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None
        return self

    def attack(self, num_budgets, structure_attack,
               feature_attack) -> "UntargetedAttacker":
        """Base method that describes the adversarial untargeted attack.

        Parameters
        ----------
        num_budgets : int or float
            the number/percentage of perturbations allowed to attack
        structure_attack : bool, optional
            whether to conduct structure attack, i.e.,
            modify the graph structure (edges),
        feature_attack : bool, optional
            whether to conduct feature attack, i.e.,
            modify the node features,
        """

        _is_setup = getattr(self, "_is_setup", True)

        if not _is_setup:
            raise RuntimeError(
                f'{self.__class__.__name__} requires '
                'a surrogate model to conduct attack. '
                'Use `attacker.setup_surrogate(surrogate_model)`.')

        if not self._is_reset:
            raise RuntimeError('Before calling attack, you must reset '
                               'your attacker. Call `attacker.reset()`.')

        if not (structure_attack or feature_attack):
            raise RuntimeError(
                'Either `structure_attack` or `feature_attack` must be True.')

        if feature_attack and not self._allow_feature_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking features.")

        if structure_attack and not self._allow_structure_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking structures.")

        num_budgets = self._check_budget(num_budgets,
                                         max_perturbations=self.num_edges // 2)

        self.num_budgets = num_budgets
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack

        self._is_reset = False

        return self
