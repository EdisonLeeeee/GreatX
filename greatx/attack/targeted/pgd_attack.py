from typing import Optional, Union

import torch

from greatx.attack.targeted.targeted_attacker import TargetedAttacker
from greatx.attack.untargeted.pgd_attack import PGD
from greatx.nn.models.surrogate import Surrogate


class PGDAttack(TargetedAttacker, PGD, Surrogate):
    r"""Implementation of `PGD` attack from the:
    `"Topology Attack and Defense for Graph Neural Networks:
    An Optimization Perspective"
    <https://arxiv.org/abs/1906.04214>`_ paper (IJCAI'19)

    Parameters
    ----------
    data : Data
        PyG-like data denoting the input graph
    device : str, optional
        the device of the attack running on, by default "cpu"
    seed : Optional[int], optional
        the random seed for reproducing the attack, by default None
    name : Optional[str], optional
        name of the attacker, if None, it would be :obj:`__class__.__name__`,
        by default None
    kwargs : additional arguments of :class:`greatx.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`

    Example
    -------
    .. code-block:: python

        from greatx.dataset import GraphDataset
        import torch_geometric.transforms as T

        import os.path as osp

        dataset = GraphDataset(root='.', name='Cora',
                                transform=T.LargestConnectedComponents())
        data = dataset[0]

        surrogate_model = ... # train your surrogate model

        from greatx.attack.targeted import PGDAttack
        attacker = PGDAttack(data)
        attacker.setup_surrogate(surrogate_model)
        attacker.reset()
        # attacking target node `1` with default budget set as node degree
        attacker.attack(target=1)

        attacker.reset()
        # attacking target node `1` with budget set as 1
        attacker.attack(target=1, num_budgets=1)

        attacker.data() # get attacked graph

        attacker.edge_flips() # get edge flips after attack

        attacker.added_edges() # get added edges after attack

        attacker.removed_edges() # get removed edges after attack
    """

    # PGDAttack cannot ensure there are no singleton nodes
    _allow_singleton: bool = True

    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        *,
        tau: float = 1.0,
        freeze: bool = True,
    ) -> "PGDAttack":
        Surrogate.setup_surrogate(self, surrogate=surrogate, tau=tau,
                                  freeze=freeze)
        self.adj = self.get_dense_adj()
        return self

    def reset(self) -> "PGDAttack":
        super().reset()
        self.perturbations = torch.zeros_like(self.adj).requires_grad_()
        return self

    def attack(
        self,
        target: int,
        *,
        target_label: Optional[int] = None,
        num_budgets: Optional[Union[float, int]] = None,
        direct_attack: bool = True,
        base_lr: float = 0.1,
        grad_clip: Optional[float] = None,
        epochs: int = 200,
        ce_loss: bool = False,
        sample_epochs: int = 20,
        structure_attack: bool = True,
        feature_attack: bool = False,
        disable: bool = False,
    ) -> "PGDAttack":
        """Adversarial attack method for
        "Project gradient descent attack (PGD)"

        Parameters
        ----------
        target : int
            the target node to attack
        target_label : Optional[int], optional
            the label of the target node, if None,
            it defaults to its ground truth label,
            by default None
        direct_attack : bool, optional
            whether to conduct direct attack on the target,
            N/A for this method when :obj:`direct_attack=False`.
        num_budgets : Union[int, float], optional
            the number of attack budgets, coubd be float (ratio)
            or int (number), if None, it defaults to the number of
            node degree of :obj:`target`
            by default None
        base_lr : float, optional
            the base learning rate for PGD training, by default 0.1
        grad_clip : float, optional
            gradient clipping for the computed gradients,
            by default None
        epochs : int, optional
            the number of epochs for PGD training, by default 200
        ce_loss : bool, optional
            whether to use cross-entropy loss (True) or
            margin loss (False), by default False
        sample_epochs : int, optional
            the number of sampling epochs for learned perturbations,
            by default 20
        structure_attack : bool, optional
            whether to conduct structure attack, i.e.,
            modify the graph structure (edges),
            by default True
        feature_attack : bool, optional
            whether to conduct feature attack, i.e.,
            modify the node features, N/A for this method.
            by default False
        disable : bool, optional
            whether to disable the tqdm progress bar,
            by default False

        Returns
        -------
        PGDAttack
            the attacker itself
        """
        if not direct_attack:
            raise RuntimeError(
                "PGDAttack is not applicable to indirect attack.")

        super().attack(target, target_label, num_budgets=num_budgets,
                       direct_attack=direct_attack,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        victim_label = self.target_label.view(-1)
        victim_node = torch.as_tensor(self.target, device=self.device,
                                      dtype=torch.long).view(-1)
        return PGD.attack(
            self,
            self.num_budgets,
            victim_nodes=victim_node,
            victim_labels=victim_label,
            base_lr=base_lr,
            grad_clip=grad_clip,
            epochs=epochs,
            ce_loss=ce_loss,
            sample_epochs=sample_epochs,
            disable=disable,
        )
