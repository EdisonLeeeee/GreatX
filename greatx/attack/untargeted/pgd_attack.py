import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from tqdm.auto import tqdm

from greatx.attack.untargeted.untargeted_attacker import UntargetedAttacker
from greatx.nn.models.surrogate import Surrogate


class PGD:
    """Base class for :class:`PGDAttack`."""
    # PGDAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton: bool = True

    def attack(
        self,
        num_budgets: int,
        victim_nodes: Tensor,
        victim_labels: Tensor,
        base_lr: float = 0.1,
        grad_clip: Optional[float] = None,
        epochs: int = 200,
        ce_loss: bool = False,
        sample_epochs: int = 20,
        disable: bool = False,
    ) -> "PGD":

        if ce_loss:
            self.loss_fn = cross_entropy_loss
        else:
            self.loss_fn = margin_loss

        perturbations = self.perturbations

        for epoch in tqdm(range(epochs), desc='PGD training...',
                          disable=disable):
            lr = base_lr * num_budgets / math.sqrt(epoch + 1)
            gradients = self.compute_gradients(perturbations, victim_nodes,
                                               victim_labels)

            gradients = self.clip_grad(gradients, grad_clip)

            with torch.no_grad():
                perturbations.data.add_(lr * gradients)
                if perturbations.clamp(0, 1).sum() <= self.num_budgets:
                    perturbations.clamp_(0, 1)
                else:
                    top = perturbations.max().item()
                    bot = (perturbations.min() - 1).clamp_min(0).item()
                    mu = (top + bot) / 2
                    while (top - bot) / 2 > 1e-5:
                        used_budget = (perturbations - mu).clamp(0, 1).sum()
                        if used_budget == self.num_budgets:
                            break
                        elif used_budget > self.num_budgets:
                            bot = mu
                        else:
                            top = mu
                        mu = (top + bot) / 2
                    perturbations.sub_(mu).clamp_(0, 1)

        best_loss = -np.inf
        best_pert = None

        perturbations.detach_()
        for it in tqdm(range(sample_epochs), desc='Bernoulli sampling...',
                       disable=disable):
            sampled = perturbations.bernoulli()
            if sampled.count_nonzero() <= self.num_budgets:
                loss = self.compute_loss(symmetric(sampled), victim_nodes,
                                         victim_labels)
                if best_loss < loss:
                    best_loss = loss
                    best_pert = sampled

        row, col = torch.where(best_pert > 0.)
        for it, (u, v) in enumerate(zip(row.tolist(), col.tolist())):
            if self.adj[u, v] > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)

        return self

    def compute_loss(
        self,
        perturbations: Tensor,
        victim_nodes: Tensor,
        victim_labels: Tensor,
    ) -> Tensor:
        adj = self.adj + perturbations * (1 - 2 * self.adj)
        logit = self.surrogate(self.feat, adj)[victim_nodes]
        if self.tau != 1:
            logit /= self.tau
        loss = self.loss_fn(logit, victim_labels)
        return loss

    def compute_gradients(
        self,
        perturbations: Tensor,
        victim_nodes: Tensor,
        victim_labels: Tensor,
    ) -> Tensor:
        pert_sym = symmetric(perturbations)
        grad_outputs = grad(
            self.compute_loss(pert_sym, victim_nodes, victim_labels), pert_sym)
        return grad(pert_sym, perturbations, grad_outputs=grad_outputs[0])[0]


class PGDAttack(UntargetedAttacker, PGD, Surrogate):
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
        name of the attacker, if None, it would be
        :obj:`__class__.__name__`, by default None
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

        dataset = GraphDataset(root='.', name='Cora',
                                transform=T.LargestConnectedComponents())
        data = dataset[0]

        surrogate_model = ... # train your surrogate model

        from greatx.attack.untargeted import PGDAttack
        attacker = PGDAttack(data)
        attacker.setup_surrogate(surrogate_model,
                                 victim_nodes=test_nodes)
        attacker.reset()
        attacker.attack(0.05) # attack with 0.05% of edge perturbations
        attacker.data() # get attacked graph

        attacker.edge_flips() # get edge flips after attack

        attacker.added_edges() # get added edges after attack

        attacker.removed_edges() # get removed edges after attack

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.

    """

    # PGDAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton: bool = True

    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        victim_nodes: Tensor,
        ground_truth: bool = False,
        *,
        tau: float = 1.0,
        freeze: bool = True,
    ) -> "PGDAttack":
        """Setup the surrogate model for adversarial attack.

        Parameters
        ----------
        surrogate : torch.nn.Module
            the surrogate model
        victim_nodes : Tensor
            the victim nodes_set
        ground_truth : bool, optional
            whether to use ground-truth label for victim nodes,
            if False, the node labels are estimated by the surrogate model,
            by default False
        tau : float, optional
            the temperature of softmax activation, by default 1.0
        freeze : bool, optional
            whether to free the surrogate model to avoid the
            gradient accumulation, by default True

        Returns
        -------
        PGDAttack
            the attacker itself
        """

        Surrogate.setup_surrogate(self, surrogate=surrogate, tau=tau,
                                  freeze=freeze)

        if victim_nodes.dtype == torch.bool:
            victim_nodes = victim_nodes.nonzero().view(-1)
        self.victim_nodes = victim_nodes.to(self.device)

        if ground_truth:
            self.victim_labels = self.label[victim_nodes]
        else:
            self.victim_labels = self.estimate_self_training_labels(
                victim_nodes)

        self.adj = self.get_dense_adj()
        return self

    def reset(self) -> "PGDAttack":
        super().reset()
        self.perturbations = torch.zeros_like(self.adj).requires_grad_()
        return self

    def attack(
        self,
        num_budgets: Union[int, float] = 0.05,
        *,
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
        num_budgets : Union[int, float], optional
            the number of attack budgets, coubd be float (ratio)
            or int (number), by default 0.05
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

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        return PGD.attack(
            self,
            self.num_budgets,
            victim_nodes=self.victim_nodes,
            victim_labels=self.victim_labels,
            base_lr=base_lr,
            grad_clip=grad_clip,
            epochs=epochs,
            ce_loss=ce_loss,
            sample_epochs=sample_epochs,
            disable=disable,
        )


def symmetric(x: Tensor) -> Tensor:
    x = x.triu(diagonal=1)
    return x + x.T


def margin_loss(logit: Tensor, y_true: Tensor) -> Tensor:
    all_nodes = torch.arange(y_true.size(0))
    # Get the scores of the true classes.
    scores_true = logit[all_nodes, y_true]
    # Get the highest scores when not considering the true classes.
    scores_mod = logit.clone()
    scores_mod[all_nodes, y_true] = -np.inf
    scores_pred_excl_true = scores_mod.amax(dim=-1)
    return -(scores_true - scores_pred_excl_true).tanh().mean()


def cross_entropy_loss(logit: Tensor, y_true: Tensor) -> Tensor:
    return F.cross_entropy(logit, y_true)
