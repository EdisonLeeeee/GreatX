from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.utils import coalesce, to_undirected

from greatx.attack.targeted.targeted_attacker import TargetedAttacker
from greatx.attack.untargeted.rbcd_attack import RBCDAttack
from greatx.attack.untargeted.utils import project
from greatx.nn.models.surrogate import Surrogate

# (predictions, labels, ids/mask) -> Tensor with one element
METRIC = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class PRBCDAttack(TargetedAttacker, RBCDAttack, Surrogate):
    r"""Projected Randomized Block Coordinate Descent (PRBCD) adversarial
    attack from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    This attack uses an efficient gradient based approach that (during the
    attack) relaxes the discrete entries in the adjacency matrix
    :math:`\{0, 1\}` to :math:`[0, 1]` and solely perturbs the adjacency matrix
    (no feature perturbations). Thus, this attack supports all models that can
    handle weighted graphs that are differentiable w.r.t. these edge weights.
    For non-differentiable models you might be able to e.g. use the gumble
    softmax trick.

    The memory overhead is driven by the additional edges (at most
    :attr:`block_size`). For scalability reasons, the block is drawn with
    replacement and then the index is made unique. Thus, the actual block size
    is typically slightly smaller than specified.

    This attack can be used for both global and local attacks as well as
    test-time attacks (evasion) and training-time attacks (poisoning). Please
    see the provided examples.

    This attack is designed with a focus on node- or graph-classification,
    however, to adapt to other tasks you most likely only need to provide an
    appropriate loss and model. However, we currently do not support batching
    out of the box (sampling needs to be adapted).

    """
    def reset(self) -> "PRBCDAttack":
        super().reset()
        self.current_block = None
        self.block_edge_index = None
        self.block_edge_weight = None
        self.loss = None
        self.metric = None

        self.victim_nodes = None
        self.victim_labels = None

        # NOTE: Since `edge_index` and `edge_weight` denote the original graph
        # here we need to name them as `edge_index`and `_edge_weight`
        self._edge_index = self.edge_index
        self._edge_weight = torch.ones(self.num_edges, device=self.device)

        # For early stopping (not explicitly covered by pseudo code)
        self.best_metric = float('-Inf')

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        return self

    def attack(
        self,
        target,
        *,
        target_label=None,
        num_budgets=None,
        direct_attack=True,
        block_size: int = 250_000,
        epochs: int = 125,
        epochs_resampling: int = 100,
        loss: Optional[str] = 'tanh_margin',
        metric: Optional[Union[str, METRIC]] = None,
        lr: float = 2_000,
        structure_attack: bool = True,
        feature_attack: bool = False,
        disable: bool = False,
        **kwargs,
    ) -> "PRBCDAttack":

        super().attack(target, target_label, num_budgets=num_budgets,
                       direct_attack=direct_attack,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)
        self.victim_nodes = torch.as_tensor(
            target,
            dtype=torch.long,
            device=self.device,
        ).view(-1)

        self.victim_labels = self.target_label.view(-1)

        return RBCDAttack.attack(self, block_size=block_size, epochs=epochs,
                                 epochs_resampling=epochs_resampling,
                                 loss=loss, metric=metric, lr=lr,
                                 disable=disable, **kwargs)

    def prepare(self, num_budgets: int, epochs: int) -> Iterable[int]:
        """Prepare attack and return the iterable sequence steps."""

        # Sample initial search space (Algorithm 1, line 3-4)
        self.sample_random_block(num_budgets)

        return range(epochs)

    @torch.no_grad()
    def update(self, epoch: int, gradient: Tensor) -> Dict[str, float]:
        """Update edge weights given gradient."""
        # Gradient update step (Algorithm 1, line 7)
        self.update_edge_weights(epoch, gradient)

        # For monitoring
        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        # Projection to stay within relaxed `L_0` num_budgets
        # (Algorithm 1, line 8)
        self.block_edge_weight = project(self.num_budgets,
                                         self.block_edge_weight,
                                         self.coeffs['eps'])

        # For monitoring
        scalars = dict(
            prob_mass_after_update=pmass_update.sum().item(),
            prob_mass_after_update_max=pmass_update.max().item(),
            prob_mass_afterprojection=self.block_edge_weight.sum().item(),
            prob_mass_afterprojection_nonzero_weights=(
                self.block_edge_weight > self.coeffs['eps']).sum().item(),
            prob_mass_afterprojection_max=self.block_edge_weight.max().item(),
        )

        if not self.coeffs['with_early_stopping']:
            return scalars

        # Calculate metric after the current epoch (overhead
        # for monitoring and early stopping)

        topk_block_edge_weight = torch.zeros_like(self.block_edge_weight)
        topk_block_edge_weight[torch.topk(self.block_edge_weight,
                                          self.num_budgets).indices] = 1

        edge_index, edge_weight = self.get_modified_graph(
            self._edge_index, self._edge_weight, self.block_edge_index,
            topk_block_edge_weight)

        prediction = self.surrogate(self.feat, edge_index,
                                    edge_weight)[self.victim_nodes]
        metric = self.metric(prediction, self.victim_labels)

        # Save best epoch for early stopping
        # (not explicitly covered by pseudo code)
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_block = self.current_block.cpu().clone()
            self.best_edge_index = self.block_edge_index.cpu().clone()
            self.best_pert_edge_weight = self.block_edge_weight.cpu().detach()

        # Resampling of search space (Algorithm 1, line 9-14)
        if epoch < self.epochs_resampling - 1:
            self.resample_random_block(self.num_budgets)
        elif epoch == self.epochs_resampling - 1:
            # Retrieve best epoch if early stopping is active
            # (not explicitly covered by pseudo code)
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            block_edge_weight = self.best_pert_edge_weight.clone()
            self.block_edge_weight = block_edge_weight.to(self.device)

        scalars['metric'] = metric.item()
        return scalars

    def get_flipped_edges(self) -> Tensor:
        """Clean up and prepare return flipped edges."""

        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs['with_early_stopping']:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        return self.sample_final_edges()


class GRBCDAttack(PRBCDAttack):
    r"""Greedy Randomized Block Coordinate Descent (GRBCD) adversarial attack
    from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    GRBCD shares most of the properties and requirements with
    :class:`PRBCDAttack`. It also uses an efficient gradient based approach.
    However, it greedily flips edges based on the gradient towards the
    adjacency matrix.
    """
    def attack(
        self,
        target,
        *,
        target_label=None,
        num_budgets=None,
        direct_attack=True,
        block_size: int = 250_000,
        epochs: int = 125,
        epochs_resampling: int = 100,
        loss: Optional[str] = 'mce',
        metric: Optional[Union[str, METRIC]] = None,
        lr: float = 1_000,
        structure_attack: bool = True,
        feature_attack: bool = False,
        disable: bool = False,
        **kwargs,
    ) -> "GRBCDAttack":

        return super().attack(target=target, target_label=target_label,
                              direct_attack=direct_attack,
                              num_budgets=num_budgets, block_size=block_size,
                              epochs=epochs,
                              epochs_resampling=epochs_resampling,
                              metric=metric, loss=loss, lr=lr, disable=disable,
                              structure_attack=structure_attack,
                              feature_attack=feature_attack, **kwargs)

    def prepare(self, num_budgets: int, epochs: int) -> List[int]:
        """Prepare attack."""

        # Determine the number of edges to be flipped in each attach step/epoch
        step_size = num_budgets // epochs
        if step_size > 0:
            steps = epochs * [step_size]
            for i in range(num_budgets % epochs):
                steps[i] += 1
        else:
            steps = [1] * num_budgets

        # Sample initial search space (Algorithm 2, line 3-4)
        self.sample_random_block(step_size)

        return steps

    def reset(self) -> "GRBCDAttack":
        super().reset()
        self.flipped_edges = self._edge_index.new_empty(2, 0)
        return self

    @torch.no_grad()
    def update(
        self,
        step_size: int,
        gradient: Tensor,
    ) -> Dict[str, Any]:
        """Update edge weights given gradient."""
        _, topk_edge_index = torch.topk(gradient, step_size)

        flip_edge_index = self.block_edge_index[:, topk_edge_index].to(
            self.device)
        flip_edge_weight = torch.ones(flip_edge_index.size(1),
                                      device=self.device)

        self.flipped_edges = torch.cat((self.flipped_edges, flip_edge_index),
                                       axis=-1)

        if self.is_undirected:
            flip_edge_index, flip_edge_weight = to_undirected(
                flip_edge_index, flip_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')

        edge_index = torch.cat((self._edge_index, flip_edge_index), dim=-1)
        edge_weight = torch.cat((self._edge_weight, flip_edge_weight))

        edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                           num_nodes=self.num_nodes,
                                           reduce='sum')

        mask = torch.isclose(edge_weight, torch.tensor(1.))

        self._edge_index = edge_index[:, mask]
        self._edge_weight = edge_weight[mask]

        # Sample initial search space (Algorithm 2, line 3-4)
        self.sample_random_block(step_size)

        # Return debug information
        scalars = {
            'number_positive_entries_in_gradient': (gradient > 0).sum().item()
        }
        return scalars

    def get_flipped_edges(self) -> Tensor:
        """Clean up and prepare return flipped edges."""
        return self.flipped_edges
