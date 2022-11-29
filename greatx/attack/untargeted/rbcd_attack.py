from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import coalesce, to_undirected
from tqdm.auto import tqdm

from greatx.attack.untargeted.untargeted_attacker import UntargetedAttacker
from greatx.attack.untargeted.utils import (
    linear_to_full_idx,
    linear_to_triu_idx,
    num_possible_edges,
    project,
)
from greatx.functional import (
    masked_cross_entropy,
    probability_margin_loss,
    tanh_margin_loss,
)
from greatx.nn.models.surrogate import Surrogate

# (predictions, labels, ids/mask) -> Tensor with one element
METRIC = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class RBCDAttack:
    """Base class for :class:`PRBCDAttack` and
    :class:`GRBCDEAttack`."""

    # RBCDAttack will not ensure there are no singleton nodes
    _allow_singleton: bool = False

    # TODO: Although RBCDAttack accepts directed graphs,
    # we currently don't explicitlyt support directed graphs.
    # This should be made available in the future.
    is_undirected: bool = True

    coeffs: Dict[str, Any] = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'with_early_stopping': True,
        'eps': 1e-7
    }

    def compute_gradients(
        self,
        feat: Tensor,
        victim_labels: Tensor,
        victim_nodes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad_()

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6 / Algorithm 2, line 7)
        edge_index, edge_weight = self.get_modified_graph(
            self._edge_index, self._edge_weight, self.block_edge_index,
            self.block_edge_weight)

        # Get prediction (Algorithm 1, line 6 / Algorithm 2, line 7)
        prediction = self.surrogate(feat, edge_index,
                                    edge_weight)[victim_nodes]

        # temperature scaling, work for cross-entropy loss
        if self.tau != 1:
            prediction /= self.tau

        # Calculate loss combining all each node
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        loss = self.loss(prediction, victim_labels)
        # Retrieve gradient towards the current block
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        return loss, gradient

    def get_modified_graph(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        block_edge_index: Tensor,
        block_edge_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Merges adjacency matrix with current block (incl. weights)"""
        if self.is_undirected:
            block_edge_index, block_edge_weight = to_undirected(
                block_edge_index, block_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')

        modified_edge_index = torch.cat((edge_index, block_edge_index), dim=-1)
        modified_edge_weight = torch.cat((edge_weight, block_edge_weight))

        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index, modified_edge_weight,
            num_nodes=self.num_nodes, reduce='sum')

        # Allow (soft) removal of edges
        mask = modified_edge_weight > 1
        modified_edge_weight[mask] = 2 - modified_edge_weight[mask]

        return modified_edge_index, modified_edge_weight

    @torch.no_grad()
    def sample_random_block(self, num_budgets: int = 0):
        for _ in range(self.coeffs['max_trials_sampling']):
            num_possible = num_possible_edges(self.num_nodes,
                                              self.is_undirected)
            self.current_block = torch.randint(num_possible,
                                               (self.block_size, ),
                                               device=self.device)
            self.current_block = torch.unique(self.current_block, sorted=True)

            if self.is_undirected:
                self.block_edge_index = linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = linear_to_full_idx(
                    self.num_nodes, self.current_block)

                self._filter_self_loops_in_block(with_weight=False)

            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)
            if self.current_block.size(0) >= num_budgets:
                return

        raise RuntimeError("Sampling random block was not successful. "
                           "Please decrease `num_budgets`.")

    def resample_random_block(self, num_budgets: int):
        # Keep at most half of the block (i.e. resample low weights)
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight <=
                      self.coeffs['eps']).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]

        # Sample until enough edges were drawn
        for _ in range(self.coeffs['max_trials_sampling']):
            n_edges_resample = self.block_size - self.current_block.size(0)
            num_possible = num_possible_edges(self.num_nodes,
                                              self.is_undirected)
            lin_index = torch.randint(num_possible, (n_edges_resample, ),
                                      device=self.device)

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True)

            if self.is_undirected:
                self.block_edge_index = linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = linear_to_full_idx(
                    self.num_nodes, self.current_block)

            # Merge existing weights with new edge weights
            block_edge_weight_prev = self.block_edge_weight[sorted_idx]
            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)

            self.block_edge_weight[
                unique_idx[:sorted_idx.size(0)]] = block_edge_weight_prev

            if not self.is_undirected:
                self._filter_self_loops_in_block(with_weight=True)

            if self.current_block.size(0) > num_budgets:
                return

        raise RuntimeError("Sampling random block was not successful."
                           "Please decrease `num_budgets`.")

    @torch.no_grad()
    def sample_final_edges(
        self,
        feat: Tensor,
        num_budgets: int,
        victim_nodes: Tensor,
        victim_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        best_metric = float('-Inf')
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0

        for i in range(self.coeffs['max_final_samples']):
            if i == 0:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(block_edge_weight)
                sampled_edges[torch.topk(block_edge_weight,
                                         num_budgets).indices] = 1
            else:
                sampled_edges = torch.bernoulli(block_edge_weight).float()

            if sampled_edges.sum() > num_budgets:
                # Allowed num_budgets is exceeded
                continue

            self.block_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_graph(
                self._edge_index, self._edge_weight, self.block_edge_index,
                self.block_edge_weight)
            prediction = self.surrogate(feat, edge_index,
                                        edge_weight)[victim_nodes]
            metric = self.metric(prediction, victim_labels)

            # Save best sample
            if metric > best_metric:
                best_metric = metric
                best_edge_weight = self.block_edge_weight.clone().cpu()

        flipped_edges = self.block_edge_index[:, best_edge_weight != 0]
        return flipped_edges

    def update_edge_weights(self, num_budgets: int, epoch: int,
                            gradient: Tensor):
        # The learning rate is refined heuristically, s.t. (1) it is
        # independent of the number of perturbations (assuming an undirected
        # adjacency matrix) and (2) to decay learning rate during fine-tuning
        # (i.e. fixed search space).
        lr = (num_budgets / self.num_nodes * self.lr /
              np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        self.block_edge_weight.data.add_(lr * gradient)

    def _filter_self_loops_in_block(self, with_weight: bool):
        mask = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[mask]
        self.block_edge_index = self.block_edge_index[:, mask]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[mask]

    def _append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)


class PRBCDAttack(UntargetedAttacker, RBCDAttack, Surrogate):
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
    def setup_surrogate(
        self,
        surrogate: torch.nn.Module,
        victim_nodes: Tensor,
        ground_truth: bool = True,
        *,
        tau: float = 1.0,
        freeze: bool = True,
    ) -> "PRBCDAttack":
        r"""Setup the surrogate model for adversarial attack.

        Parameters
        ----------
        surrogate : torch.nn.Module
            the surrogate model
        victim_nodes : Tensor
            the victim nodes_set
        ground_truth : bool, optional
            whether to use ground-truth label for victim nodes,
            if False, the node labels are estimated by the surrogate model,
            by default True
        tau : float, optional
            the temperature of softmax activation, by default 1.0
        freeze : bool, optional
            whether to free the surrogate model to avoid the
            gradient accumulation, by default True

        Returns
        -------
        PRBCDAttack
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

        return self

    def reset(self) -> "PRBCDAttack":
        super().reset()
        self.current_block = None
        self.block_edge_index = None
        self.block_edge_weight = None
        self.loss = None
        self.metric = None

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
        num_budgets: Union[int, float] = 0.05,
        *,
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

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        self.block_size = block_size

        assert loss in ['mce', 'prob_margin', 'tanh_margin']
        if loss == 'mce':
            self.loss = masked_cross_entropy
        elif loss == 'prob_margin':
            self.loss = probability_margin_loss
        else:
            self.loss = tanh_margin_loss

        if metric is None:
            self.metric = self.loss
        else:
            self.metric = metric

        self.epochs_resampling = epochs_resampling
        self.lr = lr

        self.coeffs.update(**kwargs)

        num_budgets = self.num_budgets
        feat, victim_nodes, victim_labels = (self.feat, self.victim_nodes,
                                             self.victim_labels)

        # Loop over the epochs (Algorithm 1, line 5)
        for step in tqdm(self.prepare(num_budgets, epochs),
                         desc='Peturbing graph...', disable=disable):

            loss, gradient = self.compute_gradients(feat, victim_labels,
                                                    victim_nodes)

            scalars = self.update(step, gradient, num_budgets)

            scalars['loss'] = loss.item()
            self._append_statistics(scalars)

        flipped_edges = self.get_flipped_edges()

        assert flipped_edges.size(1) <= self.num_budgets, (
            f'# perturbed edges {flipped_edges.size(1)} '
            f'exceeds num_budgets {self.num_budgets}')

        for it, (u, v) in enumerate(zip(*flipped_edges.tolist())):
            if self.adjacency_matrix[u, v] > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)

        return self

    def prepare(self, num_budgets: int, epochs: int) -> Iterable[int]:
        """Prepare attack and return the iterable sequence steps."""

        # Sample initial search space (Algorithm 1, line 3-4)
        self.sample_random_block(num_budgets)

        return range(epochs)

    @torch.no_grad()
    def update(self, epoch: int, gradient: Tensor,
               num_budgets: int) -> Dict[str, float]:
        """Update edge weights given gradient."""
        # Gradient update step (Algorithm 1, line 7)
        self.update_edge_weights(num_budgets, epoch, gradient)

        # For monitoring
        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        # Projection to stay within relaxed `L_0` num_budgets
        # (Algorithm 1, line 8)
        self.block_edge_weight = project(num_budgets, self.block_edge_weight,
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
                                          num_budgets).indices] = 1

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
            self.resample_random_block(num_budgets)
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
        return self.sample_final_edges(
            self.feat,
            self.num_budgets,
            self.victim_nodes,
            self.victim_labels,
        )


class GRBCDAttack(PRBCDAttack):
    r"""Greedy Randomized Block Coordinate Descent (GRBCD) adversarial attack
    from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    GRBCD shares most of the properties and requirements with
    :class:`PRBCDAttack`. It also uses an efficient gradient based approach.
    However, it greedily flips edges based on the gradient towards the
    adjacency matrix.
    """
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
        num_budgets: int,
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
