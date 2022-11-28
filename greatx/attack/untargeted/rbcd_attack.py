from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import coalesce, to_undirected
from tqdm.auto import tqdm

from greatx.attack.untargeted.untargeted_attacker import UntargetedAttacker
from greatx.functional import (
    margin_loss,
    masked_cross_entropy,
    probability_margin_loss,
    tanh_margin_loss,
)
from greatx.nn.models.surrogate import Surrogate

# (predictions, labels, ids/mask) -> Tensor with one element
LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class PRBCDAttack(UntargetedAttacker, Surrogate):
    # FGAttack can conduct feature attack
    _allow_feature_attack: bool = True
    is_undirected_graph: bool = True  # TODO

    coeffs: Dict[str, Any] = {
        'max_final_samples': 20,
        'max_trials_sampling': 20,
        'with_early_stopping': True,
        'eps': 1e-7
    }

    def setup_surrogate(self, surrogate: torch.nn.Module, victim_nodes: Tensor,
                        victim_labels: Optional[Tensor] = None, *,
                        eps: float = 1.0):

        Surrogate.setup_surrogate(self, surrogate=surrogate, eps=eps,
                                  freeze=True)

        if victim_nodes.dtype == torch.bool:
            victim_nodes = victim_nodes.nonzero().view(-1)
        self.victim_nodes = victim_nodes.to(self.device)

        if victim_labels is None:
            victim_labels = self.label[victim_nodes]
        self.victim_labels = victim_labels.to(self.device)
        return self

    def reset(self):
        super().reset()
        self.current_block = None
        self.block_edge_index = None
        self.block_edge_weight = None
        return self

    def attack(
        self,
        num_budgets: Union[int, float] = 0.05,
        *,
        block_size: int = 250_000,
        epochs: int = 125,
        epochs_resampling: int = 100,
        loss: Optional[Union[str, LOSS_TYPE]] = 'prob_margin',
        metric: Optional[Union[str, LOSS_TYPE]] = None,
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
        self.epochs = epochs

        if isinstance(loss, str):
            if loss == 'masked':
                self.loss = masked_cross_entropy
            elif loss == 'margin':
                self.loss = margin_loss
            elif loss == 'prob_margin':
                self.loss = probability_margin_loss
            elif loss == 'tanh_margin':
                self.loss = tanh_margin_loss
            else:
                raise ValueError(f'Unknown loss `{loss}`')
        else:
            self.loss = loss

        if metric is None:
            self.metric = self.loss
        else:
            self.metric = metric

        self.epochs_resampling = epochs_resampling
        self.lr = lr

        # self.coeffs.update(**kwargs) # TODO
        self.edge_weights = torch.ones(self.edge_index.size(1),
                                       device=self.device)

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        budget = self.num_budgets

        self.best_metric = float('-Inf')
        # Sample initial search space (Algorithm 1, line 3-4)
        self.sample_random_block(budget)

        # Loop over the epochs (Algorithm 1, line 5)
        for step in tqdm(range(self.num_budgets), desc='Peturbing graph...',
                         disable=disable):
            loss, gradient = self.compute_gradients(self.feat, self.label,
                                                    self.victim_nodes,
                                                    **kwargs)

            scalars = self._update(step, gradient, self.feat, self.label,
                                   budget, self.victim_nodes, **kwargs)

            scalars['loss'] = loss.item()
            self._append_statistics(scalars)

        self._close(self.feat, self.label, budget, self.victim_nodes, **kwargs)

        return self

    @torch.no_grad()
    def _update(self, epoch: int, gradient: Tensor, x: Tensor, labels: Tensor,
                budget: int, idx_attack: Optional[Tensor] = None,
                **kwargs) -> Dict[str, float]:
        """Update edge weights given gradient."""
        # Gradient update step (Algorithm 1, line 7)
        self.update_edge_weights(budget, epoch, gradient)
        # For monitoring
        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        # Projection to stay within relaxed `L_0` budget
        # (Algorithm 1, line 8)
        self.block_edge_weight = self.project(budget, self.block_edge_weight,
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
                                          budget).indices] = 1
        edge_index, edge_weight = self.get_modified_graph(
            self.edge_index, self.edge_weights, self.block_edge_index,
            topk_block_edge_weight)
        prediction = self.surrogate(x, edge_index, edge_weight, **kwargs)
        metric = self.metric(prediction, labels, idx_attack)

        # Save best epoch for early stopping
        # (not explicitly covered by pseudo code)
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_block = self.current_block.cpu().clone()
            self.best_edge_index = self.block_edge_index.cpu().clone()
            self.best_pert_edge_weight = self.block_edge_weight.cpu().detach()

        # Resampling of search space (Algorithm 1, line 9-14)
        if epoch < self.epochs_resampling - 1:
            self.resample_random_block(budget)
        elif epoch == self.epochs_resampling - 1:
            # Retrieve best epoch if early stopping is active
            # (not explicitly covered by pseudo code)
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            block_edge_weight = self.best_pert_edge_weight.clone()
            self.block_edge_weight = block_edge_weight.to(self.device)

        scalars['metric'] = metric.item()
        return scalars

    @torch.no_grad()
    def _close(self, x: Tensor, labels: Tensor, budget: int,
               idx_attack: Optional[Tensor] = None,
               **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs['with_early_stopping']:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        flipped_edges, edge_weight = self.sample_final_edges(
            x, labels, budget, idx_attack=idx_attack, **kwargs)

        assert flipped_edges.size(1) <= self.num_budgets, (
            f'# perturbed edges {flipped_edges.size(1)} '
            f'exceeds budget {self.num_budgets}')

        row, col = flipped_edges.tolist()
        for it, (u, v, w) in enumerate(zip(row, col, edge_weight.tolist())):
            if w > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)

    def compute_gradients(self, x: Tensor, labels: Tensor,
                          victim_nodes: Optional[Tensor] = None,
                          **kwargs) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad_()

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6 / Algorithm 2, line 7)
        edge_index, edge_weight = self.get_modified_graph(
            self.edge_index, self.edge_weights, self.block_edge_index,
            self.block_edge_weight)

        # Get prediction (Algorithm 1, line 6 / Algorithm 2, line 7)
        prediction = self.surrogate(x, edge_index, edge_weight, **kwargs)
        # Calculate loss combining all each node
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        loss = self.loss(prediction, labels, victim_nodes)
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
        if self.is_undirected_graph:
            block_edge_index, block_edge_weight = to_undirected(
                block_edge_index, block_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')

        modified_edge_index = torch.cat((edge_index, block_edge_index), dim=-1)
        modified_edge_weight = torch.cat((edge_weight, block_edge_weight))

        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index, modified_edge_weight,
            num_nodes=self.num_nodes, reduce='sum')

        # Allow (soft) removal of edges
        is_edge_in_clean_adj = modified_edge_weight > 1
        modified_edge_weight[is_edge_in_clean_adj] = 2 - modified_edge_weight[
            is_edge_in_clean_adj]

        return modified_edge_index, modified_edge_weight

    def _filter_self_loops_in_block(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    @torch.no_grad()
    def sample_random_block(self, budget: int = 0):
        for _ in range(self.coeffs['max_trials_sampling']):
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected_graph)
            self.current_block = torch.randint(num_possible_edges,
                                               (self.block_size, ),
                                               device=self.device)
            self.current_block = torch.unique(self.current_block, sorted=True)

            if self.is_undirected_graph:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block)

                self._filter_self_loops_in_block(with_weight=False)

            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)
            if self.current_block.size(0) >= budget:
                return

        raise RuntimeError('Sampling random block was not successful. '
                           'Please decrease `budget`.')

    def resample_random_block(self, budget: int):
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
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected_graph)
            lin_index = torch.randint(num_possible_edges, (n_edges_resample, ),
                                      device=self.device)

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True)

            if self.is_undirected_graph:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block)

            # Merge existing weights with new edge weights
            block_edge_weight_prev = self.block_edge_weight[sorted_idx]
            self.block_edge_weight = torch.full(self.current_block.shape,
                                                self.coeffs['eps'],
                                                device=self.device)

            self.block_edge_weight[
                unique_idx[:sorted_idx.size(0)]] = block_edge_weight_prev

            if not self.is_undirected_graph:
                self._filter_self_loops_in_block(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError('Sampling random block was not successful.'
                           'Please decrease `budget`.')

    @torch.no_grad()
    def sample_final_edges(self, x: Tensor, labels: Tensor, budget: int,
                           idx_attack: Optional[Tensor] = None,
                           **kwargs) -> Tuple[Tensor, Tensor]:
        best_metric = float('-Inf')
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0

        for i in range(self.coeffs['max_final_samples']):
            if i == 0:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(block_edge_weight)
                sampled_edges[torch.topk(block_edge_weight,
                                         budget).indices] = 1
            else:
                sampled_edges = torch.bernoulli(block_edge_weight).float()

            if sampled_edges.sum() > budget:
                # Allowed budget is exceeded
                continue

            self.block_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_graph(
                self.edge_index, self.edge_weights, self.block_edge_index,
                self.block_edge_weight)
            prediction = self.surrogate(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            # Save best sample
            if metric > best_metric:
                best_metric = metric
                best_edge_weight = self.block_edge_weight.clone().cpu()

        flipped_edges = self.block_edge_index[:,
                                              torch.where(best_edge_weight)[0]]
        return flipped_edges, best_edge_weight

    def update_edge_weights(self, budget: int, epoch: int, gradient: Tensor):
        # The learning rate is refined heuristically, s.t. (1) it is
        # independent of the number of perturbations (assuming an undirected
        # adjacency matrix) and (2) to decay learning rate during fine-tuning
        # (i.e. fixed search space).
        lr = (budget / self.num_nodes * self.lr /
              np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        self.block_edge_weight.data.add_(lr * gradient)

    @staticmethod
    def project(budget: int, values: Tensor, eps: float = 1e-7) -> Tensor:
        r"""Project :obj:`values`:
        :math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`."""
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack.bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def bisection(edge_weights: Tensor, a: float, b: float, n_pert: int,
                  eps=1e-5, max_iter=1e3) -> Tensor:
        """Bisection search for projection."""
        def shift(offset: float):
            return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

        miu = a
        for _ in range(int(max_iter)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (shift(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (shift(miu) * shift(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= eps):
                break
        return miu

    @staticmethod
    def _num_possible_edges(n: int, is_undirected_graph: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected_graph:
            return n * (n - 1) // 2
        else:
            return int(n**2)  # We filter self-loops later

    @staticmethod
    def _linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to upper triangular matrix without diagonal.
        This is similar to
        https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
        with number nodes decremented and col index incremented by one."""
        nn = n * (n - 1)
        row_idx = n - 2 - torch.floor(
            torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5).long()
        col_idx = 1 + lin_idx + row_idx - nn // 2 + torch.div(
            (n - row_idx) * (n - row_idx - 1), 2, rounding_mode='floor')
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def _linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to dense matrix including diagonal."""
        row_idx = torch.div(lin_idx, n, rounding_mode='floor')
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    def _append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)
