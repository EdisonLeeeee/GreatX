import math
from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm

from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker
from graphwar.surrogate import Surrogate
from graphwar.functional import to_dense_adj


class PGDAttack(UntargetedAttacker, Surrogate):
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
    kwargs : additional arguments of :class:`graphwar.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`       

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.untargeted import PGDAttack
    >>> attacker = PGDAttack(data)
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker.reset()
    >>> attacker.attack(0.05) # attack with 0.05% of edge perturbations
    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack       

    Note
    ----
    * MinMax attack is a variant of :class:`graphwar.attack.untargeted.PGDAttack` attack.   
    * Please remember to call :meth:`reset` before each attack.      

    """

    # PGDAttack cannot ensure that there is not singleton node after attacks.
    _allow_singleton: bool = True

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor,
                        unlabeled_nodes: Optional[Tensor] = None,
                        *,
                        eps: float = 1.0,
                        freeze: bool = True):

        Surrogate.setup_surrogate(self, surrogate=surrogate,
                                  eps=eps, freeze=freeze)

        labeled_nodes = torch.LongTensor(labeled_nodes).to(self.device)
        # poisoning attack in DeepRobust
        if unlabeled_nodes is None:
            victim_nodes = labeled_nodes
            victim_labels = self.label[labeled_nodes]
        else:  # Evasion attack in original paper
            unlabeled_nodes = torch.LongTensor(unlabeled_nodes).to(self.device)
            self_training_labels = self.estimate_self_training_labels(
                unlabeled_nodes)
            victim_nodes = torch.cat([labeled_nodes, unlabeled_nodes], dim=0)
            victim_labels = torch.cat([self.label[labeled_nodes],
                                       self_training_labels], dim=0)

        adj = to_dense_adj(self.edge_index,
                           self.edge_weight,
                           num_nodes=self.num_nodes).to(self.device)
        I = torch.eye(self.num_nodes, device=self.device)
        self.complementary = torch.ones_like(adj) - I - 2. * adj
        self.adj = adj
        self.victim_nodes = victim_nodes
        self.victim_labels = victim_labels

        return self

    def reset(self):
        super().reset()
        self.perturbations = torch.zeros_like(self.adj).requires_grad_()
        return self

    def attack(self,
               num_budgets=0.05, *,
               C=None,
               CW_loss=False,
               epochs=200,
               sample_epochs=20,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        self.CW_loss = CW_loss
        C = self.config_C(C)
        perturbations = self.perturbations
        for epoch in tqdm(range(epochs),
                          desc='PGD training...',
                          disable=disable):
            gradients = self.compute_gradients(perturbations,
                                               self.victim_nodes,
                                               self.victim_labels)
            lr = C / math.sqrt(epoch + 1)
            perturbations.data.add_(lr * gradients)
            perturbations = self.projection(perturbations)

        best_s = self.bernoulli_sample(
            perturbations, sample_epochs, disable=disable)
        row, col = torch.where(best_s > 0.)
        for it, (u, v) in enumerate(zip(row.tolist(), col.tolist())):
            if self.adj[u, v] > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)

        return self

    def config_C(self, C=None):
        if C is not None:
            return C
        if self.CW_loss:
            C = 0.1
        else:
            C = 200
        return C

    def bisection(self, perturbations, a, b, epsilon):
        def func(x):
            clipped_matrix = self.clip(perturbations - x)
            return clipped_matrix.sum() - self.num_budgets

        miu = a
        while (b - a) > epsilon:
            miu = (a + b) / 2
            # Check if middle point is root
            if func(miu) == 0:
                break
            # Decide the side to repeat the steps
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return miu

    def get_perturbed_adj(self, perturbations=None):
        perturbations = self.perturbations if perturbations is None else perturbations
        adj_triu = torch.triu(perturbations, diagonal=1)
        perturbations = adj_triu + adj_triu.t()
        adj = self.complementary * perturbations + self.adj
        return adj

    def projection(self, perturbations):
        clipped_matrix = self.clip(perturbations)
        num_modified = clipped_matrix.sum()

        if num_modified > self.num_budgets:
            left = (perturbations - 1.).min()
            right = perturbations.max()
            miu = self.bisection(perturbations, left, right, epsilon=1e-5)
            clipped_matrix = self.clip(perturbations - miu)
        else:
            pass

        perturbations.data.copy_(clipped_matrix)
        return perturbations

    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, 0., 1.)
        return clipped_matrix

    @torch.no_grad()
    def bernoulli_sample(self, perturbations, sample_epochs=20, disable=False):
        best_loss = -1e4
        best_s = None
        probs = torch.triu(perturbations, diagonal=1)
        sampler = Bernoulli(probs)
        for it in tqdm(range(sample_epochs),
                       desc='Bernoulli sampling...',
                       disable=disable):
            sampled = sampler.sample()
            if sampled.sum() > self.num_budgets:
                continue

            perturbations.data.copy_(sampled)
            loss = self.compute_loss(
                perturbations, self.victim_nodes, self.victim_labels)

            if best_loss < loss:
                best_loss = loss
                best_s = sampled

        assert best_s is not None, "Something went wrong"
        return best_s.cpu()

    def compute_loss(self, perturbations, victim_nodes, victim_labels):
        adj = self.get_perturbed_adj(perturbations)
        logit = self.surrogate(self.feat, adj)[victim_nodes] / self.eps

        if self.CW_loss:
            # logit = F.softmax(logit, dim=1)
            one_hot = torch.eye(
                logit.size(-1), device=self.device)[victim_labels]
            range_idx = torch.arange(victim_nodes.size(0), device=self.device)
            best_wrong_class = (logit - 1000 * one_hot).argmax(1)
            margin = logit[range_idx, victim_labels] - \
                logit[range_idx, best_wrong_class] + 50
            loss = -torch.clamp(margin, min=0.)
            return loss.mean()
        else:
            loss = F.cross_entropy(logit, self.victim_labels)
        return loss

    def compute_gradients(self, perturbations, victim_nodes, victim_labels):
        loss = self.compute_loss(perturbations, victim_nodes, victim_labels)
        return grad(loss, perturbations, create_graph=False)[0]


class MinmaxAttack(PGDAttack):
    r"""Implementation of `MinMax` attack from the: 
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
    kwargs : additional arguments of :class:`graphwar.attack.Attacker`,

    Raises
    ------
    TypeError
        unexpected keyword argument in :obj:`kwargs`       

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0]

    >>> surrogate_model = ... # train your surrogate model

    >>> from graphwar.attack.untargeted import MinmaxAttack
    >>> attacker = MinmaxAttack(data)
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker.reset()
    >>> attacker.attack(0.05) # attack with 0.05% of edge perturbations
    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack       

    Note
    ----
    * MinMax attack is a variant of :class:`graphwar.attack.untargeted.PGDAttack` attack.
    * Please remember to call :meth:`reset` before each attack.     

    """

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor,
                        unlabeled_nodes: Optional[Tensor] = None,
                        *,
                        eps: float = 1.0):

        super().setup_surrogate(surrogate=surrogate, labeled_nodes=labeled_nodes,
                                unlabeled_nodes=unlabeled_nodes, eps=eps, freeze=False)

        self.cached = deepcopy(self.surrogate.state_dict())
        return self

    def reset(self):
        super().reset()
        self.surrogate.load_state_dict(self.cached)
        return self

    def attack(self,
               num_budgets=0.05, *,
               C=None, lr=0.001,
               CW_loss=False,
               epochs=100,
               sample_epochs=20,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super(PGDAttack, self).attack(num_budgets=num_budgets,
                                      structure_attack=structure_attack,
                                      feature_attack=feature_attack)

        self.CW_loss = CW_loss
        C = self.config_C(C)
        perturbations = self.perturbations
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)

        for epoch in tqdm(range(epochs),
                          desc='Min-MAX training...',
                          disable=disable):

            # =========== Min-step ===================
            loss = self.compute_loss(perturbations,
                                     self.victim_nodes,
                                     self.victim_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ========================================

            # =========== Max-step ===================
            gradients = self.compute_gradients(perturbations,
                                               self.victim_nodes,
                                               self.victim_labels)
            lr = C / math.sqrt(epoch + 1)
            perturbations.data.add_(lr * gradients)
            perturbations = self.projection(perturbations)
            # ========================================

        best_s = self.bernoulli_sample(
            perturbations, sample_epochs, disable=disable)
        row, col = torch.where(best_s > 0.)
        for it, (u, v) in enumerate(zip(row.tolist(), col.tolist())):
            if self.adj[u, v] > 0:
                self.remove_edge(u, v, it)
            else:
                self.add_edge(u, v, it)

        return self
