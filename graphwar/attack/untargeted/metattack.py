import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.nn import init
from tqdm import tqdm

from graphwar.surrogate import Surrogate
from graphwar.utils import singleton_mask
from graphwar.functional import to_dense_adj
from graphwar.nn.layers.gcn_conv import dense_gcn_norm
from graphwar.attack.untargeted.untargeted_attacker import UntargetedAttacker


class Metattack(UntargetedAttacker, Surrogate):
    r"""Implementation of `Metattack` attack from the: 
    `"Adversarial Attacks on Graph Neural Networks 
    via Meta Learning" 
    <https://arxiv.org/abs/1902.08412>`_ paper (ICLR'19)

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

    >>> from graphwar.attack.untargeted import Metattack
    >>> attacker = Metattack(data)
    >>> attacker.setup_surrogate(surrogate_model)
    >>> attacker.reset()
    >>> attacker.attack(0.05) # attack with 0.05% of edge perturbations
    >>> attacker.data() # get attacked graph

    >>> attacker.edge_flips() # get edge flips after attack

    >>> attacker.added_edges() # get added edges after attack

    >>> attacker.removed_edges() # get removed edges after attack

    Note
    ----
    * Please remember to call :meth:`reset` before each attack.     
    """

    # Metattack can also conduct feature attack
    _allow_feature_attack: bool = True

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        labeled_nodes: Tensor,
                        unlabeled_nodes: Tensor,
                        lr: float = 0.1,
                        epochs: int = 100,
                        momentum: float = 0.9,
                        lambda_: float = 0.,
                        *,
                        eps: float = 1.0):

        if lambda_ not in (0., 0.5, 1.):
            raise ValueError(
                "Invalid argument `lambda_`, allowed values [0: (meta-self), 1: (meta-train), 0.5: (meta-both)]."
            )

        Surrogate.setup_surrogate(self, surrogate=surrogate, eps=eps)

        labeled_nodes = torch.LongTensor(labeled_nodes).to(self.device)
        unlabeled_nodes = torch.LongTensor(unlabeled_nodes).to(self.device)

        self.labeled_nodes = labeled_nodes
        self.unlabeled_nodes = unlabeled_nodes

        self.y_train = self.label[labeled_nodes]
        self.y_self_train = self.estimate_self_training_labels(unlabeled_nodes)
        self.adj = to_dense_adj(self.edge_index,
                                self.edge_weight,
                                num_nodes=self.num_nodes).to(self.device)

        weights = []
        w_velocities = []

        for para in self.surrogate.parameters():
            if para.ndim == 2:
                para = para.t()
                weights.append(torch.zeros_like(para, requires_grad=True))
                w_velocities.append(torch.zeros_like(para))
            else:
                # we do not consider bias terms for simplicity
                pass

        self.weights, self.w_velocities = weights, w_velocities

        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_

    def reset(self):
        super().reset()
        self.adj_changes = torch.zeros_like(self.adj)
        self.feat_changes = torch.zeros_like(self.feat)
        return self

    def get_perturbed_adj(self, adj_changes=None):
        adj_changes = self.adj_changes if adj_changes is None else adj_changes
        adj_changes_triu = torch.triu(adj_changes, diagonal=1)
        adj_changes_symm = self.clip(adj_changes_triu + adj_changes_triu.t())
        modified_adj = adj_changes_symm + self.adj
        return modified_adj

    def get_perturbed_feat(self, feat_changes=None):
        feat_changes = self.feat_changes if feat_changes is None else feat_changes
        return self.feat + self.clip(feat_changes)

    def clip(self, matrix):
        clipped_matrix = torch.clamp(matrix, -1., 1.)
        return clipped_matrix

    def reset_parameters(self):
        for w, wv in zip(self.weights, self.w_velocities):
            init.xavier_uniform_(w)
            init.zeros_(wv)

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].detach().requires_grad_()
            self.w_velocities[i] = self.w_velocities[i].detach()

    def forward(self, adj, x):
        h = x
        for w in self.weights[:-1]:
            h = adj @ (h @ w)
            h = h.relu()

        return adj @ (h @ self.weights[-1])

    def inner_train(self, adj, feat):
        self.reset_parameters()

        for _ in range(self.epochs):
            out = self(adj, feat)
            loss = F.cross_entropy(out[self.labeled_nodes], self.y_train)
            grads = torch.autograd.grad(loss,
                                        self.weights,
                                        create_graph=True)

            self.w_velocities = [
                self.momentum * v + g
                for v, g in zip(self.w_velocities, grads)
            ]

            self.weights = [
                w - self.lr * v for w, v in zip(self.weights, self.w_velocities)
            ]

    def attack(self,
               num_budgets=0.05, *,
               structure_attack=True,
               feature_attack=False,
               disable=False):

        super().attack(num_budgets=num_budgets,
                       structure_attack=structure_attack,
                       feature_attack=feature_attack)

        if feature_attack:
            self._check_feature_matrix_binary()

        adj_changes = self.adj_changes
        feat_changes = self.feat_changes
        modified_adj = self.adj
        modified_feat = self.feat

        adj_changes.requires_grad_(bool(structure_attack))
        feat_changes.requires_grad_(bool(feature_attack))

        num_nodes, num_feats = self.num_nodes, self.num_feats

        for it in tqdm(range(self.num_budgets),
                       desc='Peturbing graph...',
                       disable=disable):

            if structure_attack:
                modified_adj = self.get_perturbed_adj(adj_changes)

            if feature_attack:
                modified_feat = self.get_perturbed_feat(feat_changes)

            adj_norm = dense_gcn_norm(modified_adj)
            self.inner_train(adj_norm, modified_feat)

            adj_grad, feat_grad = self.compute_gradients(adj_norm,
                                                         modified_feat)

            adj_grad_score = modified_adj.new_zeros(1)
            feat_grad_score = modified_feat.new_zeros(1)

            with torch.no_grad():
                if structure_attack:
                    adj_grad_score = self.structure_score(
                        modified_adj, adj_grad)

                if feature_attack:
                    feat_grad_score = self.feature_score(
                        modified_feat, feat_grad)

                adj_max, adj_argmax = torch.max(adj_grad_score, dim=0)
                feat_max, feat_argmax = torch.max(feat_grad_score, dim=0)

                if adj_max >= feat_max:
                    u, v = divmod(adj_argmax.item(), num_nodes)
                    edge_weight = modified_adj[u, v].data.item()
                    adj_changes[u, v].data.fill_(1 - 2 * edge_weight)
                    adj_changes[v, u].data.fill_(1 - 2 * edge_weight)

                    if edge_weight > 0:
                        self.remove_edge(u, v, it)
                    else:
                        self.add_edge(u, v, it)
                else:
                    u, v = divmod(feat_argmax.item(), num_feats)
                    feat_weight = modified_feat[u, v].data.item()
                    feat_changes[u, v].data.fill_(1 - 2 * feat_weight)
                    if feat_weight > 0:
                        self.remove_feat(u, v, it)
                    else:
                        self.add_feat(u, v, it)

        return self

    def structure_score(self, modified_adj, adj_grad):
        score = adj_grad * (1 - 2 * modified_adj)
        score -= score.min()
        score = torch.triu(score, diagonal=1)
        if not self._allow_singleton:
            # Set entries to 0 that could lead to singleton nodes.
            score *= singleton_mask(modified_adj)
        return score.view(-1)

    def feature_score(self, modified_feat, feat_grad):
        score = feat_grad * (1 - 2 * modified_feat)
        score -= score.min()
        return score.view(-1)

    def compute_gradients(self, modified_adj, modified_feat):

        logit = self(modified_adj, modified_feat) / self.eps

        if self.lambda_ == 1:
            loss = F.cross_entropy(logit[self.labeled_nodes], self.y_train)
        elif self.lambda_ == 0.:
            loss = F.cross_entropy(
                logit[self.unlabeled_nodes], self.y_self_train)
        else:
            loss_labeled = F.cross_entropy(
                logit[self.labeled_nodes], self.y_train)
            loss_unlabeled = F.cross_entropy(
                logit[self.unlabeled_nodes], self.y_self_train)
            loss = self.lambda_ * loss_labeled + \
                (1 - self.lambda_) * loss_unlabeled

        if self.structure_attack and self.feature_attack:
            return grad(loss, [self.adj_changes, self.feat_changes])

        if self.structure_attack:
            return grad(loss, self.adj_changes)[0], None

        if self.feature_attack:
            return None, grad(loss, self.feat_changes)[0]
