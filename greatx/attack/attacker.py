import abc
from numbers import Number
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_scipy_sparse_matrix
from torch_sparse import SparseTensor

from greatx.functional import to_dense_adj


class Attacker(torch.nn.Module):
    """Adversarial attacker for graph data.
    Note that this is an abstract class.

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

    Examples
    --------
    For example, the attacker model should be defined as follows:

    .. code-block:: python

        from greatx.attacker import Attacker
        attacker = Attacker(data, device='cuda')
        attacker.reset() # reset states
        attacker.attack(attack_arguments) # attack
        attacker.data() # get the attacked graph denoted as PyG-like Data

    """
    _max_perturbations: Union[float, int] = 0
    _allow_feature_attack: bool = False
    _allow_structure_attack: bool = True
    _allow_singleton: bool = True

    def __init__(self, data: Data, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None,
                 **kwargs):
        """Initialization of an attacker model.
        """

        super().__init__()

        if kwargs:
            raise TypeError("Got an unexpected keyword argument "
                            f"'{next(iter(kwargs.keys()))}'.")

        assert isinstance(data, Data)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_weight is None

        self.device = torch.device(device)
        self.ori_data = data.to(self.device)

        self.adjacency_matrix: sp.csr_matrix = to_scipy_sparse_matrix(
            data.edge_index, num_nodes=data.num_nodes).tocsr()
        self.name = name or self.__class__.__name__
        self.seed = seed

        self._degree = degree(data.edge_index[0], num_nodes=data.num_nodes,
                              dtype=torch.float)

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_feats = data.x.size(1)
        self.nodes_set = set(range(self.num_nodes))

        if seed is not None:
            seed_everything(seed)

        self._is_reset = False

    def reset(self):
        """Reset attacker state.
        Override this method in subclass to implement specific function."""
        self._is_reset = True
        return self

    @abc.abstractmethod
    def data(self) -> Data:
        """Get the attacked graph denoted as PyG-like Data.

        Raises
        ------
        NotImplementedError
            The subclass does not implement this interface.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def attack(self) -> "Attacker":
        """Abstract method.
        The subclass must override this method to implement specific
        attack for itself.

        Raises
        ------
        NotImplementedError
            The subclass does not implement this interface.
        """
        raise NotImplementedError

    def _check_budget(self, num_budgets: Union[float, int],
                      max_perturbations: Union[float, int]) -> int:
        """Check and return attack budget."""

        max_perturbations = max(max_perturbations, self.max_perturbations)

        if not isinstance(num_budgets, Number) or num_budgets <= 0:
            raise ValueError("'num_budgets' must be a positive scalar. "
                             f"but got '{num_budgets}'.")

        if num_budgets > max_perturbations:
            raise ValueError(
                "'num_budgets' should be less than or equal "
                f"the maximum allowed perturbations: {max_perturbations}."
                "If you want to use a larger budget, you could set "
                "'attacker.set_max_perturbations(a_larger_budget)'.")

        if num_budgets < 1. or (isinstance(num_budgets, float)
                                and num_budgets == 1.0):
            assert self._max_perturbations != np.inf
            num_budgets = max_perturbations * num_budgets

        return int(num_budgets)

    def set_max_perturbations(self, max_perturbations: Union[float,
                                                             int] = np.inf,
                              verbose: bool = True) -> "Attacker":
        """Set the maximum number of allowed perturbations

        Parameters
        ----------
        max_perturbations : Union[float, int], optional
            the maximum number of allowed perturbations, by default np.inf
        verbose : bool, optional
            whether to verbose the operation, by default True

        Example
        -------
            attacker.set_max_perturbations(10)
        """

        assert isinstance(max_perturbations, Number), max_perturbations
        self._max_perturbations = max_perturbations
        if verbose:
            print(f"Set maximum perturbations: {max_perturbations}")
        return self

    @property
    def max_perturbations(self) -> Union[float, int]:
        """float or int: Maximum allowable perturbation size."""
        return self._max_perturbations

    @property
    def feat(self) -> Tensor:
        """Node features of the original graph."""
        return self.ori_data.x

    @property
    def label(self) -> Tensor:
        """Node labels of the original graph."""
        return self.ori_data.y

    @property
    def edge_index(self) -> Tensor:
        """Edge index of the original graph."""
        return self.ori_data.edge_index

    @property
    def edge_weight(self) -> Tensor:
        """Edge weight of the original graph."""
        return self.ori_data.edge_weight

    def get_dense_adj(self) -> Tensor:
        """Returns a dense adjacency denoting the
        original graph.
        If :attr:`self.ori_data` has the attribute :obj:`adj_t`,
        then it is returned, otherwise it is built from the
        tuple :obj:`(edge_index, edge_weight)`.
        """
        data = self.ori_data
        adj_t = data.get('adj_t')
        if isinstance(adj_t, Tensor):
            return adj_t.t().to(self.device)
        elif isinstance(adj_t, SparseTensor):
            return adj_t.to_dense().to(self.device)

        return to_dense_adj(data.edge_index, data.edge_weight,
                            self.num_nodes).to(self.device)

    def _check_feature_matrix_binary(self):
        """Check if the feature matrix is binary.

        Raises
        ------
        RuntimeError
            if the feature matrix is not binary
        """
        feat = self.feat
        # FIXME: (Jintang Li) this is quite time-consuming in large matrix
        # so it only checks `10` rows of the matrix randomly.
        feat = feat[torch.randint(0, feat.size(0), size=(10, ))]
        if not torch.unique(feat).tolist() == [0, 1]:
            raise RuntimeError(
                "Node feature matrix is required to be a 0-1 binary matrix.")

    def extra_repr(self) -> str:
        return f"device={self.device}, seed={self.seed},"
