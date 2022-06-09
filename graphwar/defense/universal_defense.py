from typing import Union
from copy import copy
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch

from graphwar import Surrogate
from graphwar.nn.models import SGC, GCN
from graphwar.utils import remove_edges


class UniversalDefense(torch.nn.Module):
    """Base class for graph universal defense"""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self._anchors = None

    def forward(self, data: Data, target_nodes: Union[int, Tensor],
                k: int = 50, symmetric: bool = True) -> Data:
        """Return the defended graph with defensive perturbation performed on.

        Parameters
        ----------
        data : a graph represented as PyG-like data instance 
            the graph where the defensive perturbation performed on
        target_nodes : Union[int, Tensor]
            the target nodes where the defensive perturbation performed on            
        k : int
            the number of anchor nodes in the defensive perturbation, by default 50
        symmetric : bool
            Determine whether the resulting graph is forcibly symmetric,
            by default True

        Returns
        -------
        Data: PyG-like data
            the defended graph with defensive perturbation performed on the target nodes
        """
        data = copy(data)
        data.edge_index = remove_edges(data.edge_index,
                                       self.removed_edges(target_nodes, k),
                                       symmetric=symmetric)
        return data

    def removed_edges(self, target_nodes: Union[int, Tensor], k: int = 50) -> Tensor:
        """Return edges to remove with the defensive perturbation performed on 
        on the target nodes

        Parameters
        ----------
        target_nodes : Union[int, Tensor]
            the target nodes where the defensive perturbation performed on
        k : int
            the number of anchor nodes in the defensive perturbation, by default 50

        Returns
        -------
        Tensor, shape [2, k]
            the edges to remove with the defensive perturbation performed on 
            on the target nodes
        """
        row = torch.as_tensor(target_nodes, device=self.device).view(-1)
        col = self.anchors(k)
        row, col = row.repeat_interleave(k), col.repeat(row.size(0))

        return torch.stack([row, col], dim=0)

    def anchors(self, k: int = 50) -> Tensor:
        """Return the top-k anchor nodes

        Parameters
        ----------
        k : int, optional
            the number of anchor nodes in the defensive perturbation, by default 50

        Returns
        -------
        Tensor
            the top-k anchor nodes
        """
        assert k > 0
        return self._anchors[:k]

    def patch(self, k=50) -> Tensor:
        """Return the universal patch of the defensive perturbation

        Parameters
        ----------
        k : int, optional
            the number of anchor nodes in the defensive perturbation, by default 50

        Returns
        -------
        Tensor
            the 0-1 (boolean) universal patch where 1 denotes the edges to be removed.
        """
        _patch = torch.zeros(
            self.num_nodes, dtype=torch.bool, device=self.device)
        _patch[self.anchors(k=k)] = True
        return _patch


class GUARD(UniversalDefense, Surrogate):
    """Graph Universal Adversarial Defense (GUARD)

    Parameters
    ----------
    data : Data
        the PyG-like input data
    alpha : float, optional
        the scale factor for node degree, by default 2
    batch_size : int, optional
        the batch size for computing node influence, by default 512        
    device : str, optional
        the device where the method running on, by default "cpu"        

    Example
    -------
    >>> surrogate = GCN(dataset.num_features, dataset.num_classes, bias=False, acts=None)
    >>> surrogate_trainer = Trainer(surrogate, device=device)
    >>> ckp = ModelCheckpoint('guard.pth', monitor='val_acc')
    >>> trainer.fit({'data': data, 'mask': splits.train_nodes}, 
                {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
    >>> trainer.evaluate({'data': data, 'mask': splits.test_nodes})

    >>> guard = GUARD(data, device=device)
    >>> guard.setup_surrogate(surrogate, data.y[splits.train_nodes])
    >>> target_node = 1
    >>> perturbed_data = ... # Other PyG-like Data
    >>> guard(perturbed_data, target_node, k=50)
    """

    def __init__(self, data: Data, alpha: float = 2, batch_size: int = 512, device: str = "cpu"):
        super().__init__(device=device)
        self.data = data
        self.alpha = alpha
        self.batch_size = batch_size
        self.deg = degree(data.edge_index[0],
                          num_nodes=data.num_nodes, dtype=torch.float)

    @torch.no_grad()
    def setup_surrogate(self, surrogate: torch.nn.Module,
                        victim_labels: Tensor) -> "GUARD":

        Surrogate.setup_surrogate(self, surrogate=surrogate,
                                  freeze=True, required=(SGC, GCN))
        W = None
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                continue
            if W is None:
                W = para.detach()
            else:
                W = W @ para.detach()

        W = self.data.x.to(self.device) @ W
        d = self.deg.clamp(min=1).to(self.device)

        loader = DataLoader(victim_labels, pin_memory=False,
                            batch_size=self.batch_size, shuffle=False)

        w_max = W.max(1).values
        I = 0.
        for y in loader:
            I += W[:, y].sum(1)
        I = (w_max - I / victim_labels.size(0)) / \
            d.pow(self.alpha)  # node importance
        self._anchors = torch.argsort(I, descending=True)
        return self


class DegreeGUARD(UniversalDefense):
    """Graph Universal Defense based on node degrees

    Parameters
    ----------
    data : Data
        the PyG-like input data
    descending : bool, optional
        whether the degree of chosen nodes are in descending order, by default False
    device : str, optional
        the device where the method running on, by default "cpu"        

    Example
    -------
    >>> data = ... # PyG-like Data
    >>> guard = DegreeGUARD(data))
    >>> target_node = 1
    >>> perturbed_data = ... # Other PyG-like Data
    >>> guard(perturbed_data, target_node, k=50)
    """

    def __init__(self, data: Data, descending: bool = False, device: str = "cpu"):
        super().__init__(device=device)
        deg = degree(data.edge_index[0],
                     num_nodes=data.num_nodes, dtype=torch.float)
        self._anchors = torch.argsort(deg, descending=descending)


class RandomGUARD(UniversalDefense):
    """Graph Universal Defense based on random choice

    Parameters
    ----------
    data : Data
        the PyG-like input data
    device : str, optional
        the device where the method running on, by default "cpu"    

    Example
    -------
    >>> data = ... # PyG-like Data
    >>> guard = RandomGUARD(data)
    >>> target_node = 1
    >>> perturbed_data = ... # Other PyG-like Data
    >>> guard(perturbed_data, target_node, k=50)
    """

    def __init__(self, data: Data, device: str = "cpu"):
        super().__init__(device=device)
        self._anchors = torch.randperm(data.num_nodes, device=self.device)
