import dgl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from graphwar import Surrogater
from graphwar.models import SGC, GCN
from graphwar.utils import remove_edges
from typing import Union


class UniversalDefense(torch.nn.Module):

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self._anchors = None

    def forward(self, g: dgl.DGLGraph, target_nodes: Union[int, Tensor],
                k: int = 50, symmetric: bool = True) -> dgl.DGLGraph:
        """return the defended graph with defensive perturbation on the clean graph

        Parameters
        ----------
        g : dgl.DGLGraph
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
        dgl.DGLGraph
            the defended graph with defensive perturbation performed on the target nodes
        """
        edges = self.removed_edges(target_nodes, k)
        return remove_edges(g, edges, symmetric=symmetric)

    def removed_edges(self, target_nodes: Union[int, Tensor], k: int = 50) -> Tensor:
        """return edges to remove with the defensive perturbation performed on 
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
        """return the top-k anchor nodes

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
        """return the universal patch of the defensive perturbation

        Parameters
        ----------
        k : int, optional
            the number of anchor nodes in the defensive perturbation, by default 50

        Returns
        -------
        Tensor
            the 0-1 (boolean) universal patch where 1 denots the edges to be removed.
        """
        _patch = torch.zeros(
            self.num_nodes, dtype=torch.bool, device=self.device)
        _patch[self.anchors(k=k)] = True
        return _patch


class GUARD(UniversalDefense, Surrogater):
    """Graph Universal Adversarial Defense (GUARD)

    Example
    -------
    >>> g = ... # DGLGraph
    >>> splits = ... # node splits
    >>> surrogate = GCN(num_feats, num_classes, bias=False, acts=None)
    >>> surrogate_trainer = Trainer(surrogate, device=device)
    >>> surrogate_trainer.fit(g, y_train, splits.train_nodes)
    >>> surrogate_trainer.evaluate(g, y_test, splits.test_nodes)    
    >>> guard = GUARD(g.ndata['feat'], g.in_degrees())
    >>> target_node = 1
    >>> guard(target_node, g, k=50)
    """

    def __init__(self, feat: Tensor, degree: Tensor, alpha: float = 2,
                 batch_size: int = 512, device: str = "cpu"):
        super().__init__(device=device)
        self.feat = feat.to(self.device)
        self.degree = degree.to(self.feat)
        self.alpha = alpha
        self.batch_size = batch_size

    def setup_surrogate(self, surrogate: torch.nn.Module,
                        victim_labels: Tensor) -> "GUARD":

        Surrogater.setup_surrogate(self, surrogate=surrogate,
                                   freeze=True, required=(SGC, GCN))
        W = None
        for para in self.surrogate.parameters():
            if para.ndim == 1:
                continue
            if W is None:
                W = para.detach()
            else:
                W = W @ para.detach()

        W = self.feat @ W
        d = self.degree.clamp(min=1)

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

    Example
    -------
    >>> g = ...
    >>> guard = DegreeGUARD(g.in_degrees())
    >>> target_node = 1
    >>> guard(target_node, g, k=50)
    """

    def __init__(self, degree: Tensor, descending=False, device: str = "cpu"):
        super().__init__(device=device)
        self._anchors = torch.argsort(
            degree.to(self.device), descending=descending)


class RandomGUARD(UniversalDefense):
    """Graph Universal Defense based on random choice

    Example
    -------
    >>> g = ...
    >>> guard = RandomGUARD(g.num_nodes())
    >>> target_node = 1
    >>> guard(target_node, g, k=50)
    """

    def __init__(self, num_nodes: int, device: str = "cpu"):
        super().__init__(device=device)
        self.num_nodes = num_nodes
        self._anchors = torch.randperm(self.num_nodes, device=self.device)


if __name__ == '__main__':
    import torch
    from graphwar.data import GraphWarDataset
    from graphwar.training import Trainer
    from graphwar.training.callbacks import ModelCheckpoint
    from graphwar.models import GCN, SGC
    from graphwar.utils import split_nodes

    data = GraphWarDataset('cora', verbose=True, standardize=True)
    g = data[0]
    y = g.ndata['label']
    splits = split_nodes(y, random_state=15)

    num_feats = g.ndata['feat'].size(1)
    num_classes = data.num_classes
    y_train = y[splits.train_nodes]
    y_val = y[splits.val_nodes]
    y_test = y[splits.test_nodes]

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    g = g.to(device)

    defense = 'GUARD'

    if defense == "GUARD":
        surrogate = GCN(num_feats, num_classes, bias=False, acts=None)
        surrogate_trainer = Trainer(surrogate, device=device)
        cb = ModelCheckpoint('guard.pth', monitor='val_accuracy')
        surrogate_trainer.fit(g, y_train, splits.train_nodes, val_y=y_val,
                              val_index=splits.val_nodes, callbacks=[cb], verbose=0)
        guard = GUARD(g.ndata['feat'], g.in_degrees(), device=device)
        guard.setup_surrogate(surrogate, y_train)
    elif defense == "RandomGUARD":
        guard = RandomGUARD(g.num_nodes(), device=device)
    elif defense == "DegreeGUARD":
        guard = DegreeGUARD(g.in_degrees(), device=device)
    else:
        raise ValueError(f"Unknown defense {defense}")

    # get a defensed graph
    defense_g = guard(g, target_nodes=1, k=50)

    # get anchors nodes (potential attacker nodes)
    anchors = guard.anchors(k=50)
