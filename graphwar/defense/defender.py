import abc
import dgl
import torch
import scipy.sparse as sp
from typing import Optional

from graphwar import set_seed, Config

_FEATURE = Config.feat
_LABEL = Config.label


class Defender(torch.nn.Module):
    """Adversarial defender for graph data.
    For example, the defender model should be defined as follows:

    >>> defender = Defender(graph, device='cuda')
    >>> defender.fit(defense_arguments)

    """

    def __init__(self, graph: dgl.DGLGraph, device: str = "cpu",
                 seed: Optional[int] = None, name: Optional[str] = None, **kwargs):
        f"""Initialization of an defender model.

        Parameters
        ----------
        graph : dgl.DGLGraph
            the DGL graph. If the defense requires node features,
            `graph.ndata[{_FEATURE}]` should be specified. 
            If the defense requires node labels, 
            `graph.ndata[{_LABEL}]` should be specified
        device : str, optional
            the device of the defense running on, by default "cpu"
        seed : Optional[int], optional
            the random seed of reproduce the defense, by default None
        name : Optional[str], optional
            name of the defender, if None, it would be `__class__.__name__`, 
            by default None
        kwargs : optional
            additional arguments of :class:`graphwar.defense.Defender`,
            including (`{_FEATURE}`, `{_LABEL}`) to specify the node features 
            and the node labels, if they are not in `graph.ndata`


        Note
        ----
        * If the defense requires node features,
        `graph.ndata[{_FEATURE}]` should be specified. 

        * If the defense requires node labels, 
        `graph.ndata[{_LABEL}]` should be specified.
        """
        super().__init__()
        feat = kwargs.pop(_FEATURE, None)
        label = kwargs.pop(_LABEL, None)

        if kwargs:
            raise TypeError(
                f"Got an unexpected keyword argument '{next(iter(kwargs.keys()))}' "
                f"expected ({_FEATURE}, {_LABEL})."
            )

        self.device = torch.device(device)
        self._graph = graph.to(self.device)

        if feat is not None:
            feat = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
            assert feat.size(0) == graph.num_nodes()
        else:
            feat = self.graph.ndata.get(_FEATURE, None)

        if label is not None:
            label = torch.as_tensor(label, dtype=torch.long, device=self.device)
        else:
            label = self.graph.ndata.get(_LABEL, None)

        setattr(self, '_' + _FEATURE, feat)
        setattr(self, '_' + _LABEL, label)

        self.adjacency_matrix: sp.csr_matrix = graph.adjacency_matrix(scipy_fmt='csr')
        self.name = name or self.__class__.__name__
        self.seed = seed

        self._degree = self._graph.in_degrees()

        self.edges = self._graph.edges()
        self.nodes = self._graph.nodes()
        self.num_nodes = self._graph.num_nodes()
        self.num_edges = self._graph.num_edges() // 2
        self.num_feats = feat.size(-1) if feat is not None else None

        set_seed(seed)

        self.is_reseted = False

    def reset(self):
        self.is_reseted = True
        return self

    def g(self):
        raise NotImplementedError

    @property
    def feat(self):
        return getattr(self, '_' + _FEATURE, None)

    @property
    def label(self):
        return getattr(self, '_' + _LABEL, None)

    @property
    def graph(self):
        return self._graph

    @abc.abstractmethod
    def fit(self) -> "Defender":
        """defined for defender model."""
        raise NotImplementedError

    def _check_feature_matrix_exists(self):
        if self.feat is None:
            raise RuntimeError("Node feature matrix does not exist"
                               f", please add node feature data externally via `g.ndata['{_FEATURE}'] = {_FEATURE}` "
                               f"or initialize via `attacker = {self.__class__.__name__}(g, {_FEATURE}={_FEATURE})`.")

    def _check_node_label_exists(self):
        if self.label is None:
            raise RuntimeError("Node labels does not exist"
                               f", please add node labels externally via `g.ndata['{_LABEL}'] = {_LABEL}` "
                               f"or initialize via `attacker = {self.__class__.__name__}(g, {_LABEL}={_LABEL})`.")

    def _check_feature_matrix_binary(self):
        self._check_feature_matrix_exists()
        feat = self.feat
        # FIXME: (Jintang Li) this is quite time-consuming in large matrix
        # so we only check `10` rows of the matrix randomly.
        feat = feat[torch.randint(0, feat.size(0), size=(10,))]
        if not torch.unique(feat).tolist() == [0, 1]:
            raise RuntimeError("Node feature matrix is required to be a 0-1 binary matrix.")

    def extra_repr(self) -> str:
        return f"device={self.device}, seed={self.seed},"
