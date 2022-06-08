from typing import Callable, Optional, List

import torch
import numpy as np
import scipy.sparse as sp
import os.path as osp
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import LabelEncoder


def load_npz(file_name: str) -> Data:
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = loader['adj_matrix'].item()
        adj_matrix = adj_matrix.maximum(adj_matrix.T)
        attr_matrix = loader['node_attr']

        if attr_matrix.dtype.kind == 'O':
            # scipy sparse matrix
            attr_matrix = attr_matrix.item().A

        labels = loader['node_label']
        if labels.shape[0] != adj_matrix.shape[0]:
            _labels = np.full((adj_matrix.shape[0] - labels.shape[0],), -1)
            labels = np.hstack([labels, _labels])

        if np.unique(labels).shape[0] != labels.max() + 1:
            labels = LabelEncoder().fit_transform(labels)

        x = torch.from_numpy(attr_matrix).to(torch.float)
        adj_matrix = adj_matrix.tocoo()
        row = torch.from_numpy(adj_matrix.row).to(torch.long)
        col = torch.from_numpy(adj_matrix.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))

        y = torch.from_numpy(labels).to(torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


_DATASETS = {
    'citeseer', 'citeseer_full', 'cora', 'cora_ml', 'cora_full', 'amazon_cs',
    'amazon_photo', 'coauthor_cs', 'coauthor_phy', 'polblogs', 'karate_club',
    'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai', 'pdn',
}


class GraphWarDataset(InMemoryDataset):
    r"""A series of datasets used in GraphWar. These datasets are
    stored in :obj:`.npz` format, consisting of a single graph.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    name : str
        The name of the dataset. See :meth:`available_datasets`
        for all available datasets.
    transform : Optional[Callable], optional
        A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access, 
        by default None
    pre_transform : Optional[Callable], optional
        A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk, by default None    

    Example
    -------
    >>> from graphwar.dataset import GraphWarDataset
    >>> import torch_geometric.transforms as T

    >>> GraphWarDataset.available_datasets() # see all available datasets.
    ['cora', 'citeseer', 'pubmed', ...]

    >>> dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())
    >>> data = dataset[0] # there is only one graph

    Note
    ----
    We follow the setting in :obj:`Nettack` from the: `"Adversarial Attacks on Neural Networks 
    for Graph Data" <https://arxiv.org/abs/1805.07984>`_ paper, 
    which considers the largest connected component for each graph.

    For more details of these datasets, see https://github.com/EdisonLeeeee/GraphData
    """

    url = 'https://github.com/EdisonLeeeee/GraphData/raw/master/datasets/{}.npz'

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in _DATASETS
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, f"GraphWar-{self.name}", 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"GraphWar-{self.name}", 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        data = load_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def available_datasets() -> List[str]:
        """Return all available datasets.
        """
        return list(_DATASETS)

    def __repr__(self) -> str:
        return f'GraphWar-{self.name.capitalize()}'
