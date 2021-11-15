"""Graph benchmark datasets for adversarial graph learning."""
import torch
import sys
import scipy.sparse as sp
import numpy as np
import os

from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import save_graphs, load_graphs, download
from dgl.convert import graph as dgl_graph
from dgl import transform

_DATASETS = {
    'citeseer', 'citeseer_full', 'cora', 'cora_ml', 'cora_full', 'amazon_cs',
    'amazon_photo', 'coauthor_cs', 'coauthor_phy', 'polblogs',
    'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai',
}


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A - sp.diags(A.diagonal(), format='csr')
    A.eliminate_zeros()
    return A


def largest_connected_components(A):
    _, component_indices = sp.csgraph.connected_components(A)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[-1]
    nodes_to_keep = np.where(component_indices == components_to_keep)[0]
    return nodes_to_keep


def _get_adversarial_graph_url(file_url):
    """Get online dataset url for download."""
    repo_url = 'https://github.com/EdisonLeeeee/GraphWarData/raw/master/datasets/'
    return repo_url + file_url


class GraphWarDataset(DGLBuiltinDataset):
    fr"""Base Class for adversarial graph dataset

    Reference: 
    [1] GitHub: hhttps://github.com/EdisonLeeeee/GraphWarData
    [2] Gitee: hhttps://gitee.com/EdisonLeeeee/GraphWarData
    
    
    Allowed Datasets
    ----------------
    {tuple(_DATASETS)}
    """

    def __init__(self, name, raw_dir=None, 
                 force_reload=False, verbose=False, standardize=True):
        if name not in _DATASETS:
            raise ValueError(f"Unknow dataset {name}, allowed datasets are {tuple(_DATASETS)}.")
            
        name = 'graphwar-' + name
        _url = _get_adversarial_graph_url(name + '.npz')
            
        self.standardize = standardize
        super().__init__(name=name,
                         url=_url,
                         raw_dir=raw_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def download(self):
        r"""Automatically download data"""
        download_path = os.path.join(self.raw_dir, 'GraphWarData/datasets', self.name + '.npz')
        if not os.path.exists(download_path):
            download(self.url, path=download_path)

    def process(self):
        npz_path = os.path.join(self.raw_dir, 'GraphWarData/datasets', self.name + '.npz')
        g = self._load_npz(npz_path)
        # g = transform.reorder_graph(
        #     g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
        self._graph = g
        self._data = [g]
        self._print_info()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._data = [graphs[0]]
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[-1]))
            print('  NumbClasses: {}'.format(self.num_classes))

    def _load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            adj_matrix = loader['adj_matrix'].item()
            adj_matrix = adj_matrix.maximum(adj_matrix.T)
            adj_matrix = eliminate_self_loops(adj_matrix)
            attr_matrix = loader['attr_matrix'].item().A
            labels = loader['label']
            if labels.shape[0] != adj_matrix.shape[0]:
                _labels = np.full((adj_matrix.shape[0] - labels.shape[0],), -1)
                labels = np.hstack([labels, _labels])

            if self.standardize:
                nodes_to_keep = largest_connected_components(adj_matrix)
                adj_matrix = adj_matrix[nodes_to_keep][:, nodes_to_keep]
                attr_matrix = attr_matrix[nodes_to_keep]
                labels = labels[nodes_to_keep]

            adj_matrix = adj_matrix.tocoo()

        g = dgl_graph((adj_matrix.row, adj_matrix.col), num_nodes=adj_matrix.shape[0])
        # g = transform.to_bidirected(g)
        g.ndata['feat'] = torch.FloatTensor(attr_matrix)
        g.ndata['label'] = torch.LongTensor(labels)
        return g

    @property
    def num_classes(self):
        """Number of classes."""
        return self._graph.ndata['label'].max().item() + 1
    
    @property
    def save_path(self):
        r"""Path to save the processed dataset.
        """
        return os.path.join(self._save_dir, 'GraphWarData/datasets', self.name)    

    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        r"""Number of graphs in the dataset"""
        return 1
