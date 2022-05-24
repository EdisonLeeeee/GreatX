"""Graph benchmark datasets for adversarial graph learning."""
import os

import numpy as np
import scipy.sparse as sp
import torch
# from dgl import transform
from dgl.convert import graph as dgl_graph
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import download, load_graphs, save_graphs
from sklearn.preprocessing import LabelEncoder

_DATASETS = {
    'citeseer', 'citeseer_full', 'cora', 'cora_ml', 'cora_full', 'amazon_cs',
    'amazon_photo', 'coauthor_cs', 'coauthor_phy',
    'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai', 'reddit'
}


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix.

    Args:
        A (scipy.sparse.csr_matrix): Compressed sparse matrix represents the adjacency matrix of the graph. 

    Returns:
        scipy.sparse.csr_matrix: The adjacency matrix of the graph with eliminated self-loop.

    """
    A = A - sp.diags(A.diagonal(), format='csr')
    A.eliminate_zeros()
    return A


def largest_connected_components(A):
    """Get the largest connected component of A.

    Args:
        A (array_like or sparse matrix): 
            The N x N matrix representing the compressed sparse graph(csgraph). 
            The input csgraph will be converted to csr format for the calculation.

    Returns:
        numpy.array: Nodes in the largest connected component.

    """
    _, component_indices = sp.csgraph.connected_components(A)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[-1]
    nodes_to_keep = np.where(component_indices == components_to_keep)[0]
    return nodes_to_keep


def _get_adversarial_graph_url(file_url):
    """Get online dataset url for download.

    Args:
        file_url (str): Data filename to download.

    Returns:
        str: Complete data file download path.
    """
    # repo_url = 'https://github.com/EdisonLeeeee/GraphWarData/raw/master/datasets/'
    repo_url = 'https://gitee.com/EdisonLeeeee/GraphWarData/raw/master/datasets/'
    return repo_url + file_url


class GraphWarDataset(DGLBuiltinDataset):
    """Base Class for adversarial graph dataset.

    References: 
        * [1] GitHub: https://github.com/EdisonLeeeee/GraphWarData
        * [2] Gitee: https://gitee.com/EdisonLeeeee/GraphWarData


    Available Datasets:
        ['citeseer', 'citeseer_full', 'cora', 'cora_ml', 'cora_full', 'amazon_cs',
        'amazon_photo', 'coauthor_cs', 'coauthor_phy',
        'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai', 'reddit']

    
    Args:
        name (str): Dataset name. 
            It can be chosen from ['citeseer', 'citeseer_full', 'cora', 
            'cora_ml', 'cora_full', 'amazon_cs',
            'amazon_photo', 'coauthor_cs', 'coauthor_phy',
            'pubmed', 'flickr', 'blogcatalog', 'dblp', 'acm', 'uai', 'reddit']

        raw_dir (str): 
            Specifying the directory that will store the
            downloaded data or the directory that
            already stores the input data.
            Default: ~/.dgl/
        
        force_reload (bool): Whether to reload the dataset. Default: False

        verbose (bool): 
            Whether to print out progress information. Default: False

        standardize (bool):
            Whether to use the largest connected components of the graph. Default: True

    """

    def __init__(self, name, raw_dir=None,
                force_reload=False, verbose=False,
                standardize=True):
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
        """Automatically download raw data to local disk.
        Implement abstract method of the base class `DGLDataset`. 
        """
        download_path = os.path.join(self.raw_dir, self.name + '.npz')
        if not os.path.exists(download_path):
            if self.name == 'graphwar-reddit':
                raise RuntimeError("`reddit` dataset is too large to download. Please download it manually.")  # TODO: add reddit dataset links
            download(self.url, path=download_path)

    def process(self):
        """Process the input raw data to DGLGraph.
        Implement abstract method of the base class `DGLDataset`.
        """
        npz_path = os.path.join(self.raw_dir, self.name + '.npz')
        g = self._load_npz(npz_path)
        # g = transform.reorder_graph(
        #     g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
        self._graph = g
        self._data = [g]
        self._print_info()

    def has_cache(self):
        """Deciding whether there exists a cached dataset.
        Implement abstract method of the base class `DGLDataset`.
        
        Returns:
            bool : 
                If there exists a cached dataset in self.save_path, return `True`.
                Otherwise, return `False`.
        """
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        """Save the processed dataset into files.
        Implement abstract method of the base class `DGLDataset`.
        """
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        """Load the saved datasets from files.
        Implement abstract method of the base class `DGLDataset`.
        """
        graph_path = os.path.join(self.save_path, 'dgl_graph_v1.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        self._data = [graphs[0]]
        self._print_info()

    def _print_info(self):
        """Print information about the dataset: 
            * NumNodes: number of nodes in the graph.
            * NumEdges: number of edges in the graph.
            * NumFeats: number of input features of nodes.
            * NumClasses: number of output categories.
        """
        if self.verbose:
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._graph.ndata['feat'].shape[-1]))
            print('  NumClasses: {}'.format(self.num_classes))

    def _load_npz(self, file_name):
        """Load data from npz files.

        Args:
            file_name (str): Absolute path of the dataset, same as `self.save_path`.

        Returns:
            :class:`dgl.heterograph.DGLHeteroGraph`: Graph build from `file_name`.
        """
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            adj_matrix = loader['adj_matrix'].item()
            adj_matrix = adj_matrix.maximum(adj_matrix.T)
            adj_matrix = eliminate_self_loops(adj_matrix)
            attr_matrix = loader['attr_matrix']
            if attr_matrix.dtype.kind == 'O':
                # scipy sparse matrix
                attr_matrix = attr_matrix.item().A

            labels = loader['label']
            if labels.shape[0] != adj_matrix.shape[0]:
                _labels = np.full((adj_matrix.shape[0] - labels.shape[0],), -1)
                labels = np.hstack([labels, _labels])

            if self.standardize:
                nodes_to_keep = largest_connected_components(adj_matrix)
                adj_matrix = adj_matrix[nodes_to_keep][:, nodes_to_keep]
                attr_matrix = attr_matrix[nodes_to_keep]
                labels = labels[nodes_to_keep]

                if np.unique(labels).shape[0] != labels.max() + 1:
                    labels = LabelEncoder().fit_transform(labels)

            adj_matrix = adj_matrix.tocoo()

        g = dgl_graph((adj_matrix.row, adj_matrix.col),
                      num_nodes=adj_matrix.shape[0])
        # g = transform.to_bidirected(g)
        g.ndata['feat'] = torch.FloatTensor(attr_matrix)
        g.ndata['label'] = torch.LongTensor(labels)
        return g

    @property
    def num_classes(self):
        """int: Number of classes."""
        return self._graph.ndata['label'].max().item() + 1

    @property
    def save_path(self):
        """str: Path to save the processed dataset."""
        return os.path.join(self._save_dir, self.name)

    def __getitem__(self, idx):
        """Get graph by index

        Args:
            idx (int): Item index

        Returns:
            :class:`dgl.heterograph.DGLHeteroGraph`: 
                The graph contains:
                    - ``ndata['feat']``: node features
                    - ``ndata['label']``: node labels
        """
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        """Number of graphs in the dataset.

        Returns:
            int: Number of graphs in the dataset.
        """
        return 1
