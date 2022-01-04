
import dgl
import torch

from graphwar.config import Config

from .functions import pairwise_cosine_similarity

_EDGE_WEIGHT = Config.edge_weight


def knn_graph(x: torch.Tensor, k: int = 20):
    """Return a K-NN graph based on cosine similarity.

    """
    x = x.bool().float()  # x[x!=0] = 1
    sims = pairwise_cosine_similarity(x)
    sims = sims - torch.diag(sims)  # remove self-loops

    row = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    topk = torch.topk(sims, k=k, dim=1)
    col = topk.indices.flatten()
    edge_weight = topk.values.flatten()

    g = dgl.graph((row, col), device=x.device)
    g.edata[_EDGE_WEIGHT] = edge_weight

    return g
