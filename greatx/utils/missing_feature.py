import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class MissingFeature(BaseTransform):
    r"""Implementation of :class:`MissingFeature`
    from the `"On the Unreasonable Effectiveness 
    of Feature propagation in Learning 
    on Graphs with Missing Node Features"
    <https://arxiv.org/abs/2111.12128>`_ paper (ICLR'21)

    :class:`MissingFeature` generates missing feature mask 
    indicating whether each feature is present or missing.
    according differemt stractegies.

    Parameters
    ----------
    missing_rate : float, optional
        ratio of missing features, by default 0.5
    missing_type : str, optional
        type of strategies to generate missing 
        feature mask. If `type`='uniform', then each feature of 
        each node is missing uniformly at random with probability 
        :obj:`missing_rate`. Instead, if `type`='structural', 
        either we observe all features for a node, 
        or we observe none. For each node
        there is a probability of :obj:`missing_rate` 
        of not observing any feature, by default 'uniform'
    missing_value : float, optional
        value to fill missing features, by default float("nan")    

    Reference:

    * https://github.com/twitter-research/feature-propagation

    """

    def __init__(self, missing_rate: float = 0.5,
                 missing_type: str = 'uniform',
                 missing_value: float = float("nan")):
        assert missing_type in ("uniform", "structural"), missing_type
        assert missing_rate < 1, missing_rate
        self.missing_rate = missing_rate
        self.missing_type = missing_type
        self.missing_value = missing_value

    def __call__(self, data: Data) -> Data:

        num_nodes, num_features = data.x.size()
        if self.missing_type == "structural":  # either remove all of a nodes features or none
            missing_mask = torch.bernoulli(torch.Tensor([self.missing_rate]).repeat(
                num_nodes)).bool().unsqueeze(1).repeat(1, num_features)
        else:
            missing_mask = torch.bernoulli(torch.Tensor(
                [self.missing_rate]).repeat(num_nodes, num_features)).bool()

        data.missing_mask = missing_mask
        data.x[missing_mask] = self.missing_value
        return data
