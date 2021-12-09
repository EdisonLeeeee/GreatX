from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration of DGLGraph in GraphWar.
    
    assume that we have a DGLGraph instance `g`.
    
    * feat : node features in `g.ndata`
    * label : nodel labels in `g.ndata`
    * edge_weight : edge weights in `g.edata`
    
    Examples
    --------
    >>> from graphwar.config import Config
    >>> g.ndata[Config.feat] = YOUR_NODE_FEATURE
    >>> g.ndata[Config.label] = YOUR_NODE_LABEL
    >>> g.edata[Config.edge_weight] = YOUR_EDGE_WEIGHT
    
    """
    feat = 'feat'
    label = 'label'
    edge_weight = 'edge_weight'
