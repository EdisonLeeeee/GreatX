from .utils.check import is_edge_index
from .version import __version__
from . import utils
from . import nn
from . import functional
from . import attack
from . import datasets

__all__ = ['__version__', 'is_edge_index',
           'datasets', 'attack', 'defense',
           'training', 'nn', 'functional', 'utils']
