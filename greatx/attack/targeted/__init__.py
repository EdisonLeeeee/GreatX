from .dice_attack import DICEAttack
from .fg_attack import FGAttack
from .gf_attack import GFAttack
from .ig_attack import IGAttack
from .nettack import Nettack
from .random_attack import RandomAttack
from .sg_attack import SGAttack
from .targeted_attacker import TargetedAttacker

classes = __all__ = ['TargetedAttacker', 'RandomAttack',
                     'DICEAttack', 'FGAttack', 'IGAttack', 'SGAttack',
                     'Nettack', 'GFAttack',
                     ]
