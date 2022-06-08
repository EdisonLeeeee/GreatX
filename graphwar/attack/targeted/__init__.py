from .targeted_attacker import TargetedAttacker
from .random_attack import RandomAttack
from .dice_attack import DICEAttack
from .fg_attack import FGAttack
from .sg_attack import SGAttack
from .gf_attack import GFAttack
from .ig_attack import IGAttack
from .nettack import Nettack

classes = __all__ = ['TargetedAttacker', 'RandomAttack',
                     'DICEAttack', 'FGAttack', 'IGAttack', 'SGAttack',
                     'Nettack', 'GFAttack',
                     ]
