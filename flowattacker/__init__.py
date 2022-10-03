"""Top-level package for flow-attacker."""

name = "flow-attacker"

__author__ = 'Shengzhe Xu'
__email__ = 'shengzx@vt.edu'
__version__ = '0.0.1'

from flowattacker.helper import eval_helper
from flowattacker.model import behav2inst, context_loader

__all__ = [
    'eval_helper',
    'NetflowAttacker',
    'ScenarioDataset',
]

print("imported FlowAttacker")