import numpy as np
import torch
import torch_geometric.transforms as T

from greatx.attack.untargeted import RandomAttack
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
trainer_before = Trainer(
    GCN(data.x.size(-1), data.y.max().item() + 1), device=device)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit({'data': data, 'mask': splits.train_nodes},
                   {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
logs = trainer_before.evaluate({'data': data, 'mask': splits.test_nodes})
print(f"Before attack\n {logs}")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = RandomAttack(data, device=device)
attacker.reset()
attacker.attack(0.05)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
logs = trainer_before.evaluate(
    {'data': attacker.data(), 'mask': splits.test_nodes})
print(f"After evasion attack\n {logs}")
# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
trainer_after = Trainer(
    GCN(data.x.size(-1), data.y.max().item() + 1), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit({'data': attacker.data(), 'mask': splits.train_nodes},
                  {'data': attacker.data(), 'mask': splits.val_nodes}, callbacks=[ckp])
logs = trainer_after.evaluate(
    {'data': attacker.data(), 'mask': splits.test_nodes})
print(f"After poisoning attack\n {logs}")
