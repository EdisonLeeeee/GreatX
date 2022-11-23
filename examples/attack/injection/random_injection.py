import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.attack.injection import RandomInjection
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = GraphDataset(root=root, name=dataset,
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
trainer_before = Trainer(GCN(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp])
logs = trainer_before.evaluate(data, splits.test_nodes)
print(f"Before attack\n {logs}")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = RandomInjection(data, device=device)
attacker.reset()
attacker.attack(10, feat_limits=(0, 1))  # for continuous features
# attacker.attack(10, feat_budgets=10)  # for binary features

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
logs = trainer_before.evaluate(attacker.data(), splits.test_nodes)
print(f"After evasion attack\n {logs}")
# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
trainer_after = Trainer(GCN(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),
                  callbacks=[ckp])
logs = trainer_after.evaluate(attacker.data(), splits.test_nodes)
print(f"After poisoning attack\n {logs}")
