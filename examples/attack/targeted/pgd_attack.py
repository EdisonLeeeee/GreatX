import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.attack.targeted import PGDAttack
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import mark, split_nodes

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
#                     Attack Setting                                 #
# ================================================================== #
target = 1  # target node to attack
target_label = data.y[target].item()

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
trainer_before = Trainer(GCN(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp])
output = trainer_before.predict(data, mask=target)
print("Before attack:")
print(mark(output, target_label))

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = PGDAttack(data, device=device)
attacker.setup_surrogate(trainer_before.model)
attacker.reset()
attacker.attack(target)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
output = trainer_before.predict(attacker.data(), mask=target)
print("After evasion attack:")
print(mark(output, target_label))

# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
trainer_after = Trainer(GCN(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),
                  callbacks=[ckp])
output = trainer_after.predict(attacker.data(), mask=target)
print("After poisoning attack:")
print(mark(output, target_label))
