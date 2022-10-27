import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T

from greatx.attack.targeted import Nettack
from greatx.datasets import GraphDataset
from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')
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
width = 5

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
trainer_before = Trainer(
    GCN(data.x.size(-1),
        data.y.max().item() + 1, bias=False, acts=None), device=device)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp])
output = trainer_before.predict(data, mask=target)
print(f"Before attack (target_label={target_label})\n "
      f"{np.round(output.tolist(), 2)}")
print('-' * target_label * width + '----👆' +
      '-' * max(dataset.num_classes - target_label - 1, 0) * width)

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = Nettack(data, device=device)
attacker.setup_surrogate(trainer_before.model)
attacker.reset()
attacker.attack(target)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
output = trainer_before.predict(attacker.data(), mask=target)
print(f"After evasion attack (target_label={target_label})\n "
      f"{np.round(output.tolist(), 2)}")
print('-' * target_label * width + '----👆' +
      '-' * max(dataset.num_classes - target_label - 1, 0) * width)

# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
trainer_after = Trainer(GCN(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),
                  callbacks=[ckp])
output = trainer_after.predict(attacker.data(), mask=target)

print(f"After poisoning attack (target_label={target_label})\n "
      f"{np.round(output.tolist(), 2)}")
print('-' * target_label * width + '----👆' +
      '-' * max(dataset.num_classes - target_label - 1, 0) * width)
