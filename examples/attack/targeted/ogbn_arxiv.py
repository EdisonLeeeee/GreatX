import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from greatx.attack.targeted import SGAttack
from greatx.nn.models import SGC
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import BunchDict

root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = PygNodePropPredDataset(root=root, name='ogbn-arxiv',
                                 transform=T.ToUndirected())
data = dataset[0]
splits = dataset.get_idx_split()
splits = BunchDict(train_nodes=splits['train'], val_nodes=splits['valid'],
                   test_nodes=splits['test'])

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
trainer_before = Trainer(SGC(num_features, num_classes), device=device, lr=0.1,
                         weight_decay=1e-5)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes),
                   callbacks=[ckp])
trainer_before.cache_clear()
output = trainer_before.predict(data, mask=target)
print(f"Before attack (target_label={target_label})\n "
      f"{np.round(output.tolist(), 2)}")
print('-' * target_label * width + '----👆' +
      '-' * max(dataset.num_classes - target_label - 1, 0) * width)

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = SGAttack(data, device=device)
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
trainer_after = Trainer(SGC(num_features, num_classes), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit(attacker.data(), mask=(splits.train_nodes, splits.val_nodes),
                  callbacks=[ckp])
trainer_after.cache_clear()
output = trainer_after.predict(attacker.data(), mask=target)

print(f"After poisoning attack (target_label={target_label})\n "
      f"{np.round(output.tolist(), 2)}")
print('-' * target_label * width + '----👆' +
      '-' * max(dataset.num_classes - target_label - 1, 0) * width)
