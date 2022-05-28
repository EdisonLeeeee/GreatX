import torch
import numpy as np
import torch_geometric.transforms as T

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import GCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes
from graphwar.attack.targeted import Nettack

dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)
set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ================================================================== #
#                     Attack Setting                                 #
# ================================================================== #
target = 1  # target node to attack
target_label = data.y[target].item()
width = 5

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
trainer_before = Trainer(GCN(dataset.num_features, dataset.num_classes), device=device)
ckp = ModelCheckpoint('model_before.pth', monitor='val_acc')
trainer_before.fit({'data': data, 'mask': splits.train_nodes}, 
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
output = trainer_before.predict({'data': data, 'mask': target})
print(f"Before attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
print('-'* target_label * width + '----👆' + '-'* max(dataset.num_classes - target_label - 1, 0) * width)

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
output = trainer_before.predict({'data': attacker.data(), 'mask': target})
print(f"After evasion attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
print('-'* target_label * width + '----👆' + '-'* max(dataset.num_classes - target_label - 1, 0) * width)

# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
trainer_after = Trainer(GCN(dataset.num_features, dataset.num_classes), device=device)
ckp = ModelCheckpoint('model_after.pth', monitor='val_acc')
trainer_after.fit({'data': attacker.data(), 'mask': splits.train_nodes}, 
            {'data': attacker.data(), 'mask': splits.val_nodes}, callbacks=[ckp])
output = trainer_after.predict({'data': attacker.data(), 'mask': target})

print(f"After poisoning attack (target_label={target_label})\n {np.round(output.tolist(), 2)}")
print('-'* target_label * width + '----👆' + '-'* max(dataset.num_classes - target_label - 1, 0) * width)