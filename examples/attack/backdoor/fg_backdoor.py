from greatx.attack.backdoor import FGBackdoor
import torch
import torch_geometric.transforms as T

from tqdm import tqdm

from greatx.datasets import GraphDataset
from greatx.nn.models import GCN
from greatx.training.trainer import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

# ================================================================== #
#                      Load datasets                                 #
# ================================================================== #
dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
model = GCN(data.x.size(-1), data.y.max().item() + 1)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})

target_class = 0
predict = trainer.predict({'data': data}).argmax(-1)
count = (predict == target_class).int().sum().item()
print(f"{count/data.num_nodes:.2%} of nodes are classified as class {target_class} before attack")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #

attacker = FGBackdoor(data, device=device)
attacker.setup_surrogate(model)
attacker.reset()
attacker.attack(num_budgets=50, target_class=0)


count = 0
for t in tqdm(range(data.num_nodes)):
    t_class = trainer.predict({'data': attacker.data(t), 'mask': t}).argmax(-1)
    if t_class == target_class:
        count += 1

print(f"{count/data.num_nodes:.2%} of nodes are classified as class {target_class} after backdoor attack")
