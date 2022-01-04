import torch
from tqdm import tqdm

from graphwar import set_seed
from graphwar.data import GraphWarDataset
from graphwar.models import GCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes

# ================================================================== #
#                      Loading datasets                              #
# ================================================================== #
data = GraphWarDataset('cora', verbose=True, standardize=True)
g = data[0]
splits = split_nodes(g.ndata['label'], random_state=15)

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y_train = g.ndata['label'][splits.train_nodes]
y_val = g.ndata['label'][splits.val_nodes]
y_test = g.ndata['label'][splits.test_nodes]


set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
model = GCN(num_feats, num_classes, hids=[32], bias=False)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[ckp])

target_class = 0
predict = trainer.predict(g).argmax(-1)
count = (predict == target_class).int().sum().item()
print(f"{count/g.num_nodes():.2%} of nodes are classified as class {target_class} before attack")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
from graphwar.attack.backdoor import LGCBackdoor

attacker = LGCBackdoor(g, device)
attacker.setup_surrogate(model)
attacker.reset()
attacker.attack(num_budgets=50, target_class=0)


count = 0
for t in tqdm(range(g.num_nodes())):
    attacked_g = attacker.g(t)
    t_class = trainer.predict(attacked_g, t).argmax(-1)
    if t_class == target_class:
        count += 1

print(f"{count/g.num_nodes():.2%} of nodes are classified as class {target_class} after backdoor-attack")
