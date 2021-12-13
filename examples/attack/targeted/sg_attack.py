import torch
from graphwar.data import GraphWarDataset
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.models import GCN, SGC
from graphwar.utils import split_nodes
from graphwar import set_seed


# ============ Loading datasets ================================
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

target = 1  # target node to attack

print(f"Target node {target} has label {g.ndata['label'][target]}")

# ============ Before Attack ==================================
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device, lr=0.1, weight_decay=5e-5)
trainer.fit(g, y_train, splits.train_nodes)
output = trainer.predict(g, target)

print(f"Before attack\n {output.tolist()}")

# ============ Attacking ==================================
from graphwar.attack.targeted import SGAttack
attacker = SGAttack(g, device=device)
attacker.setup_surrogate(model)
attacker.reset()
attacker.attack(target)

# ============ After evasion Attack ==================================
model.cache_clear()  # Important! Since SGC has cached results
output = trainer.predict(attacker.g(), target)
print(f"After evasion attack\n {output.tolist()}")


# ============ After poisoning Attack ==================================
model = GCN(num_feats, num_classes)
trainer = Trainer(model, device=device)
trainer.fit(attacker.g(), y_train, splits.train_nodes)
output = trainer.predict(attacker.g(), target)

print(f"After poisoning attack\n {output.tolist()}")
