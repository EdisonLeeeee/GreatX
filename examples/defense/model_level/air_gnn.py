import torch

from graphwar import set_seed
from graphwar.data import GraphWarDataset
from graphwar.defense.model_level import AirGNN
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

set_seed(42)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
model = GCN(num_feats, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[ckp])
logs = trainer.evaluate(g, y_test, splits.test_nodes)

print(f"Before attack\n {logs}")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
from graphwar.attack.untargeted import FGAttack

attacker = FGAttack(g, device=device)
attacker.setup_surrogate(model, splits.train_nodes)
attacker.reset()
attacker.attack(0.05)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
model = AirGNN(num_feats, num_classes, k=10)
trainer = Trainer(model, device=device)
trainer.fit(g, y_train, splits.train_nodes)
logs = trainer.evaluate(attacker.g(), y_test, splits.test_nodes)

print(f"After evasion attack\n {logs}")

# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
model = AirGNN(num_feats, num_classes, k=10)
trainer = Trainer(model, device=device)
trainer.fit(attacker.g(), y_train, splits.train_nodes)
logs = trainer.evaluate(attacker.g(), y_test, splits.test_nodes)

print(f"After poisoning attack\n {logs}")
