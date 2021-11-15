import torch
from graphwar.data import GraphWarDataset
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.models import GCN
from graphwar.utils import split_nodes
from graphwar import set_seed


# ============ Loading datasets ================================
data = GraphWarDataset('cora', verbose=True, standardize=True)
g = data[0].add_self_loop()
splits = split_nodes(g.ndata['label'], random_state=15)

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y_train = g.ndata['label'][splits.train_nodes]
y_val = g.ndata['label'][splits.val_nodes]
y_test = g.ndata['label'][splits.test_nodes]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ============ Before Attack ==================================
model = GCN(num_feats, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('gcn.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[ckp])
logs = trainer.evaluate(g, y_test, splits.test_nodes)

print(f"Before random attack\n {logs}")

# ============ Attacking ==================================
from graphwar.attack.untargeted import Metattack
attacker = Metattack(g, device=device)
attacker.setup_surrogate(model, labeled_nodes=splits.train_nodes, unlabeled_nodes=splits.test_nodes, lambda_=0.)

attacker.reset()
attacker.attack(0.05)

# ============ After evasion Attack ==================================
logs = trainer.evaluate(attacker.g(), y_test, splits.test_nodes)
print(f"After evasion attack\n {logs}")


# ============ After poisoning Attack ==================================
model = GCN(num_feats, num_classes)
trainer = Trainer(model, device=device)
trainer.fit(attacker.g(), y_train, splits.train_nodes)
logs = trainer.evaluate(attacker.g(), y_test, splits.test_nodes)

print(f"After poisoning attack\n {logs}")
