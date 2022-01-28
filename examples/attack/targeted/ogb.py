import torch
from ogb.nodeproppred import DglNodePropPredDataset

from graphwar import set_seed
from graphwar.attack.targeted import SGAttack
from graphwar.models import GCN, SGC
from graphwar.training import Trainer

# ================================================================== #
#                      Load datasets                                 #
# ================================================================== #
data = DglNodePropPredDataset(name="ogbn-arxiv")
splits = data.get_idx_split()
g, y = data[0]
y = y.flatten()

srcs, dsts = g.edges()
g.add_edges(dsts, srcs)
g = g.remove_self_loop()

num_feats = g.ndata["feat"].size(1)
num_classes = (y.max() + 1).item()
y_train = y[splits['train']]
y_val = y[splits['valid']]
y_test = y[splits['test']]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)


set_seed(123)
target = 2  # target node to attack

print(f"Target node {target} has label {y[target]}")

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device, lr=0.1, weight_decay=5e-5)
trainer.fit(g, y_train, splits['train'], epochs=200)
print(trainer.evaluate(g, y_test, splits['test']))
output = trainer.predict(g, target)
print(f"Before attack\n {output[y[target]].item()}")


# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
attacker = SGAttack(g, device=device, label=y)
attacker.setup_surrogate(model)
attacker.reset()
attacker.attack(target)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
model.cache_clear()  # Important! Since SGC has cached results
output = trainer.predict(attacker.g(), target)
print(f"After evasion attack\n {output[y[target]].item()}")


# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device, lr=0.1, weight_decay=5e-5)
trainer.fit(attacker.g(), y_train, splits['train'], epochs=200)
output = trainer.predict(attacker.g(), target)

print(f"After poisoning attack\n {output[y[target]].item()}")
