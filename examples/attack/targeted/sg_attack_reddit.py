import torch
from dgl.data import RedditDataset
from graphwar import set_seed
from graphwar.training import Trainer
from graphwar.models import GCN, SGC
from graphwar.utils import BunchDict


# ================================================================== #
#                      Loading datasets                              #
# ================================================================== #
data = RedditDataset()
g = data[0]
splits = BunchDict(train_nodes=g.ndata.pop('train_mask').nonzero().squeeze(),
                   val_nodes=g.ndata.pop('val_mask').nonzero().squeeze(),
                   test_nodes=g.ndata.pop('test_mask').nonzero().squeeze())

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y = g.ndata['label']
y_train = y[splits.train_nodes]
y_val = y[splits.val_nodes]
y_test = y[splits.test_nodes]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

target = 0  # target node to attack

print(f"Target node {target} has label {y[target]}")

# ================================================================== #
#                      Before Attack                                 #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device, lr=0.1, weight_decay=5e-5)
trainer.fit(g, y_train, splits.train_nodes, epochs=200)
output = trainer.predict(g, target)

print(f"Before attack\n {output[y[target]].item()}")

# ================================================================== #
#                      Attacking                                     #
# ================================================================== #
from graphwar.attack.targeted import SGAttackLarge
# use large-scale version of SGAttack
attacker = SGAttackLarge(g, device='cpu')  # you can still use GPU for accelerating
attacker.set_max_perturbations(10)
attacker.setup_surrogate(model)
attacker.reset()
# since reddit is dense, we do not use the degree as attack budgets by default
attacker.attack(target, num_budgets=10)

# ================================================================== #
#                      After evasion Attack                          #
# ================================================================== #
model.cache_clear()  # Important! Since SGC has cached results
output = trainer.predict(attacker.g().to(device), target)
print(f"After evasion attack\n {output[y[target]].item()}")


# ================================================================== #
#                      After poisoning Attack                        #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device, lr=0.1, weight_decay=5e-5)
trainer.fit(attacker.g().to(device), y_train, splits.train_nodes, epochs=200)
output = trainer.predict(attacker.g().to(device), target)

print(f"After poisoning attack\n {output[y[target]].item()}")
