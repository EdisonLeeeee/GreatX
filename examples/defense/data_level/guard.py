import torch
from graphwar.data import GraphWarDataset
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.models import GCN, SGC
from graphwar.utils import split_nodes
from graphwar.defense.data_level import GUARD

data = GraphWarDataset('cora', verbose=True, standardize=True)
g = data[0]
y = g.ndata['label']
splits = split_nodes(y, random_state=15)

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y_train = y[splits.train_nodes]
y_val = y[splits.val_nodes]
y_test = y[splits.test_nodes]

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

defense = 'GUARD'

if defense == "GUARD":
    surrogate = GCN(num_feats, num_classes, bias=False, acts=None)
    surrogate_trainer = Trainer(surrogate, device=device)
    cb = ModelCheckpoint('guard.pth', monitor='val_accuracy')
    surrogate_trainer.fit(g, y_train, splits.train_nodes, val_y=y_val,
                          val_index=splits.val_nodes, callbacks=[cb], verbose=0)
    guard = GUARD(g.ndata['feat'], g.in_degrees(), device=device)
    guard.setup_surrogate(surrogate, y_train)
elif defense == "RandomGUARD":
    guard = RandomGUARD(g.num_nodes(), device=device)
elif defense == "DegreeGUARD":
    guard = DegreeGUARD(g.in_degrees(), device=device)
else:
    raise ValueError(f"Unknown defense {defense}")

# get a defensed graph
defense_g = guard(g, target_nodes=1, k=50)

# get anchors nodes (potential attacker nodes)
anchors = guard.anchors(k=50)
