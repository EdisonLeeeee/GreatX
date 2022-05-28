import torch
import torch_geometric.transforms as T

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import GCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes
from graphwar.defense import GUARD, RandomGUARD, DegreeGUARD
from torch_geometric.utils import degree

dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes}, 
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})

defense = 'DegreeGUARD'

if defense == "GUARD":
    surrogate = GCN(dataset.num_features, dataset.num_classes, bias=False, acts=None)
    surrogate_trainer = Trainer(surrogate, device=device)
    ckp = ModelCheckpoint('guard.pth', monitor='val_acc')
    trainer.fit({'data': data, 'mask': splits.train_nodes}, 
                {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
    trainer.evaluate({'data': data, 'mask': splits.test_nodes})
    guard = GUARD(device=device)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
    guard.setup_surrogate(surrogate, data.x, deg, data.y[splits.train_nodes])
elif defense == "RandomGUARD":
    guard = RandomGUARD(data.num_nodes, device=device)
elif defense == "DegreeGUARD":
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
    guard = DegreeGUARD(deg, device=device)
else:
    raise ValueError(f"Unknown defense {defense}")

# get a defensed graph
defense_data = guard(data, target_nodes=2, k=50)

# get anchors nodes (potential attacker nodes)
anchors = guard.anchors(k=50)