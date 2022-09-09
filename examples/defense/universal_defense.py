import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.defense import GUARD, DegreeGUARD, RandomGUARD
from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(data.x.size(-1), data.y.max().item() + 1)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})

defense = 'GUARD'

if defense == "GUARD":
    surrogate = GCN(data.x.size(-1), data.y.max().item() + 1,
                    bias=False, acts=None)
    surrogate_trainer = Trainer(surrogate, device=device)
    ckp = ModelCheckpoint('guard.pth', monitor='val_acc')
    trainer.fit({'data': data, 'mask': splits.train_nodes},
                {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp], verbose=0)
    guard = GUARD(data, device=device)
    guard.setup_surrogate(surrogate, data.y[splits.train_nodes])
elif defense == "RandomGUARD":
    guard = RandomGUARD(data, device=device)
elif defense == "DegreeGUARD":
    guard = DegreeGUARD(data, device=device)
else:
    raise ValueError(f"Unknown defense {defense}")

# get a defensed graph
defense_data = guard(data, target_nodes=2, k=50)

# get anchors nodes (potential attacker nodes)
anchors = guard.anchors(k=50)
