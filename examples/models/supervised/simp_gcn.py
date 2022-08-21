import torch
import torch_geometric.transforms as T

from greatx.dataset import GraphDataset
from greatx import set_seed
from greatx.nn.models import SimPGCN
from greatx.training import SimPGCNTrainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SimPGCN(data.x.size(-1), data.y.max().item() + 1, hids=128)
trainer = SimPGCNTrainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
