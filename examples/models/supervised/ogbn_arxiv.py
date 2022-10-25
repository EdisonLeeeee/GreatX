import os.path as osp

import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import BunchDict

root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = PygNodePropPredDataset(
    root=root, name='ogbn-arxiv',
    transform=T.Compose([T.ToUndirected(),
                         T.ToSparseTensor()]))

data = dataset[0]
splits = dataset.get_idx_split()

splits = BunchDict(train_nodes=splits['train'], val_nodes=splits['valid'],
                   test_nodes=splits['test'])

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features, num_classes, hids=[256, 256], bn=True)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
trainer.evaluate(data, splits.test_nodes)
