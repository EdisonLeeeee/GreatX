import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.nn.models import GAT, GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import CKA, split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = GraphDataset(root=root, name=dataset,
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m1 = GCN(num_features, num_classes)
t1 = Trainer(m1, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t1.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
t1.evaluate(data, splits.test_nodes)

m2 = GAT(num_features, num_classes)
t2 = Trainer(m2, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t2.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
t2.evaluate(data, splits.test_nodes)

cka = CKA(m1, m2, device=device)
cka.compare(data)
print(cka.export())
cka.plot_results()
