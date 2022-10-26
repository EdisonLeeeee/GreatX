import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.nn.models import NLGAT, NLGCN, NLMLP
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = GraphDataset(root=root, name=dataset,
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, train=0.6, val=0.2, test=0.2, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
method = "NLGCN"
if method == "NLGCN":
    model = NLGCN(num_features, num_classes)
elif method == "NLMLP":
    model = NLMLP(num_features, num_classes)
else:
    model = NLGAT(num_features, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
trainer.evaluate(data, splits.test_nodes)
