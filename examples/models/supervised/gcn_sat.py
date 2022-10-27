import os.path as osp

import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.defense import EigenDecomposition
from greatx.nn.models import SAT
from greatx.training import SATTrainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = GraphDataset(
    root=root, name=dataset, transform=T.Compose(
        [T.LargestConnectedComponents(),
         EigenDecomposition(35)]))

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAT(num_features, num_classes)
trainer = SATTrainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit(data, mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp])
trainer.evaluate(data, splits.test_nodes)
