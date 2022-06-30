import torch
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset
from greatx import set_seed
from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes, BunchDict

dataset = PygNodePropPredDataset(root='~/data/pygdata', name=f'ogbn-arxiv', 
                                 transform=T.ToUndirected())
data = dataset[0]
splits = dataset.get_idx_split()

splits = BunchDict(train_nodes=splits['train'], 
                   val_nodes=splits['valid'], 
                   test_nodes=splits['test'])

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(dataset.num_features, dataset.num_classes, hids=[256, 256], bn=True)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes}, 
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
