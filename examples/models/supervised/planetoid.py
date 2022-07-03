import torch
import torch_geometric.transforms as T

from greatx import set_seed
from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='~/data/pygdata', name='Cora')
data = dataset[0]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': data.train_mask},
            {'data': data, 'mask': data.val_mask}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': data.test_mask})
