import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer

dataset = Planetoid(root='~/data/pygdata', name='Cora')
data = dataset[0]


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(data.x.size(-1), data.y.max().item() + 1)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': data.train_mask},
            {'data': data, 'mask': data.val_mask}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': data.test_mask})
