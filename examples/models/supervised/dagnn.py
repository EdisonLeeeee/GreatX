import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx.nn.models import DAGNN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pyg', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DAGNN(data.x.size(-1), data.y.max().item() + 1)
trainer = Trainer(model, device=device, weight_decay=5e-3)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
