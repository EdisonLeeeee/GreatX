import torch
import torch_geometric.transforms as T

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import TAGCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes

dataset = GraphWarDataset(root='~/data/pygdata', name='cora',
                          transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TAGCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
