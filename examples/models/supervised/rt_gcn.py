import torch
import torch_geometric.transforms as T

from greatx.dataset import GraphDataset
from greatx import set_seed
from greatx.nn.models import RTGCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes
from greatx.defense import TSVD

num_channels = 3
svd_rank = 50
set_seed(123)

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.Compose([T.LargestConnectedComponents(),
                                            T.NormalizeFeatures(),
                                            TSVD(svd_rank, num_channels)]))

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = RTGCN(data.x.size(-1), data.y.max().item() + 1, data.num_nodes, num_channels)
trainer = Trainer(model, device=device, weight_decay=5e-3)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
