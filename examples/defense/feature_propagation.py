import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset

from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes, MissingFeature
from greatx.defense import FeaturePropagation


dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.Compose([T.LargestConnectedComponents(),
                                            # here we generate 50% missing features
                                            MissingFeature(missing_rate=0.5)
                                            ]))


data = dataset[0]
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
data = data.to(device)
data = FeaturePropagation(missing_mask=data.missing_mask)(data)
splits = split_nodes(data.y.cpu(), random_state=15)

model = GCN(data.x.size(-1), data.y.max().item() + 1)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
