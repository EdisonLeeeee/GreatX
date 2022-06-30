import torch
import torch_geometric.transforms as T

from graphwar.dataset import GraphDataset
from graphwar import set_seed
from graphwar.nn.models import GCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes, MissingFeature
from graphwar.defense import FeaturePropagation

set_seed(123)
dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.Compose([T.LargestConnectedComponents(),
                                            MissingFeature(missing_rate=0.5)  # here we generate 50% missing features
                                            ]))


data = dataset[0]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = data.to(device)
data = FeaturePropagation(missing_mask=data.missing_mask)(data)
splits = split_nodes(data.y.cpu(), random_state=15)

model = GCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
