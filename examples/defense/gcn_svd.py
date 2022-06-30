import torch
import torch_geometric.transforms as T

from greatx.dataset import GraphDataset
from greatx import set_seed
from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes
from greatx.defense import SVDPurification

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.Compose([T.LargestConnectedComponents(),
                                            SVDPurification(K=50)]))

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(dataset.num_features, dataset.num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
