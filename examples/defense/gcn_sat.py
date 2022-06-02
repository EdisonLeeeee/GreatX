import torch
import torch_geometric.transforms as T

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import SAT
from graphwar.defense import EigenDecomposition
from graphwar.training import SATTrainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes

dataset = GraphWarDataset(root='~/data/pygdata', name='cora',
                          transform=T.Compose([T.LargestConnectedComponents(),
                                               EigenDecomposition(35)]))

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SAT(dataset.num_features, dataset.num_classes)
trainer = SATTrainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
