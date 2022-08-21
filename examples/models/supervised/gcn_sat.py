import torch
import torch_geometric.transforms as T

from greatx.dataset import GraphDataset
from greatx import set_seed
from greatx.nn.models import SAT
from greatx.defense import EigenDecomposition
from greatx.training import SATTrainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.Compose([T.LargestConnectedComponents(),
                                            EigenDecomposition(35)]))

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = SAT(data.x.size(-1), data.y.max().item() + 1)
trainer = SATTrainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
