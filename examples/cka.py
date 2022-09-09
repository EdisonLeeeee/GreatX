from greatx.utils import CKA
import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset

from greatx.nn.models import GCN, GAT
from greatx.training.trainer import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes

dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


m1 = GCN(data.x.size(-1), data.y.max().item() + 1)
t1 = Trainer(m1, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t1.fit({'data': data, 'mask': splits.train_nodes},
       {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
t1.evaluate({'data': data, 'mask': splits.test_nodes})

m2 = GAT(data.x.size(-1), data.y.max().item() + 1)
t2 = Trainer(m2, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t2.fit({'data': data, 'mask': splits.train_nodes},
       {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
t2.evaluate({'data': data, 'mask': splits.test_nodes})


cka = CKA(m1, m2, device=device)
cka.compare(data)
print(cka.export())
cka.plot_results()
