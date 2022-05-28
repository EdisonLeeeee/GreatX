import torch
import torch_geometric.transforms as T

from graphwar.dataset import GraphWarDataset
from graphwar import set_seed
from graphwar.nn.models import GCN, GAT
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes

dataset = GraphWarDataset(root='~/data/pygdata', name='cora', 
                          transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


m1 = GCN(dataset.num_features, dataset.num_classes)
t1 = Trainer(m1, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t1.fit({'data': data, 'mask': splits.train_nodes}, 
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
t1.evaluate({'data': data, 'mask': splits.test_nodes})

m2 = GAT(dataset.num_features, dataset.num_classes)
t2 = Trainer(m2, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
t2.fit({'data': data, 'mask': splits.train_nodes}, 
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
t2.evaluate({'data': data, 'mask': splits.test_nodes})


from graphwar.utils import CKA
cka = CKA(m1, m2, device=device)
cka.compare(data)
print(cka.export())
cka.plot_results()
