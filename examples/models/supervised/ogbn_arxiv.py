import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from greatx.nn.models import GCN
from greatx.training.callbacks import ModelCheckpoint
from greatx.training.trainer import Trainer
from greatx.utils import BunchDict, split_nodes

dataset = PygNodePropPredDataset(root='~/data/pygdata', name=f'ogbn-arxiv',
                                 transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
data = dataset[0]
splits = dataset.get_idx_split()

splits = BunchDict(train_nodes=splits['train'],
                   val_nodes=splits['valid'],
                   test_nodes=splits['test'])


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(data.x.size(-1), data.y.max().item() + 1, hids=[256, 256], bn=True)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
