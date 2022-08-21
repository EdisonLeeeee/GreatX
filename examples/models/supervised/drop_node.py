import torch
import torch_geometric.transforms as T

from greatx.datasets import GraphDataset
from greatx import set_seed
from greatx.nn.models import GCN
from greatx.training import Trainer
from greatx.training.callbacks import ModelCheckpoint
from greatx.utils import split_nodes
from greatx.functional import drop_node


def drop_hook(self, inputs):
    x, edge_index, edge_weight = inputs
    return (x, *drop_node(edge_index, edge_weight, num_nodes=x.size(0), p=0.2, training=self.training))


dataset = GraphDataset(root='~/data/pygdata', name='cora',
                       transform=T.LargestConnectedComponents())

data = dataset[0]
splits = split_nodes(data.y, random_state=15)

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GCN(data.x.size(-1), data.y.max().item() + 1)
hook = model.register_forward_pre_hook(drop_hook)
# hook.remove() # remove hook
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
trainer.fit({'data': data, 'mask': splits.train_nodes},
            {'data': data, 'mask': splits.val_nodes}, callbacks=[ckp])
trainer.evaluate({'data': data, 'mask': splits.test_nodes})
