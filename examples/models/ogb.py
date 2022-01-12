import torch

from graphwar import set_seed
from graphwar.models import GCN
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from ogb.nodeproppred import DglNodePropPredDataset


# ================================================================== #
#                      Loading datasets                              #
# ================================================================== #
data = DglNodePropPredDataset(name='ogbn-arxiv')
splits = data.get_idx_split()
g, y = data[0]
y = y.flatten()

srcs, dsts = g.edges()
g.add_edges(dsts, srcs)
g = g.remove_self_loop()

num_feats = g.ndata["feat"].size(1)
num_classes = (y.max() + 1).item()
y_train = y[splits['train']]
y_val = y[splits['valid']]
y_test = y[splits['test']]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ================================================================== #
#                      Train Your Model                               #
# ================================================================== #
model = GCN(num_feats, num_classes, hids=[256, 256], bn=True)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits['train'], val_y=y_val, val_index=splits['valid'], callbacks=[ckp])
trainer.evaluate(g, y_test, splits['test'])
