import torch

from graphwar import set_seed
from graphwar.models import SGC
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import BunchDict
from dgl.data import RedditDataset

# ================================================================== #
#                      Loading datasets                              #
# ================================================================== #
data = RedditDataset()
g = data[0]
splits = BunchDict(train_nodes=g.ndata.pop('train_mask').nonzero().squeeze(),
                   val_nodes=g.ndata.pop('val_mask').nonzero().squeeze(),
                   test_nodes=g.ndata.pop('test_mask').nonzero().squeeze())

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y = g.ndata['label']
y_train = y[splits.train_nodes]
y_val = y[splits.val_nodes]
y_test = y[splits.test_nodes]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ================================================================== #
#                      Train Your Model                               #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[ckp])
trainer.evaluate(g, y_test, splits.test_nodes)
