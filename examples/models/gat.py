import torch
from graphwar.data import GraphWarDataset
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.models import GAT
from graphwar.utils import split_nodes
from graphwar import set_seed


# ============ Loading datasets ================================
data = GraphWarDataset('cora', verbose=True, standardize=True)
g = data[0]
splits = split_nodes(g.ndata['label'], random_state=15)

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y_train = g.ndata['label'][splits.train_nodes]
y_val = g.ndata['label'][splits.val_nodes]
y_test = g.ndata['label'][splits.test_nodes]

set_seed(123)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ============ Train you model ==================================
model = GAT(num_feats, num_classes)
trainer = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[ckp])
trainer.evaluate(g, y_test, splits.test_nodes)
