# ================================================================== #
#           Plot feature similarity of GCN and SGC                   #
# ================================================================== #
import torch

from graphwar import set_seed
from graphwar.data import GraphWarDataset
from graphwar.models import GCN, SGC
from graphwar.training import Trainer
from graphwar.training.callbacks import ModelCheckpoint
from graphwar.utils import split_nodes
from graphwar.utils import CKA

# ================================================================== #
#                      Load datasets                                 #
# ================================================================== #
data = GraphWarDataset('cora', verbose=True, standardize=True)
g = data[0]
splits = split_nodes(g.ndata['label'], random_state=15)

num_feats = g.ndata['feat'].size(1)
num_classes = data.num_classes
y_train = g.ndata['label'][splits.train_nodes]
y_val = g.ndata['label'][splits.val_nodes]
y_test = g.ndata['label'][splits.test_nodes]

set_seed(123)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
g = g.to(device)

# ================================================================== #
#                      Train your model1                             #
# ================================================================== #
model = GCN(num_feats, num_classes)
trainer1 = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer1.fit(g, y_train, splits.train_nodes, val_y=y_val,
             val_index=splits.val_nodes, callbacks=[ckp])
trainer1.evaluate(g, y_test, splits.test_nodes)


# ================================================================== #
#                      Train your model2                             #
# ================================================================== #
model = SGC(num_feats, num_classes)
trainer2 = Trainer(model, device=device)
ckp = ModelCheckpoint('model.pth', monitor='val_accuracy')
trainer2.fit(g, y_train, splits.train_nodes, val_y=y_val,
             val_index=splits.val_nodes, callbacks=[ckp])
trainer2.evaluate(g, y_test, splits.test_nodes)


# ================================================================== #
#                      Get CKA similarity                            #
# ================================================================== #

dataloader = trainer1.config_test_data(g)
m1 = trainer1.model
m2 = trainer2.model
cka = CKA(m1, m2)
cka.compare(dataloader)
print(cka.export())
cka.plot_results()
