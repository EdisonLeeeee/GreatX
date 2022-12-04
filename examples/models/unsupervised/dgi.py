import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from greatx.nn.models import DGI, LogisticRegression
from greatx.training import Trainer, UnspuervisedTrainer
from greatx.training.callbacks import EarlyStopping, ModelCheckpoint

dataset = 'Cora'
root = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data')
dataset = Planetoid(root=root, name=dataset, transform=T.NormalizeFeatures())

data = dataset[0]

num_features = data.x.size(-1)
num_classes = data.y.max().item() + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================== #
#                 Self-supervised Learning                           #
# ================================================================== #
model = DGI(num_features, 512)
trainer = UnspuervisedTrainer(model, device=device, lr=0.001, weight_decay=0.)
es = EarlyStopping(monitor='loss', patience=20)
trainer.fit(data, epochs=500, callbacks=[es])

# ================================================================== #
#                   Get node embedding                               #
# ================================================================== #
model.eval()
with torch.no_grad():
    embedding = model.encode(data.x, data.edge_index)

# ================================================================== #
#                    Linear evaluation                               #
# ================================================================== #
LR = LogisticRegression(embedding.size(1), num_classes)
LR_trainer = Trainer(LR, device=device, weight_decay=0.)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
emb = Data(x=embedding, y=data.y)
LR_trainer.fit(emb, (data.train_mask, data.val_mask), callbacks=[ckp],
               epochs=200)
LR_trainer.evaluate(emb, data.test_mask)
