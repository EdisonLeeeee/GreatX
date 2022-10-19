import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from greatx.nn.models import DGI, LogisticRegression
from greatx.training import DGITrainer, MLPTrainer
from greatx.training.callbacks import EarlyStopping, ModelCheckpoint

dataset = Planetoid(root='~/data/pyg', name='Cora',
                    transform=T.NormalizeFeatures())
data = dataset[0]


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# ================================================================== #
#                 Self-supervised Learning                           #
# ================================================================== #
model = DGI(dataset.num_features, 512)
trainer = DGITrainer(model, device=device, lr=0.001, weight_decay=0.)
es = EarlyStopping(monitor='loss', patience=20)
trainer.fit({'data': data}, epochs=500, callbacks=[es])

# ================================================================== #
#                   Get node embedding                               #
# ================================================================== #
with torch.no_grad():
    embedding = model.encode(data.x, data.edge_index)

# ================================================================== #
#                    Linear evaluation                               #
# ================================================================== #
LR = LogisticRegression(embedding.size(1), dataset.num_classes)
LR_trainer = MLPTrainer(LR, device=device, weight_decay=0.)
ckp = ModelCheckpoint('model.pth', monitor='val_acc')
LR_trainer.fit({'x': embedding, 'y': data.y, 'mask': data.train_mask},
               {'x': embedding, 'y': data.y, 'mask': data.val_mask}, callbacks=[ckp], epochs=200)
LR_trainer.evaluate({'x': embedding, 'y': data.y, 'mask': data.test_mask})
