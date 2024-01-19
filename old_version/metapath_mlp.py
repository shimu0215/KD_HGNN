import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import OGB_MAG, IMDB
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv.han_conv import group
from torch.utils.data import DataLoader, TensorDataset

from models import MLP, GNN

import os.path as osp
import numpy as np
from typing import Dict, List, Union


def resample(data):

    # Set the seed for reproducibility
    np.random.seed(42)

    # Total number of data points
    total_size = len(data.x_dict['movie'])

    # Specify the ratios for train, validation, and test sets
    train_ratio = 0.2
    val_ratio = 0.3
    test_ratio = 0.5

    # Calculate the sizes of each set
    num_train = int(train_ratio * total_size)
    num_val = int(val_ratio * total_size)
    num_test = total_size - num_train - num_val

    # Create an array of indices for the data points
    indices = np.arange(total_size)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Create masks
    train_mask = np.zeros(total_size, dtype=bool)
    val_mask = np.zeros(total_size, dtype=bool)
    test_mask = np.zeros(total_size, dtype=bool)

    # Assign indices to the masks
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    # Convert masks to lists if needed
    data['movie'].train_mask = torch.tensor(train_mask.tolist())
    data['movie'].val_mask = torch.tensor(val_mask.tolist())
    data['movie'].test_mask = torch.tensor(test_mask.tolist())


# Set random seed for reproducibility
torch.manual_seed(123)

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../../data/IMDB')
metapaths = [[('movie', 'actor'), ('actor', 'movie')],
             [('movie', 'director'), ('director', 'movie')]]
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                           drop_unconnected_node_types=True)
dataset = IMDB(path, transform=transform)

data = dataset[0]
print(data)
resample(data)

gnn_predict = torch.load('GNN_result/han/result')
gnn_metapath_prediction = torch.load('GNN_result/han/metapath')['movie']
gnn_semantic = torch.load('GNN_result/han/semantic')['movie']

gnn_linear_layer = nn.Linear(128, 3)
gnn_linear_layer.load_state_dict(torch.load('GNN_result/han/lin.pth'))

gnn_metapath_prediction1 = gnn_metapath_prediction[0]
gnn_metapath_prediction2 = gnn_metapath_prediction[1]

# Instantiate the model, define loss function and optimizer

train_mask = data['movie'].train_mask
X_train = data.x_dict['movie'][train_mask]
Y_train = data['movie'].y[train_mask]

test_mask = data['movie'].test_mask
X_test = data.x_dict['movie'][test_mask]
Y_test = data['movie'].y[test_mask]

input_size = X_train.shape[1]
hidden_size = 128
output_size = 128
model1 = MLP(num_layers=2, input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size)
model2 = MLP(num_layers=2, input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size)

optimizer1 = optim.Adam(model1.parameters(), lr=0.005, weight_decay=0.0)
optimizer2 = optim.Adam(model2.parameters(), lr=0.005, weight_decay=0.0)
criterion = nn.KLDivLoss()

# Training the model
target_data1 = gnn_metapath_prediction1[train_mask]
# log_target_data1 = torch.nn.functional.softmax(target_data1, dim=1)
target_data2 = gnn_metapath_prediction2[train_mask]
# log_target_data2 = torch.nn.functional.softmax(target_data2, dim=1)

batch_size = len(target_data1)
train_dataset1 = TensorDataset(X_train, target_data1)
train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)

train_dataset2 = TensorDataset(X_train, target_data2)
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)

@torch.no_grad()
def test() -> List[float]:

    model1.eval()
    _, pred1 = model1(data.x_dict['movie'])

    model2.eval()
    _, pred2 = model2(data.x_dict['movie'])

    pred_metapaths = torch.stack([pred1, pred2])
    pred_sem = gnn_linear_layer(pred_metapaths)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['movie'][split]
        out = torch.sum(gnn_semantic.view(2, 1, -1) * pred_sem, dim=0).argmax(-1)
        acc = (out[mask] == data['movie'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


start_patience = patience = 30

best_val_acc = 0
num_epochs = 300  # 0.5748 0.5458 0.6089
a = 0.0

model1.train()
model2.train()
for epoch in range(num_epochs):
    for (data1, labels1), (data2, labels2) in zip(train_loader1, train_loader2):
        # Forward pass
        _, outputs1 = model1(data1)
        # log_outputs1 = torch.nn.functional.log_softmax(outputs1, dim=1)

        # loss1 = criterion(log_outputs1, log_target_data1)
        loss1 = F.mse_loss(outputs1, labels1)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        _, outputs2 = model2(data2)
        # log_outputs2 = torch.nn.functional.log_softmax(outputs2, dim=1)

        # loss2 = criterion(log_outputs2, log_target_data2)
        loss2 = F.mse_loss(outputs2, labels2)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

    train_acc, val_acc, test_acc = test()

    print(f'Epoch: {epoch:03d}, Loss1: {loss1:.4f}, Loss2: {loss2:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    if best_val_acc <= val_acc:
        patience = start_patience
        best_val_acc = val_acc
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

#        dataset: acm dblp, hgt

# Testing the model
# with torch.no_grad():
#     model.eval()
#     _, predictions = model(X_test)
#     acc = (predictions.argmax(dim=-1) == Y_test).sum() / test_mask.sum()
#
# print(acc)
# 3.24