import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch import nn
from torch_geometric.nn import Linear, HANConv, HGTConv
from typing import Dict, Union


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.linear_layer = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.linear_layer(x)
        return x


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=128,
            num_layers=3,
            dropout_ratio=0.1,
            embedding_dim=128
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.layers = torch.nn.ModuleList()

        if num_layers == 1:
            self.layers.append(torch.nn.Linear(input_dim, output_dim))
        elif num_layers == 2:
            self.layers.append(torch.nn.Linear(input_dim, embedding_dim))
            self.layers.append(torch.nn.Linear(embedding_dim, output_dim))
        else:
            self.layers.append(torch.nn.Linear(input_dim, hidden_dim))

            for i in range(num_layers - 3):
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

            self.layers.append(torch.nn.Linear(hidden_dim, embedding_dim))
            self.layers.append(torch.nn.Linear(embedding_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


class HAN(nn.Module):
    def __init__(self, data, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, num_layers = 1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata()))
        for _ in range(num_layers - 1):
            conv = HANConv(hidden_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, node_type):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            embedding = x_dict
        out = self.lin(x_dict[node_type])
        return out, embedding

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, node_type):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        out = self.lin(x_dict[node_type])

        return out, x_dict
