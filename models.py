import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch import nn
from torch_geometric.nn import HANConv, HGTConv, Linear
from typing import Dict, List, Union


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.linear_layer = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x_post = torch.relu(x)
        x_post = self.linear_layer(x_post)
        return x, x_post


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=128,
            num_layers=2,
            dropout_ratio=0.4,
            norm_type="none",

    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = torch.nn.Dropout(dropout_ratio)
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if num_layers == 1:
            self.layers.append(torch.nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(torch.nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(torch.nn.LayerNorm(hidden_dim))

            self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


class HAN(nn.Module):
    def __init__(self, data, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, node_type, return_metapath_level_embedding=False):
        if return_metapath_level_embedding:
            out, semantic_attention_weights, metapath_level_embedding = self.han_conv(x_dict, edge_index_dict,
                                                                                      return_metapath_level_embedding=return_metapath_level_embedding)
            out = self.lin(out[node_type])
            return out, semantic_attention_weights, metapath_level_embedding

        out = self.han_conv(x_dict, edge_index_dict,
                            return_metapath_level_embedding=return_metapath_level_embedding)
        out = self.lin(out[node_type])
        return out

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

        return self.lin(x_dict[node_type])

