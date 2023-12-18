import torch
import numpy as np
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB, DBLP


def resample(data, train_ratio, val_ratio, node_type='movie'):

    total_size = len(data.x_dict[node_type])

    num_train = int(train_ratio * total_size)
    num_val = int(val_ratio * total_size)

    indices = np.arange(total_size)

    np.random.shuffle(indices)

    train_mask = np.zeros(total_size, dtype=bool)
    val_mask = np.zeros(total_size, dtype=bool)
    test_mask = np.zeros(total_size, dtype=bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    data[node_type].train_mask = torch.tensor(train_mask.tolist())
    data[node_type].val_mask = torch.tensor(val_mask.tolist())
    data[node_type].test_mask = torch.tensor(test_mask.tolist())


def load_data_metapath(args):

    if args.dataset == 'IMDB':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = IMDB(path, transform=transform)
        data = dataset[0]

    if args.dataset == 'DBLP':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = DBLP(path, transform=transform)
        data = dataset[0]

        # resample(data, node_type='movie', train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    return data

def load_data_HGT(args):

    if args.dataset == 'IMDB':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        dataset = IMDB(path)
        data = dataset[0]

    if args.dataset == 'DBLP':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        dataset = DBLP(path, transform=T.Constant(node_types='conference'))
        data = dataset[0]


    return data
