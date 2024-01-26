import torch
import numpy as np
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB, DBLP, OGB_MAG


def re_split(args, data):
    node_type = args.node
    total_size = len(data.x_dict[node_type])

    num_train = int(args.train_ratio * total_size)
    num_val = int(args.val_ratio * total_size)

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

    return data


def get_one_hop_neighbor(args, data):
    edge_dict = data.edge_index_dict
    neighbor_dict = {}
    for edge_type in edge_dict:
        # get metapath type
        if edge_type[0] != args.node:
            continue
        current_type = edge_dict[edge_type]

        # initialize
        neighbors = torch.zeros([data.x_dict[args.node].shape[0], data.x_dict[edge_type[2]][0].shape[0]])
        counts = torch.zeros(data.x_dict[args.node].shape[0])

        for i in range(edge_dict[edge_type].shape[1]):
            neighbors[current_type[0][i]] += data.x_dict[edge_type[2]][current_type[1][i]]
            counts[current_type[0][i]] += 1

        neighbor_dict[edge_type[2]] = neighbors / counts.unsqueeze(1).expand_as(neighbors)

    return neighbor_dict


def load_data_metapath(args):
    if args.dataset == 'IMDB':
        args.node = 'movie'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, IMDB(path).data)
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = IMDB(path, transform=transform)
        data = dataset[0]

    if args.dataset == 'DBLP':
        args.node = 'author'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, IMDB(path).data)
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = DBLP(path, transform=transform)
        data = dataset[0]

    if args.dataset == 'OGB':
        args.node = 'paper'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB_MAG')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, OGB_MAG(path, preprocess='TransE').data)
        metapaths = [[('paper', 'author'), ('author', 'paper')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = OGB_MAG(path, preprocess='TransE', transform=transform)
        data = dataset.data

    if args.split:
        data = re_split(args, data)

    return data, neighbor


def load_data_HGT(args):
    if args.dataset == 'IMDB':
        args.node = 'movie'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, IMDB(path).data)
        dataset = IMDB(path)
        data = dataset[0]

    if args.dataset == 'DBLP':
        args.node = 'author'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, IMDB(path).data)
        dataset = DBLP(path, transform=T.Constant(node_types='conference'))
        data = dataset[0]

    if args.dataset == 'OGB':
        args.node = 'paper'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB_MAG')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, OGB_MAG(path, preprocess='TransE').data)
        dataset = OGB_MAG(path, preprocess='TransE')
        data = dataset.data

    if args.split:
        data = re_split(args, data)

    return data, neighbor
