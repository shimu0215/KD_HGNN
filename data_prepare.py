import torch
import numpy as np
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB, DBLP, OGB_MAG
from torch_geometric.loader import HGTLoader


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
        if edge_type[0] != args.node:
            continue
        current_type = edge_dict[edge_type]

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
        transform = T.Constant(node_types='conference')
        data = dataset[0]
        data = transform(data)

    if args.dataset == 'DBLP':
        args.node = 'author'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, DBLP(path).data)
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        transform_2 = T.Constant(node_types='conference')
        dataset = DBLP(path, transform=transform)
        data = dataset[0]
        data = transform_2(data)

    if args.dataset == 'OGB':
        args.node = 'paper'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB_MAG')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, OGB_MAG(path, preprocess='metapath2vec').data)
        metapaths = [[('paper', 'author'), ('author', 'paper')]]
        transform_1 = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)

        transform_2 = T.ToUndirected(merge=True)
        dataset = OGB_MAG(path, preprocess='TransE', transform=transform_1)
        data = dataset.data
        data = transform_2(data)

    if args.split:
        data = re_split(args, data)

    return data, neighbor


def load_data_HGT(args, get_whole_OGB=False):

    if args.dataset == 'OGB':
        args.node = 'paper'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB_MAG')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = T.ToUndirected(merge=True)
        dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
        data = dataset[0].to(device, 'x', 'y')

        if get_whole_OGB:
            return data, {}

        train_input_nodes = ('paper', data['paper'].train_mask)
        val_input_nodes = ('paper', data['paper'].val_mask)
        test_input_nodes = ('paper', data['paper'].test_mask)
        kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}

        train_loader = HGTLoader(data, num_samples=[1024] * 4, shuffle=True,
                                 input_nodes=train_input_nodes, **kwargs)
        val_loader = HGTLoader(data, num_samples=[1024] * 4,
                               input_nodes=val_input_nodes, **kwargs)
        test_loader = HGTLoader(data, num_samples=[1024] * 4,
                               input_nodes=test_input_nodes, **kwargs)

        data = {}
        data['train'] = train_loader
        data['val'] = val_loader
        data['test'] = test_loader
        return data, {}

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
            neighbor = get_one_hop_neighbor(args, DBLP(path).data)
        dataset = DBLP(path, transform=T.Constant(node_types='conference'))
        data = dataset[0]

    if args.split:
        data = re_split(args, data)

    return data, neighbor

def load_data_homo(args):
    if args.dataset == 'IMDB':
        args.node = 'movie'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')

        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = IMDB(path, transform=transform)
        transform = T.Constant(node_types='conference')
        data = dataset[0]
        data = transform(data)

    if args.dataset == 'DBLP':
        args.node = 'author'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        neighbor = []
        if args.use_neighbor:
            neighbor = get_one_hop_neighbor(args, DBLP(path).data)
        metapaths = [[('author', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')],
                     [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        transform_2 = T.Constant(node_types='conference')
        dataset = DBLP(path, transform=transform)
        data = dataset[0]
        data = transform_2(data)

    if args.split:
        data = re_split(args, data)

    return data.to_homogeneous()
