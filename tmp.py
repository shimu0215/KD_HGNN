import numpy as np
import scipy.sparse as sp
import os.path as osp

from old_version.load_data import load_acm
import torch
from torch_geometric.data import (
    HeteroData,
)

class ACM():
    def __init__(self, feat, metapath, label, train, val, test):
        self.feat = feat
        self.metapath = metapath
        self.label = label
        self.train = train
        self.val = val
        self.test = val

    def process(self):
        data = HeteroData()

        # node_types = ['a', 'p', 's']
        # for i, node_type in enumerate(node_types):
        #     x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
        data['author'].x = self.feat_a
        data['paper'].x = self.feat_p
        data['subject'].x = self.feat_s

        # y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['paper'].y = label

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['movie'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['movie'][f'{name}_mask'] = mask

        s = {}
        N_m = data['movie'].num_nodes
        N_d = data['director'].num_nodes
        N_a = data['actor'].num_nodes
        s['movie'] = (0, N_m)
        s['director'] = (N_m, N_m + N_d)
        s['actor'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

[nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, train, val, test = load_acm([20, 40, 60], [4019, 7167, 60])
neighbor = {}
neighbor['a'] = [torch.mean(nei_a[i].float()) for i in range(len(nei_a))]
neighbor['s'] = [torch.mean(nei_s[i].float()) for i in range(len(nei_s))]
split = {}
split['train'] = train

data = ACM([feat_p, feat_a, feat_s], [pap, psp], label, train, val, test)
print('a')
