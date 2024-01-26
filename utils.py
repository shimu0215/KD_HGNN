from sklearn.metrics import f1_score
import random
import numpy as np
import torch
import torch.nn.functional as F


def get_f1_macro(labels, predictions):
    macro_f1 = f1_score(labels, predictions, average='macro')
    return macro_f1


def get_f1_micro(labels, predictions):
    micro_f1 = f1_score(labels, predictions, average='micro')
    return micro_f1


def evaluate_model(data, node_type, pred):
    mask = data[node_type]['test_mask']
    Y_test = data[node_type].y[mask]
    predctions = pred[mask]
    acc = (predctions == Y_test).sum() / mask.sum()
    f1_macro = get_f1_macro(labels=Y_test, predictions=predctions)
    f1_micro = get_f1_micro(labels=Y_test, predictions=predctions)

    return acc, f1_macro, f1_micro


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_similarity(data, emb, sample=False, seed=123):
    similarity_list = []
    indices_list = []

    for edge_type, edge_index in data.edge_index_dict.items():
        indices = range(len(edge_index[0]))
        if sample:
            set_seed(seed)
            indices = random.sample(indices, 1000)

        similarity = F.cosine_similarity(emb[edge_index[0][indices]], emb[edge_index[1][indices]], dim=1, eps=1e-8)

        similarity_list.append(similarity)
        indices_list.append(indices)

    return similarity_list, indices_list