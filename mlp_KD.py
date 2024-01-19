import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loss_function import LogitLoss, StructureLoss, EmbeddingLoss
from data_prepare import load_data_metapath

from models import MLP, GNN

from typing import Dict, List, Union
import random

def get_similarity(data, emb, sample=False, seed=123):

    similarity_list = []
    indices_list = []

    for edge_type, edge_index in data.edge_index_dict.items():
        indices = [i for i in range(len(edge_index[0]))]
        if sample:
            random.seed(seed)
            indices = random.sample(indices, 1000)
        # edge_index = edge_index[indices]

        # X = emb[node_type]
        similarity = F.cosine_similarity(emb[edge_index[0][indices]], emb[edge_index[1][indices]], dim=1)
        similarity_list.append(similarity)
        indices_list.append(indices)

    return similarity_list, indices_list

def run_mlp_KD(args, data, neighbor = None):

    node_type = args.node
    num_class = args.num_class

    X = data.x_dict[node_type]

    train_mask = data[node_type].train_mask
    Y_train = data[node_type].y[train_mask]

    test_mask = data[node_type].test_mask
    Y_test = data[node_type].y[test_mask]

    teacher_predict = torch.load('GNN_result/'+args.dataset+'/'+args.teacher_model+'/result')
    teacher_embedding = torch.load('GNN_result/' + args.dataset + '/' + args.teacher_model + '/embedding')
    teacher_similarity = torch.load('GNN_result/'+args.dataset+'/'+args.teacher_model+'/sim')
    metapath_data, _ = load_data_metapath(args)

    if args.use_neighbor :
        input_size = X.shape[1]
        for item in neighbor:
            input_size += neighbor[item].shape[1]
            X = torch.cat((X, neighbor[item]), dim=1)
    else:
        input_size = X.shape[1]

    X_train = X[train_mask]
    X_test = X[test_mask]

    hidden_size = args.hidden_size
    dropout_ratio = args.dropout_ratio
    num_layers = args.num_layers
    embedding_dim = teacher_embedding[node_type].shape[1]
    output_size = num_class
    model = MLP(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size, dropout_ratio=dropout_ratio,
                num_layers=num_layers, embedding_dim=embedding_dim)

    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logit_criterion = LogitLoss()
    embedding_criterion = EmbeddingLoss()
    structure_criterion = StructureLoss()

    @torch.no_grad()
    def test() -> List[float]:

        model.eval()
        _, pred = model(X)
        pred = pred.argmax(dim=-1)

        accs = []
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = data[node_type][split]
            acc = (pred[mask] == data[node_type].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
        return accs

    start_patience = patience = args.patient

    best_val_acc = 0
    num_epochs = args.epochs

    model.train()
    for epoch in range(num_epochs):

        student_embedding, student_predict = model(X)
        predict_train = student_predict[train_mask]
        student_similarity, student_sim_ind = get_similarity(metapath_data, emb=student_embedding[0], sample=True, seed=epoch)
        gt_loss, logit_loss = logit_criterion(predict_train, student_predict, teacher_predict, Y_train)
        embedding_loss = embedding_criterion(student_embedding, teacher_embedding[node_type])
        structure_loss = structure_criterion(student_similarity, teacher_similarity, student_sim_ind)
        loss = args.logit_weight * logit_loss + args.emb_weight * embedding_loss\
            + args.struc_weight * structure_loss + args.gt_weight * gt_loss

        # print(logit_loss.item(), embedding_loss.item(), structure_loss.item(), gt_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = test()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
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

    with torch.no_grad():
        model.eval()
        _, predictions = model(X_test)
        acc = (predictions.argmax(dim=-1) == Y_test).sum() / test_mask.sum()

    return acc
