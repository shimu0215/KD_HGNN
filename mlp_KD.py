import torch
import torch.optim as optim
import torch.nn.functional as F
from loss_function import LogitLoss, StructureLoss, EmbeddingLoss, GtLoss
from data_prepare import load_data_metapath
from utils import get_f1_macro, get_f1_micro, get_similarity

from models import MLP, GNN

from typing import List
import random



def run_mlp_KD(args, data, neighbor=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metapath_data, _ = load_data_metapath(args)

    metapath_data = metapath_data.to(device)
    data = data.to(device)

    node_type = args.node
    num_class = data[node_type].y.unique().size(0)

    # split data
    train_mask = data[node_type].train_mask
    test_mask = data[node_type].test_mask
    X = data.x_dict[node_type]
    Y_train = data[node_type].y[train_mask]
    Y_test = data[node_type].y[test_mask]

    # append one hop neighbors
    if args.use_neighbor:
        input_size = X.shape[1]
        for item in neighbor:
            input_size += neighbor[item].shape[1]
            X = torch.cat((X, neighbor[item]), dim=1)
    else:
        input_size = X.shape[1]

    X_test = X[test_mask]

    # load teacher result
    teacher_logit = torch.load('GNN_result/' + args.dataset + '/' + args.teacher_model + '/result')
    teacher_embedding = torch.load('GNN_result/' + args.dataset + '/' + args.teacher_model + '/embedding')[node_type]
    teacher_similarity = torch.load('GNN_result/' + args.dataset + '/' + args.teacher_model + '/sim')

    # initialize model
    embedding_dim = teacher_embedding.shape[1]
    output_dim = num_class
    model = MLP(input_dim=input_size, hidden_dim=args.hidden_size, output_dim=output_dim,
                dropout_ratio=args.dropout_ratio, num_layers=args.num_layers, embedding_dim=embedding_dim)
    model = model.to(device)

    # loss functions

    logit_criterion = LogitLoss()
    gt_criterion = GtLoss()
    embedding_criterion = EmbeddingLoss()
    structure_criterion = StructureLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    # training process
    start_patience = patience = args.patience
    best_val_acc = 0
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        model.train()

        student_embedding, student_logit = model(X)
        student_similarity, student_sim_ind = get_similarity(metapath_data, emb=student_embedding[-1], sample=True,
                                                             seed=epoch)

        logit_loss = logit_criterion(student_logit, teacher_logit)
        gt_loss = gt_criterion(student_logit[train_mask], Y_train)
        embedding_loss = embedding_criterion(student_embedding, teacher_embedding)
        structure_loss = structure_criterion(student_similarity, teacher_similarity, student_sim_ind)

        loss = args.logit_weight * logit_loss + args.emb_weight * embedding_loss + args.struc_weight * structure_loss \
               + args.gt_weight * gt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = test()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            # print(f'gt: {gt_loss.item():.4f}, logit: {logit_loss.item():.4f}, emb: {embedding_loss.item():.4f}, '
            #       f'struc: {structure_loss.item():.4f}')

        # early stop
        if best_val_acc <= val_acc:
            patience = start_patience
            best_val_acc = val_acc
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

    # model evaluate
    with torch.no_grad():
        model.eval()
        _, predictions = model(X_test)
        acc = (predictions.argmax(dim=-1) == Y_test).sum() / test_mask.sum()
        f1_macro = get_f1_macro(labels=Y_test, predictions=predictions.argmax(dim=-1))
        f1_micro = get_f1_micro(labels=Y_test, predictions=predictions.argmax(dim=-1))

    return acc, f1_macro, f1_micro
