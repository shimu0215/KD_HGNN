from typing import List
from models import HAN

import torch
import torch.nn.functional as F

from data_prepare import load_data_metapath
from utils import evaluate_model, get_similarity


@torch.no_grad()
def test(model, data, node_type) -> List[float]:
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, node_type)[0].argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[node_type][split]
        acc = (pred[mask] == data[node_type].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


def train_HAN(args, data):
    node_type = args.node
    num_class = data[node_type].y.unique().size(0)

    model = HAN(in_channels=-1, out_channels=num_class, data=data, hidden_channels=args.teacher_hidden, num_layers=args.teacher_num_layer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out, embedding = model(data.x_dict, data.edge_index_dict, args.node)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train() -> float:
        model.train()
        optimizer.zero_grad()
        out, embedding = model(data.x_dict, data.edge_index_dict, args.node)
        mask = data[node_type].train_mask
        loss = F.cross_entropy(out[mask], data[node_type].y[mask])
        loss.backward()
        optimizer.step()
        return float(loss)

    best_val_acc = 0
    start_patience = patience = args.teacher_patience
    epochs = args.teacher_epochs
    for epoch in range(1, epochs):

        loss = train()
        train_acc, val_acc, test_acc = test(model, data, node_type)

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

    path = './GNN_result/' + args.dataset + '/' + args.teacher_model + '/'

    torch.save({'model_state_dict': model.state_dict()}, path + args.teacher_model)

    with torch.no_grad():
        model.eval()
        predictions, embedding = model(data.x_dict, data.edge_index_dict, args.node)
        acc, f1_macro, f1_micro = evaluate_model(data, node_type, predictions)

    torch.save(predictions, path + 'result')
    torch.save(embedding, path + 'embedding')

    metapath_data, _ = load_data_metapath(args)
    teacher_similarity, _ = get_similarity(metapath_data, emb=embedding[node_type])
    torch.save(teacher_similarity, path + 'sim')

    return acc, f1_macro, f1_micro


def eval_HAN(args, data):

    num_class = data[args.node].y.unique().size(0)
    model = HAN(in_channels=-1, out_channels=num_class, data=data)

    path = './GNN_result/' + args.dataset + '/' + args.teacher_model + '/' + args.teacher_model
    record = torch.load(path)

    model.load_state_dict(record['model_state_dict'])
    predictions = model(data.x_dict, data.edge_index_dict, args.node)[0].argmax(dim=-1)

    acc, f1_macro, f1_micro = evaluate_model(data, args.node, predictions)

    return acc, f1_macro, f1_micro
