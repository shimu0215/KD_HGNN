import torch
import torch.nn as nn
import torch.optim as optim

from models import MLP

from typing import Dict, List, Union


def run_mlp_label(args, data):

    node_type = args.node
    num_class = args.num_class

    train_mask = data[node_type].train_mask
    X_train = data.x_dict[node_type][train_mask]
    Y_train = data[node_type].y[train_mask]

    test_mask = data[node_type].test_mask
    X_test = data.x_dict[node_type][test_mask]
    Y_test = data[node_type].y[test_mask]

    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = num_class
    model = MLP(input_dim=input_size, hidden_dim=hidden_size, output_dim=output_size, dropout_ratio=args.dropout_ratio,
                num_layers=args.num_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    @torch.no_grad()
    def test() -> List[float]:

        model.eval()
        _, pred = model(data.x_dict[node_type])
        pred = pred.argmax(dim=-1)

        accs = []
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = data[node_type][split]
            acc = (pred[mask] == data[node_type].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
        return accs

    start_patience = patience = 50

    best_val_acc = 0
    num_epochs = 300

    model.train()
    for epoch in range(num_epochs):
        # Forward pass
        _, outputs = model(X_train)
        loss = criterion(outputs, Y_train)  # 0.4914

        # Backward pass and optimization
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

    # Testing the model
    with torch.no_grad():
        model.eval()
        _, predictions = model(X_test)
        acc = (predictions.argmax(dim=-1) == Y_test).sum() / test_mask.sum()

    return acc
