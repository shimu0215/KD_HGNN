from argument import parse_opt
from data_prepare import load_data_homo
from utils import set_seed
from models import GNN, MLP
from torch.optim import Adam
import torch
from utils import get_f1_macro, get_f1_micro
from loss_function import LogitLoss, GtLoss

def evaluate_model(data, pred):
    mask = data['test_mask']
    Y_test = data.y[mask]
    predictions = pred[mask].argmax(dim=-1)
    acc = (predictions == Y_test).sum() / mask.sum()
    f1_macro = get_f1_macro(labels=Y_test, predictions=predictions)
    f1_micro = get_f1_micro(labels=Y_test, predictions=predictions)

    return acc.item(), f1_macro, f1_micro

def run_GNN(args):

    data = load_data_homo(args)
    print(data)
    train_mask = data['train_mask']

    model = GNN(128, data.y.unique().size(0))

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    start_patience = 50
    best_val_acc = 0.0

    for epoch in range(100):
        model.train()

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = evaluate_model(data, out)
            print('Acc:', val_acc[0])

        if best_val_acc <= val_acc[0]:
            patience = start_patience
            best_val_acc = val_acc[0]
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        acc, macro, micro = evaluate_model(data, out)
        print('acc:', acc, ', f1-macro:', macro, ', f1-micro:', micro)

    path = './GNN_result/' + args.dataset + '/GNN/'
    torch.save(out, path + 'logit')

    return model, data

def run_GLNN(args, data):
    teacher_logit = torch.load('./GNN_result/' + args.dataset + '/GNN/logit')

    model = MLP(input_dim=data.x.shape[1], hidden_dim=args.hidden_size, output_dim=data.y.unique().size(0),
                dropout_ratio=args.dropout_ratio, num_layers=args.num_layers)

    train_mask = data['train_mask']

    logit_criterion = LogitLoss()
    gt_criterion = GtLoss()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(100):
        model.train()


        h_list, student_logit = model(data.x)

        logit_loss = logit_criterion(student_logit, teacher_logit)
        gt_loss = gt_criterion(student_logit[train_mask], data.y[train_mask])
        loss = args.logit_weight * logit_loss + args.gt_weight * gt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        start_patience = 50
        best_val_acc = 0.0

        model.eval()
        with torch.no_grad():
            val_acc = evaluate_model(data, student_logit)
            print('Acc:', val_acc[0])

        if best_val_acc <= val_acc[0]:
            patience = start_patience
            best_val_acc = val_acc[0]
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

    model.eval()
    with torch.no_grad():
        out = model(data.x)
        acc, macro, micro = evaluate_model(data, out[-1])
        print('acc:', acc, ', f1-macro:', macro, ', f1-micro:', micro)




args = parse_opt()
set_seed(args.random_seed)
print(args)

args.dataset = 'DBLP'
args.teacher_model = 'HGT'

args.split = False
gnn_model, data = run_GNN(args)
run_GLNN(args, data)


