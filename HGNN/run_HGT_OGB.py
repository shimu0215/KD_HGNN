import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import HGT

from data_prepare import load_data_metapath, load_data_HGT
from utils import evaluate_model, get_similarity


def train_HGT_OGB(args, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = data['train']
    val_loader = data['val']
    test_loader = data['test']
    node_type = args.node
    num_class = train_loader.data[node_type].y.unique().size(0)

    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device, 'edge_index')
        return batch
    batch = init_params()

    model = HGT(hidden_channels=args.teacher_hidden, out_channels=num_class, num_heads=8, num_layers=args.teacher_num_layer, data=batch)
    model = model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out, _ = model(batch.x_dict, batch.edge_index_dict, node_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train():
        model.train()

        total_examples = total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict, node_type)[0][:batch_size]
            loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
            break

        return total_loss / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()

        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict, node_type)[0][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

        return total_correct / total_examples

    def get_result(loader):
        model.eval()

        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict, node_type)[0][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

        return total_correct / total_examples

    best_val_acc = 0
    patience = start_patience = args.teacher_patience
    epochs = args.teacher_epochs

    for epoch in range(1, epochs):
        loss = train()
        val_result = test(val_loader)

        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Val: {val_result:.4f}')

        if best_val_acc <= val_result:
            patience = start_patience
            best_val_acc = val_result
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break
    test_result = test(test_loader)

    path = './GNN_result/' + args.dataset + '/' + args.teacher_model + '/'

    torch.save({'model_state_dict': model.state_dict()}, path + args.teacher_model)

    whole_data, _ = load_data_HGT(args, True)
    with torch.no_grad():
        predictions, embedding = model(whole_data.x_dict, whole_data.edge_index_dict, node_type)

    torch.save(predictions, path + 'result')
    torch.save(embedding, path + 'embedding')

    metapath_data, _ = load_data_metapath(args)
    teacher_similarity, _ = get_similarity(metapath_data, emb=embedding[node_type])
    torch.save(teacher_similarity, path + 'sim')

    return test_result

def eval_HGT(args, data):

    node_type = args.node
    num_class = data[node_type].y.unique().size(0)
    model = HGT(hidden_channels=args.teacher_hidden, out_channels=num_class, num_heads=8, num_layers=1, data=data)

    path = './GNN_result/' + args.dataset + '/' + args.teacher_model + '/' + args.teacher_model
    record = torch.load(path)

    model.load_state_dict(record['model_state_dict'])
    predictions = model(data.x_dict, data.edge_index_dict, args.node)[0].argmax(dim=-1)

    acc, f1_macro, f1_micro = evaluate_model(data, args.node, predictions)

    return acc, f1_macro, f1_micro
