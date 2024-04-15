import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # global settings
    parser.add_argument('--dataset', type=str, default='DBLP', help="select the dataset, IMDB / DBLP / AMiner")
    parser.add_argument('--node', type=str, default='movie', help="select the target node, movie / author")
    parser.add_argument('--random_seed', type=int, default=0, help=" ")

    # running process
    parser.add_argument('--retrain_teacher', type=bool, default=False, help="train teacher model and save the result")
    parser.add_argument('--eval_teacher_model', type=bool, default=False, help="evaluate the saved teacher model")
    parser.add_argument('--train_student', type=bool, default=False, help="train student model")
    parser.add_argument('--compare_to_mlp', type=bool, default=False, help="train basic mlp")

    # student model settings
    parser.add_argument('--hidden_size', type=int, default=128, help="hidden size for student")
    parser.add_argument('--num_layers', type=int, default=3, help="number of layers for student")
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help="dropout ratio for student")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs for student")
    parser.add_argument('--patience', type=int, default=50, help="patience for student")
    parser.add_argument('--use_neighbor', type=bool, default=False, help="use one hop neighbor as part of input")

    # optimizer settings for student
    parser.add_argument('--lr', type=float, default=0.003 * 1, help="lr for student")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight decay for student")

    # weights of loss functions
    parser.add_argument('--logit_weight', type=int, default=5 * 1, help="weight of loss on logit")
    parser.add_argument('--gt_weight', type=int, default=2 * 1, help="weight of loss on ground truth")
    parser.add_argument('--emb_weight', type=int, default=0.5 * 2, help="weight of loss on embedding")
    parser.add_argument('--struc_weight', type=int, default=100 * 0.1, help="weight of loss on structural")

    # teacher settings
    parser.add_argument('--teacher_model', type=str, default='HAN', help="select the teacher model, HAN-0.2-0.1 / HGT")
    parser.add_argument('--teacher_patience', type=int, default=100, help="patience for teacher")
    parser.add_argument('--teacher_epochs', type=int, default=300, help="epochs for teacher")
    parser.add_argument('--teacher_hidden', type=int, default=128, help="hidden size for teacher")
    parser.add_argument('--teacher_num_layer', type=int, default=1, help="hidden size for teacher")

    # split dataset
    parser.add_argument('--split', type=bool, default=False, help="re group the dataset")
    parser.add_argument('--train_ratio', type=int, default=0.2, help="ratio of training set")
    parser.add_argument('--val_ratio', type=int, default=0.2, help="ratio of validation set")

    args, unknowns = parser.parse_known_args()

    return args
