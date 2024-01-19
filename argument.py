import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_model', type=str, default='HGT', help="HAN / HGT")
    parser.add_argument('--dataset', type=str, default='IMDB', help="IMDB / DBLP")
    parser.add_argument('--node', type=str, default='movie', help="movie / author")
    parser.add_argument('--num_class', type=int, default=3, help="3 / 4")

    parser.add_argument('--retrain_teacher', type=bool, default=False, help=" ")
    parser.add_argument('--eval_teacher_model', type=bool, default=False, help=" ")
    parser.add_argument('--train_student', type=bool, default=True, help=" ")
    parser.add_argument('--compare_to_mlp', type=bool, default=True, help=" ")

    parser.add_argument('--hidden_size', type=int, default=128, help=" ")
    parser.add_argument('--num_layers', type=int, default=3, help=" ")
    parser.add_argument('--dropout_ratio', type=float, default=0.4, help=" ")
    parser.add_argument('--epochs', type=int, default=1000, help=" ")
    parser.add_argument('--patient', type=int, default=80, help=" ")

    parser.add_argument('--lr', type=float, default=0.005, help=" ")
    parser.add_argument('--weight_decay', type=float, default=0.0, help=" ")

    parser.add_argument('--logit_weight', type=int, default=0.001, help=" 0.003 0.001 1 0.1")
    parser.add_argument('--gt_weight', type=int, default=0.001, help=" 0.003 0.001 0.001 0.01")
    parser.add_argument('--emb_weight', type=int, default=5, help="1 5 0.1 0.1")
    parser.add_argument('--struc_weight', type=int, default=2, help="1 2.25 1 1")

    parser.add_argument('--use_neighbor', type=bool, default=False, help=" ")

    parser.add_argument('--teacher_patient', type=int, default=30, help=" ")
    parser.add_argument('--teacher_epochs', type=int, default=100, help=" ")
    parser.add_argument('--teacher_hidden', type=int, default=128, help=" ")

    parser.add_argument('--random_seed', type=int, default=123, help=" ")

    parser.add_argument('--train_ratio', type=int, default=0.1, help=" ")
    parser.add_argument('--val_ratio', type=int, default=0.1, help=" ")

    args, unknowns = parser.parse_known_args()

    return args
