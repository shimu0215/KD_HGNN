import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_model', type=str, default='HAN', help="HAN / HGT")
    parser.add_argument('--teacher_patient', type=int, default=30, help=" ")
    parser.add_argument('--teacher_epochs', type=int, default=100, help=" ")
    parser.add_argument('--retrain_teacher', type=bool, default=False, help=" ")
    parser.add_argument('--random_seed', type=int, default=123, help=" ")
    parser.add_argument('--dataset', type=str, default='DBLP', help="IMDB / DBLP")
    parser.add_argument('--num_class', type=int, default=4, help="3 / 4")
    parser.add_argument('--node', type=str, default='author', help="movie / author")
    parser.add_argument('--train_ratio', type=int, default=0.1, help=" ")
    parser.add_argument('--val_ratio', type=int, default=0.1, help=" ")
    parser.add_argument('--train_student', type=bool, default=True, help=" ")
    parser.add_argument('--compare_to_mlp', type=bool, default=True, help=" ")
    parser.add_argument('--eval_teacher_model', type=bool, default=False, help=" ")

    args, unknowns = parser.parse_known_args()

    return args
