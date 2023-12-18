from argument import parse_opt
from HGNN.run_HAN import train_HAN, eval_HAN
from HGNN.run_HGT import train_HGT, eval_HGT
from data_prepare import load_data_metapath, load_data_HGT
from mlp_label import run_mlp_label
from mlp_KD import run_mlp_KD
from models import HAN, HGT
import numpy as np
import torch


def train_teacher_model(args, data):
    if args.teacher_model == 'HAN':
        acc_teacher = train_HAN(args, data)
        return acc_teacher

    if args.teacher_model == 'HGT':
        acc_teacher = train_HGT(args, data)
        return acc_teacher

def eval_model(args, data):

    if args.teacher_model == 'HAN':
        eval_HAN(args, data)
    elif args.teacher_model == 'HGT':
        eval_HGT(args, data)


if __name__ == "__main__":
    args = parse_opt()
    print(args)

    # set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.teacher_model == 'HGT':
        data = load_data_HGT(args)
    else:
        data = load_data_metapath(args)

    if args.retrain_teacher:
        acc_teacher = train_teacher_model(args, data)
        print('Accuracy of', args.teacher_model, " : ", acc_teacher)

    if args.train_student:
        accuracy_KD = run_mlp_KD(args, data)
        print("Accuracy of KD: ", accuracy_KD.item())

    if args.compare_to_mlp:
        accuracy_label = run_mlp_label(args, data)
        print('Accuracy of MLP: ', accuracy_label.item())

    if args.eval_teacher_model:
        eval_model(args, data)
