from argument import parse_opt
from HGNN.run_HAN import train_HAN, eval_HAN
from HGNN.run_HGT import train_HGT, eval_HGT
from HGNN.run_HGT_OGB import train_HGT_OGB
from HGNN.run_HAN_OGB import train_HAN_OGB
from data_prepare import load_data_metapath, load_data_HGT
from utils import set_seed
from mlp_label import run_mlp_label
from mlp_KD import run_mlp_KD
import numpy as np


def train_teacher_model(args, data):
    """
    train the teacher model
    :return: evaluate result of the teacher model
    """

    if args.teacher_model == 'HAN':
        res_teacher = train_HAN(args, data)
        return res_teacher

    if args.teacher_model == 'HGT':
        res_teacher = train_HGT(args, data)
        return res_teacher


def eval_model(args, data):
    """
    Evaluate the teacher model and print the result
    """
    if args.teacher_model == 'HAN':
        train_acc, val_acc, test_acc = eval_HAN(args, data)
        return train_acc, val_acc, test_acc
    elif args.teacher_model == 'HGT':
        train_acc, val_acc, test_acc = eval_HGT(args, data)
        return train_acc, val_acc, test_acc

def run(args):
    set_seed(args.random_seed)

    # load data and one hop neighbor
    if args.teacher_model == 'HGT':
        data, neighbor = load_data_HGT(args)
    else:
        data, neighbor = load_data_metapath(args)

    # train teacher model
    if args.retrain_teacher:
        res_teacher = train_teacher_model(args, data)
        print('Accuracy of', args.teacher_model, " : ", res_teacher)

    # train student model
    if args.train_student:
        f1_macro_list = []
        f1_micro_list = []
        for i in range(5):
            acc_mlp_KD, f1_macro_KD, f1_micro_KD = run_mlp_KD(args, data, neighbor)
            f1_micro_list.append(f1_micro_KD * 100)
            f1_macro_list.append(f1_macro_KD * 100)

        print(" Macro-F1: mean-", np.mean(f1_macro_list), " std- ", np.std(f1_macro_list))
        print(" Micro-F1: mean-", np.mean(f1_micro_list), " std- ", np.std(f1_micro_list))

    # train a mlp with ground truth in train set
    if args.compare_to_mlp:
        set_seed(args.random_seed)
        acc_mlp_mlp, f1_macro_mlp, f1_micro_mlp = run_mlp_label(args, data)
        print("MLP-Acc: ", acc_mlp_mlp.item(), " Macro-F1: ", f1_macro_mlp.item(), " Micro-F1: ", f1_micro_mlp.item())

    # evaluate teacher model
    if args.eval_teacher_model:
        set_seed(args.random_seed)
        acc_teacher, f1_macro_teacher, f1_micro_teacher = eval_model(args, data)
        print(f'Teacher-Acc: {acc_teacher:.4f}, Macro-F1: {f1_macro_teacher:.4f}, Micro-F1: {f1_micro_teacher:.4f}')


if __name__ == "__main__":

    # get parameters
    args = parse_opt()
    print(args)

    args.dataset = 'DBLP'
    args.teacher_model = 'HGT'

    args.split = True
    args.train_ratio = 0.4
    args.val_ratio = 0.4

    run(args)
