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
    if args.dataset == 'OGB':
        # if args.teacher_model == 'HAN':
        #     res_teacher = train_HAN_OGB(args, data)
        #     return res_teacher

        if args.teacher_model == 'HGT':
            res_teacher = train_HGT_OGB(args, data)
            return res_teacher

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
        for i in range(1):
            set_seed(args.random_seed)
            acc_mlp_KD, f1_macro_KD, f1_micro_KD = run_mlp_KD(args, data, neighbor)
            f1_micro_list.append(f1_micro_KD * 100)
            f1_macro_list.append(f1_macro_KD * 100)

        print(" Macro-F1: mean-", np.mean(f1_macro_list), " std- ", np.std(f1_macro_list))
        print(" Micro-F1: mean-", np.mean(f1_micro_list), " std- ", np.std(f1_micro_list))

    # train a mlp with ground truth in train set
    if args.compare_to_mlp:
        acc_mlp_mlp, f1_macro_mlp, f1_micro_mlp = run_mlp_label(args, data)
        print("MLP-Acc: ", acc_mlp_mlp.item(), " Macro-F1: ", f1_macro_mlp.item(), " Micro-F1: ", f1_micro_mlp.item())

    # evaluate teacher model
    if args.eval_teacher_model:
        acc, f1_macro, f1_micro = eval_model(args, data)
        print(f'Teacher-Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Micro-F1: {f1_micro:.4f}')


if __name__ == "__main__":

    # get parameters
    args = parse_opt()
    print(args)

    args.dataset = 'DBLP'

    args.split = False
    args.train_ratio = 0.4
    args.val_ratio = 0.4


    args.teacher_model = 'HAN'
    print(args.teacher_model)

    print('train teacher')
    args.retrain_teacher = True
    args.train_student = False
    run(args)

    print('train student')
    args.retrain_teacher = False
    args.train_student = True

    print('train KD')
    args.use_neighbor = False
    args.lr = 0.003
    args.num_layers = 3
    args.logit_weight = 5 * 1
    args.gt_weight = 2 * 1
    args.emb_weight = 0.5 * 1
    args.struc_weight = 100 * 1
    run(args)

    print('train KD-n')
    args.use_neighbor = True
    args.lr = 0.0005
    args.num_layers = 2
    args.logit_weight = 5 * 1
    args.gt_weight = 2 * 1
    args.emb_weight = 0.5 * 0.1
    args.struc_weight = 100 * 0.001
    run(args)


