# SVEN: https://github.com/gao-lab/SVEN

import argparse
import os
import h5py
import numpy as np
from scipy import stats
from sven.train import enformer_predict, enformer_transform, train_xgb, train_elasticNet
from sven.data import load_data


parser = argparse.ArgumentParser(description = 'Customize your own SVEN model')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/')
parser.add_argument('--action', type = str, help = 'Action to take: enformer_predict')
parser.add_argument('--enformer_path', type = str, help = 'Path to the trained enformer model')
parser.add_argument('--batch_size', type = int, default = 4 , help = 'Batch size for enformer_predict, default is 4')
parser.add_argument('--decay_list', type = str, default = "0.01, 0.02, 0.05, 0.10, 0.20", help = 'List of decay constants, default is [0.01, 0.02, 0.05, 0.10, 0.20].')
parser.add_argument('--exp_id', type = int, default = 0, help = 'Expression data id, default is 0')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode, default is fast. Use fast, full or enformer.')
parser.add_argument('--cutoff', type = float, default = 0.5, help = 'Cutoff for performance filter, default is 0.5')
parser.add_argument('--utr_features', type = str, default = "true", help = 'Use UTR features, default is true')
parser.add_argument('--ignore_rRNA', type = str, default = "true", help = 'Ignore rRNA, default is true')
parser.add_argument('--test_chr', type = str, default = "chr8", help = 'Test chromosome, default is chr8')
parser.add_argument('--pseudo_count', type = float, default = 0.0001, help = 'Pseudo count for expression data, default is 0.0001')
parser.add_argument('--model_type', type = str, default = "xgb", help = 'Model type, default is xgb. Use xgb or elasticNet.')
parser.add_argument('--device', type = str, default = "cpu", help = 'Device to use. Use cpu or gpu, default is cpu. Only for xgb.')
parser.add_argument('--threads', type = int, default = 8, help = 'Number of threads, default is 8. Only for xgb.')
parser.add_argument('--seed', type = int, default = 233, help = 'Random seed, default is 233')
args = parser.parse_args()


def anno_predict():
    # load data
    seq_data = h5py.File(args.work_dir + "temp.h5", 'r')['seq'][:]
    # predict
    enformer_predict(args.enformer_path, args.work_dir, seq_data, args.batch_size)
    print("###### Success: Annotations predicted with Enformer. ######")

def anno_transform():
    # transform annotations
    enformer_transform(args.decay_list, args.work_dir)
    print("###### Success: (Enformer) Annotations transformed . ######")

def exp_train():
    # annotation file
    anno_file = args.work_dir + "annotations/transformed/anno.tss.merged." + args.mode + ".npy"
    # performance filter file
    if args.mode == "fast" or args.mode == "full":
        performance_filter_file = "./resources/performance_filter_file.txt"
    elif args.mode == "enformer":
        performance_filter_file = "./performance_filter_file_enformer.txt"
    else:
        raise ValueError("Mode not found. Please use 'fast', 'full' or 'enformer'.")
    # tss file
    gene_tss_file = "./resources/tss_gene_list.txt"
    # exp_file
    exp_file = "./resources/gene_exp.txt"

    # load data
    print("###### Loading data... ######")
    x_train, y_train, x_test, y_test = load_data(args.work_dir, gene_tss_file, anno_file, args.utr_features, exp_file, 
                args.ignore_rRNA, args.test_chr, args.cutoff, performance_filter_file, args.exp_id, args.pseudo_count)

    # train model
    print("###### Training model... ######")
    model_dir  = args.work_dir + "train/model/"
    result_dir = args.work_dir + "train/result/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if args.model_type == "xgb":
        print("###### Training XGBoost model... ######")
        y_pred = train_xgb(x_train, y_train, x_test, model_dir, args.threads, args.device, args.seed)
    elif args.model_type == "elasticNet":
        print("###### Training ElasticNet model... ######")
        y_pred = train_elasticNet(x_train, y_train, x_test, model_dir, args.seed)
    else:
        raise ValueError("Model type not found. Please use 'xgb' or 'elasticNet'.")
    # evaluate
    output = np.hstack((y_pred, y_test))
    np.savetxt(result_dir + "test_pred_result.txt", output, delimiter = "\t")
    spearman_cor = stats.spearmanr(y_pred, y_test)[0]
    print("##### Spearman correlation: %.4f #####" % spearman_cor)
    print("###### Success: Model trained and evaluated. ######")
    

if __name__ == '__main__':
    if args.action == "enformer_predict":
        anno_predict()
    elif args.action == "enformer_transform":
        anno_transform()
    elif args.action == "exp_train":
        exp_train()
    else:
        raise ValueError("Action not found. Please use 'enformer_predict', 'enformer_transform' or 'exp_train'.")