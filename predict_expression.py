# SVEN: https://github.com/gao-lab/SVEN

import os
import argparse
import numpy as np
from sven.utils import get_performance_filter_index, get_utr_features, cal_fold_change
from sven.predict import model_predict_fast, model_predict_full

parser = argparse.ArgumentParser(description='Predict expression')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode, default is fast. Use fast or full.')
parser.add_argument('--type', type = str, default = "tss", help = 'Type of input file: tss, sv or snv. Default is tss.')
parser.add_argument('--verbose', type = str, default = "true", help = 'Verbose mode, only work when target_idx is None, default is true.')
parser.add_argument('--target_idx', type = int, default = None, help = 'Target Model Index')


args = parser.parse_args()


def model_prediction(output_prefix):
    # model path
    model_path = "./models/" + args.mode + "_mode/"
    work_dir = os.path.abspath(args.work_dir) + "/"
    output_dir = work_dir + "/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "exp_" + output_prefix + ".txt"
    
    print("##### Loading annotations... #####")
    anno_whole = np.load(work_dir + "annotations/transformed/anno." + output_prefix + ".merged." + args.mode + ".npy")
    anno_utr = get_utr_features(work_dir)
    cell_line_info = np.loadtxt("./resources/cell_line_list.txt", delimiter = "\t", dtype = str, skiprows = 1)
        
    # filter features according to performance
    filter_index = get_performance_filter_index()
    anno_whole = anno_whole[:, :, filter_index].reshape(anno_whole.shape[0], -1)

    if args.mode == "fast":
        anno_model = np.hstack((anno_whole, anno_utr))
    
    if args.target_idx is None:
        print("##### Predict expression with all 365 models... #####")
        cell_line_title = cell_line_info[:, 1].reshape(1, -1)
        total = 365
        pred_exp = np.empty((anno_whole.shape[0], total), dtype = np.float32)   
        for eid in range(total):
            if args.mode == "fast":
                exp_tmp = model_predict_fast(eid, anno_model, model_path)
            elif args.mode == "full":
                exp_tmp = model_predict_full(eid, anno_whole, anno_utr, model_path)
            else:
                raise ValueError("Mode not found. Please use 'fast' or 'full'.")
            pred_exp[:, eid] = exp_tmp
            # print progress bar
            if args.verbose == "true":
                progress = (eid + 1) / total
                print('\r[{0}] {1}%'.format('#'*(int(progress*50)), int(progress*100)), end='')
        pred_exp = np.vstack((cell_line_title, pred_exp))
        np.savetxt(output_file, pred_exp, delimiter = "\t", fmt = "%s")
        print("\n")
    else:
        print("##### Predict expression with model %d... #####" % args.target_idx)
        # check args.target_idx is int, between 0 and 364
        if not isinstance(args.target_idx, int):
            raise ValueError("target_idx must be an integer.")
        if args.target_idx < 0 or args.target_idx > 364:
            raise ValueError("target_idx must be an integer between 0 and 217.")
        # cell line info
        cell_line_title = cell_line_info[args.target_idx, 1].reshape(1, 1)
        
        # predict
        if args.mode == "fast":
            exp_tmp = model_predict_fast(args.target_idx, anno_model, model_path)
        elif args.mode == "full":
            exp_tmp = model_predict_full(args.target_idx, anno_whole, anno_utr, model_path)
        else:
            raise ValueError("Mode not found. Please use 'fast' or 'full'.")
        
        exp_tmp = exp_tmp.reshape(-1, 1)
        exp_tmp = np.vstack((cell_line_title, exp_tmp))
        np.savetxt(output_file, exp_tmp, delimiter = "\t", fmt = "%s") 

if __name__ == "__main__":
    print("##### Predict with %s mode... #####" % args.mode)
    if args.type == "tss":
        model_prediction(args.type)
        print("##### Success: expression prediction. #####")
    elif args.type == "sv" or args.type == "snv":
        model_prediction("ref")
        model_prediction("alt")
        print("##### Success: expression prediction. #####")
        cal_fold_change(args.work_dir)
        print("##### Success: calculate log2 fold change. #####")
    else:
        raise ValueError("Invalid type. Please use 'tss', 'sv' or 'snv'.")

    