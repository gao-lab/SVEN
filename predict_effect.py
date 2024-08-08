# SVEN: https://github.com/gao-lab/SVEN

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Calculate effects of small variants.')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode, default is fast. Use fast or full.')
parser.add_argument('--cutoff', type = float, default = 0.65758955, help = 'Cut off value for prediction, default is 0.65758955.')
args = parser.parse_args()

def cal_anno_difference(category):
    if args.mode == "full" and category != "acc":
        ref_pred_file = args.work_dir + "annotations/ref_" + category + ".full.npy"
        alt_pred_file = args.work_dir + "annotations/alt_" + category + ".full.npy"
    else:
        ref_pred_file = args.work_dir + "annotations/ref_" + category + ".npy"
        alt_pred_file = args.work_dir + "annotations/alt_" + category + ".npy"
    # load data
    ref_pred = np.load(ref_pred_file)
    alt_pred = np.load(alt_pred_file)
    # calculate difference
    anno_diff = np.sum(np.abs(alt_pred - ref_pred), axis=1)
    return anno_diff

def run_cal_diff(output_dir):
    print("##### Calculating annotation differences... #####")
    acc_diff = cal_anno_difference("acc")
    tf_diff = cal_anno_difference("tf")
    his_diff = cal_anno_difference("his")
    print("##### Merging differences... #####")
    diff_all = np.concatenate((acc_diff, tf_diff, his_diff), axis=1) # shape (N, 4516)
    diff_mean = np.mean(diff_all, axis=1).reshape((-1, 1)) # shape (N, 1)
    # assign labels based on cut off value
    labels = np.where(diff_mean > args.cutoff, 1, 0)
    labels = labels.reshape((-1, 1))
    title_info = np.array(["Effect", "Label"]).reshape((1, 2))
    # save results
    output = np.hstack((diff_mean, labels))
    output = np.vstack((title_info, output))
    np.savetxt(output_dir + "effect_snv.txt", output, delimiter = "\t", fmt = "%s")


if __name__ == "__main__":
    output_dir = args.work_dir + "/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    run_cal_diff(output_dir)
    print("##### Success: variant effects calculation. #####")
