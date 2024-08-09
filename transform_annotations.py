# SVEN: https://github.com/gao-lab/SVEN

import argparse
import os
import numpy as np
from sven.utils import transform_feature

parser = argparse.ArgumentParser(description='Transform annotations')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode, default is fast. Use fast or full.')
parser.add_argument('--type', type = str, default = "tss", help = 'Type of input file: tss, sv or snv. Default is tss.')
parser.add_argument('--decay_list', type = str, default = "0.01, 0.02, 0.05, 0.10, 0.20", help = 'List of decay constants, default is [0.01, 0.02, 0.05, 0.10, 0.20].')

args = parser.parse_args()


def process_data(category, output_prefix):
    # Create output directory
    out_dir = args.work_dir + "annotations/transformed/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if args.mode == "fast":
        pred_file = args.work_dir + "annotations/" + output_prefix + "_" + category + ".npy"
    elif args.mode == "full":
        if category != "acc":
            pred_file = args.work_dir + "annotations/" + output_prefix + "_" + category + ".full.npy"
        else:
            pred_file = args.work_dir + "annotations/" + output_prefix + "_" + category + ".npy"
    else:
        raise ValueError("Mode not found. Please use 'fast' or 'full'.")
    
    # Load data
    raw_annotations = np.load(pred_file)
    # Transform annotations
    transformed_annotations = transform_feature(raw_annotations, args.decay_list)
    return transformed_annotations

def merge_anno(output_prefix):
    output_file = args.work_dir + "annotations/transformed/anno." + output_prefix + ".merged." + args.mode + ".npy"
    anno_acc = process_data("acc", output_prefix)
    anno_his = process_data("his", output_prefix)
    anno_tf = process_data("tf", output_prefix)
    anno_merged = np.concatenate((anno_acc, anno_his, anno_tf), axis = -1)
    #print(anno_merged.shape)
    # save
    np.save(output_file, anno_merged)

if __name__ == "__main__":
    if args.mode == "fast":
        print("##### Transforming annotations with fast mode... #####")
    elif args.mode == "full":
        print("##### Transforming annotations with full mode... #####")
    else:
        raise ValueError("Invalid mode. Please use 'fast' or 'full'.")
    
    if args.type == "tss":
        merge_anno(args.type)
    elif args.type == "sv" or args.type == "snv":
        merge_anno("ref")
        merge_anno("alt")
    else:
        raise ValueError("Invalid type. Please use 'tss', 'sv' or 'snv'.")
    
    print("##### Success: Annotations transformed. #####")