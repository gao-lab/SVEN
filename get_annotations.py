# SVEN: https://github.com/gao-lab/SVEN

import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append("./")
import h5py
import numpy as np
import tensorflow as tf
from sven.predict import run_hol_prediction, run_sep_prediction
from sven.utils import replace_anno



parser = argparse.ArgumentParser(description='Get functional annotations.')
parser.add_argument('--input_file', type = str, default = "./work_dir/temp.h5", help = 'Input file in hdf5 format, default is ./work_dir/temp.h5.')
parser.add_argument('--type', type = str, default = "tss", help = 'Type of input file: tss, sv or snv. Default is tss.')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/.')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode, default is fast. Use fast or full.')
parser.add_argument('--seq_len', type = int, default = 131072, help = 'Input sequence length, default is 131_072. No need to change.')
parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size, default is 16.')
parser.add_argument('--gpu', type = str, default = "-1", help = 'GPU id, default is -1 (CPU)')
parser.add_argument('--model_path', type = str, default = "./model_params/", help = 'Model parameters path, default is ./model_params/')
parser.add_argument('--resume', type = str, default = "true", help = 'Conintue prediction or start new prediction, default is true.')
args = parser.parse_args()

# select GPU or CPU
if args.gpu != "-1":
    # check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    gpu_num = len(gpus)
    if gpu_num == 0:
        print("##### No GPU available. Use CPUs instead. #####")
    else:
        print("##### There are %s GPU(s) available in the system. #####" % gpu_num)
        # select GPU
        if int(args.gpu) < gpu_num:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            print("##### GPU %s is selected for prediction. #####" % args.gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("##### GPU %s is not available. Use GPU 0 instead. #####" % args.gpu)
else:
    print("##### Use CPUs for prediction. #####")
    # force to use CPUs in a system with GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def run_prediction(data_input, out_dir, output_prefix):
    run_hol_prediction(data_input, "acc", out_dir, output_prefix, args.seq_len, args.model_path, args.batch_size, args.resume, args.gpu)
    run_hol_prediction(data_input, "tf", out_dir, output_prefix, args.seq_len, args.model_path, args.batch_size, args.resume, args.gpu)
    run_hol_prediction(data_input, "his", out_dir, output_prefix, args.seq_len, args.model_path, args.batch_size, args.resume, args.gpu)
    if args.mode == "full":
        run_sep_prediction(data_input, "tf", out_dir, output_prefix, args.seq_len, args.model_path, args.batch_size, args.resume, args.gpu)
        run_sep_prediction(data_input, "his", out_dir, output_prefix, args.seq_len, args.model_path, args.batch_size, args.resume, args.gpu)
        # replace annotations
        print("##### Replacing annotations... #####")
        replace_anno("tf", out_dir, output_prefix)
        replace_anno("his", out_dir, output_prefix)

if __name__ == "__main__":
    # get absolute paths
    args.model_path = os.path.abspath(args.model_path)
    args.work_dir = os.path.abspath(args.work_dir)
    
    # Create output directory
    out_dir = args.work_dir + "/annotations/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Load data
    print("##### Loading data... #####")
    if args.type == "tss":
        data_x  = h5py.File(args.input_file, 'r')['seq'][:]
    elif args.type == "sv" or args.type == "snv":
        with h5py.File(args.input_file, 'r') as h5file:
            data_ref = h5file['ref_seq'][:]
            data_alt = h5file['alt_seq'][:]
    else:
        raise ValueError("Invalid type. Please use 'tss', 'sv' or 'snv'.")
    
    # Run prediction
    if args.mode == "fast":
        print("##### Predicting with fast mode... #####")
    elif args.mode == "full":
        print("##### Predicting with full mode... #####")
    else:
        raise ValueError("Invalid mode. Please use 'fast' or 'full'.")
    
    if args.type == "tss":
        run_prediction(data_x, out_dir, "tss")
    else:
        run_prediction(data_ref, out_dir, "ref")
        run_prediction(data_alt, out_dir, "alt")

    print("##### Success: get functional annotations. #####")
    