# SVEN: https://github.com/gao-lab/SVEN
# Thanks to Calico for the Basenji codebase, which was used as a reference for this code.

import glob
from natsort import natsorted
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sven.utils import get_performance_filter_index, get_utr_features

# function for making dataset
def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

class SeqDataset:
    def __init__(self, data_dir, batch_size, seq_len=131_072, data_mode="test", tfr_pattern=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_mode = data_mode
        self.seq_len = seq_len
        self.tfr_pattern = tfr_pattern
        #make dataset
        self.make_dataset()
    
    #decode data
    def generate_parser(self):
        def parse_proto(example_protos):
            features = {
                'sequence':tf.io.FixedLenFeature([], tf.string),
                'target':tf.io.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.io.parse_single_example(example_protos, features=features)
            #decode sequence and reshape
            sequence = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
            sequence = tf.reshape(sequence, [self.seq_len, 4])
            sequence = tf.cast(sequence, tf.float32)
            #return data
            return sequence
        return parse_proto

    def make_dataset(self):
        #data path
        if self.tfr_pattern == None:
            self.tfr_pattern = "-0-*.tfr"
        tfr_path = self.data_dir + self.data_mode + self.tfr_pattern
        tfr_files = natsorted(glob.glob(tfr_path))

        dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        if self.data_mode == 'train':
            dataset = dataset.repeat()
            dataset = dataset.interleave(file_to_records, cycle_length=6, num_parallel_calls=tf.data.AUTOTUNE)
            #shuffle
            dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=True)
        else:
            dataset = dataset.flat_map(file_to_records)
        
        dataset = dataset.map(self.generate_parser())
        dataset = dataset.batch(self.batch_size)  
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        #hold
        self.dataset = dataset

# one-hot encoding
def onehot_code(seq, target, order):
    for i in range(len(seq)):
        if((seq[i] == "A")|(seq[i] == 'a')):
            target[order, i, 0]=1
        if((seq[i] == 'C')|(seq[i] == 'c')):
            target[order, i, 1]=1
        if((seq[i] == 'G')|(seq[i] == 'g')):
            target[order, i, 2]=1
        if((seq[i] == 'T')|(seq[i] == 't')):
            target[order, i, 3]=1


# function for load data
def load_data(work_dir, tss_file, anno_file, utr_features, exp_file, 
                ignore_rRNA, test_chr, cutoff, performance_filter_file, exp_id, pseudo_count):
    tss_info = np.loadtxt(tss_file, delimiter = "\t", dtype = str, skiprows = 1)
    performance_filter_index = get_performance_filter_index(cutoff, performance_filter_file)
    if ignore_rRNA == "true":
        target_index = np.where(tss_info[:,4] != "rRNA")[0]
    else:
        target_index = np.arange(tss_info.shape[0])
    
    # load main annotation
    anno_main = np.load(anno_file)[target_index][:, :, performance_filter_index]
    anno_main = anno_main.reshape(anno_main.shape[0], -1)
    
    # load UTR features
    if utr_features == "true":
        anno_utr = get_utr_features(work_dir)[target_index]
        anno_main = np.hstack((anno_main, anno_utr))
    
    # load expression data
    exp_raw_data = np.loadtxt(exp_file, delimiter = "\t", dtype = str, skiprows = 1)
    exp_raw_data = exp_raw_data.reshape((exp_raw_data.shape[0], -1)) # for only one tissue
    exp_raw_data = exp_raw_data[target_index, exp_id]
    # add pseudo count
    exp_data = np.log10(exp_raw_data.astype(float) + pseudo_count)
    exp_data = exp_data.reshape((exp_data.shape[0], 1))

    # split data
    chr_info = tss_info[target_index, 0]
    train_index = np.where(chr_info != test_chr)[0]
    test_index = np.where(chr_info == test_chr)[0]
    print("##### Training set number: %d #####" % len(train_index))
    print("##### Testing set number: %d #####" % len(test_index))

    x_train = anno_main[train_index]
    y_train = exp_data[train_index]
    x_test = anno_main[test_index]
    y_test = exp_data[test_index]
    return x_train, y_train, x_test, y_test
