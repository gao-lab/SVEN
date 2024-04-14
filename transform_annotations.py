import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Transform annotations')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode')

args = parser.parse_args()

decay_list=[0.01,0.02,0.05,0.10,0.20]

def trans_array(decay_constant):
    tmp_array = []
    for x in range(896):
        if x-447>0:
            tmp_array.append(np.exp(-decay_constant*(x-447)))
        else:
            tmp_array.append(np.exp(-decay_constant*(448-x)))
    tmp_array = np.array(tmp_array)
    tmp_array = tmp_array.reshape((896,1))
    return tmp_array

def transform_feature(data_input):
    data_num = data_input.shape[0]
    feature_num = data_input.shape[2]
    #placeholder for transformed data
    sum_value = np.empty([data_num,len(decay_list)*2,feature_num], dtype=np.float32)
    for m in range(len(decay_list)):
        decay_c = decay_list[m]
        decay_array = trans_array(decay_c)
        decay_value = data_input*decay_array
        sum_value[:,2*m:2*m+2,:] = np.stack((np.sum(decay_value[:,:448,:],axis=1),np.sum(decay_value[:,448:,:],axis=1)),axis=1)
    return sum_value

def process_data(category):
    if args.mode == "fast":
        pred_file = args.work_dir + "annotations/class_" + category + ".npy"
    else:
        raise ValueError("Full mode not supported yet.")
    out_dir = args.work_dir + "annotations/transformed/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    raw_annotations = np.load(pred_file)
    transformed_annotations = transform_feature(raw_annotations)
    np.save(out_dir + "class_" + category + "_transformed.npy", transformed_annotations)

if __name__ == "__main__":
    process_data("acc")
    process_data("his")
    process_data("tf")
    print("Success: Annotations transformed.")