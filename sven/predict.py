# SVEN: https://github.com/gao-lab/SVEN

import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import xgboost as xgb
from sven.models import acc_hol_model, tf_hol_model, his_hol_model, sep_f1_model, sep_f2_model
from sven.utils import generate_model_anno_full



# function for holistic prediction
def run_hol_prediction(data_x, category, out_dir, output_prefix, seq_len, model_path, batch_size, resume_predict, gpu_state):
    model_params_file = model_path + "/class_oriented_" + category + ".h5"
    pred_file = out_dir + output_prefix + "_" + category + ".npy"
    
    if resume_predict == "true":
        if os.path.exists(pred_file):
            print("##### Prediction file exists. Skip prediction. #####")
            return
    
    if category == "acc":
        model = acc_hol_model(seq_len)
        feature_num = 684
    elif category == "tf":
        model = tf_hol_model(seq_len)
        feature_num = 2540
    elif category == "his":
        model = his_hol_model(seq_len)
        feature_num = 1976
    else:
        raise ValueError("Category not found.")
    
    # load weights
    model.load_weights(model_params_file)

    if gpu_state == "-1":
        y_pred = model.predict(data_x, batch_size=batch_size, verbose=0)
    else:
        y_pred = np.empty([data_x.shape[0], 896, feature_num], dtype=np.float32)
        for y in range(data_x.shape[0]):
            predictions = model(data_x[y].reshape((1, seq_len, 4)))
            y_pred[y] = predictions
    if category == "tf":
        y_pred = y_pred[:, :, 684:]
    # save
    np.save(pred_file, y_pred)


# function for separate prediction
def run_sep_prediction(data_x, category, out_dir, output_prefix, seq_len, model_path, batch_size, resume_predict, gpu_state):
    # create tmp output directory
    out_dir = out_dir + category + "_" + output_prefix + "_tmp/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    model_params_path = model_path + "/" + category + "/"
    model_params_list = np.loadtxt("./resources/" + category + "_select.filtered.txt", delimiter="\t", dtype=str)
    model_num = model_params_list.shape[0]

    # start prediction
    for x in range(model_num):
        model_select_id = model_params_list[x, 0]
        model_select_type = model_params_list[x, 1]
        model_params_file = model_params_path + model_select_id + ".h5"
        pred_file = out_dir + category + "_" + model_select_id + ".npy"
        
        if resume_predict == "true":
            if os.path.exists(pred_file):
                continue

        # build model
        if model_select_type == "F1":
            model = sep_f1_model(seq_len)
        elif model_select_type == "F2":
            model = sep_f2_model(seq_len)
        else:
            raise ValueError("Model type not found. Please check the model list: F1 or F2.")
        # load weights
        model.load_weights(model_params_file)
        # predict
        if gpu_state == "-1":
            y_pred = model.predict(data_x, batch_size=batch_size, verbose=0)
        else:
            y_pred = np.empty([data_x.shape[0], 896, 1], dtype=np.float32)
            for y in range(data_x.shape[0]):
                predictions = model(data_x[y].reshape((1, seq_len, 4)))
                y_pred[y] = predictions
        # save
        np.save(pred_file, y_pred)
        # clear session
        keras.backend.clear_session()


# xgb model prediction, fast mode
def model_predict_fast(eid, model_anno, model_path):
    # model param
    model_param = model_path + "xgb_" + str(eid) + ".json"
    # for only one sample
    if len(model_anno.shape) == 1:
        model_anno = model_anno.reshape(1, -1)
    # init model
    xgb_model = xgb.XGBRegressor()
    # load model
    xgb_model.load_model(model_param)
    # predict
    exp_tmp = xgb_model.predict(model_anno)
    return exp_tmp


# xgb model prediction, full mode
def model_predict_full(eid, anno_whole, anno_utr, model_path):
    # model params
    model_param = model_path + "xgb_" + str(eid) + ".json"
    # generate model annotations
    model_anno = generate_model_anno_full(eid, anno_whole, anno_utr, model_path)
    # for one sample
    if len(model_anno.shape) == 1:
        model_anno = model_anno.reshape(1, -1)
    # init model
    xgb_model = xgb.XGBRegressor()
    # load model
    xgb_model.load_model(model_param)
    # predict
    exp_tmp = xgb_model.predict(model_anno)
    return exp_tmp