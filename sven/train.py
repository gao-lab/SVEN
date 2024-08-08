# SVEN: https://github.com/gao-lab/SVEN

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import joblib
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sven.utils import transform_feature

def enformer_predict(model_path, work_path, input_data, batch_size):        
    output_path = work_path + "annotations/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load model
    enformer = hub.load(model_path).model
    # placeholder for output
    output = np.empty((input_data.shape[0], 896, 4516), dtype=np.float32)
    for i in range(0, input_data.shape[0], batch_size):
        end = min(i+batch_size, input_data.shape[0])
        predictions = enformer.predict_on_batch(input_data[i:end])
        output[i:end] = predictions['human'][:, :, :4516]
    # save output
    output_file = output_path + "tss_all_enformer.npy"
    np.save(output_file, output)

def enformer_transform(decay_list, work_path):
    anno_path = work_path + "annotations/"
    output_path = work_path + "annotations/transformed/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = output_path + "anno.tss.merged.enformer.npy"
    # load data
    raw_annotations = np.load(anno_path + "tss_all_enformer.npy")
    # Transform annotations
    transformed_annotations = transform_feature(raw_annotations, decay_list)
    # save
    np.save(output_file, transformed_annotations)

def train_xgb(x_train, y_train, x_test, model_dir, threads, device, seed):
    assert device in ['cpu', 'gpu'], f"device must be either 'cpu' or 'gpu', but got {device}"
    model = xgb.XGBRegressor(n_estimators=1500,
        learning_rate = 0.05,
        objective = 'reg:squarederror',
        booster = 'dart',
        max_depth = 6,
        tree_method = 'hist',
        device = device,
        gamma = 0,
        min_child_weight = 3,
        reg_alpha = 30,
        reg_lambda = 60,
        random_state = seed,
        base_score = 1,
        n_jobs = threads)
    model.fit(x_train, y_train)
    # save model
    model.save_model(model_dir + "xgb_model.json")
    print("##### Predicting with test set... #####")
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1, 1)
    return y_pred

def train_elasticNet(x_train, y_train, x_test, model_dir, seed):
    model = ElasticNet(alpha = 1.0, l1_ratio = 0.5, max_iter = 5000, random_state = seed)
    model.fit(x_train, y_train)
    # save model
    joblib.dump(model, model_dir + "elasticNet_model.joblib")
    print("##### Predicting with test set... #####")
    test_pred = model.predict(x_test)
    test_pred = test_pred.reshape(-1, 1)
    return test_pred
    