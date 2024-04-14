import argparse
import numpy as np
import xgboost as xgb

parser = argparse.ArgumentParser(description='Predict expression')
parser.add_argument('output_file', type = str, help = 'Output file')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode')
parser.add_argument('--verbose', type = bool, default = True, help = 'Verbose mode, only work when target_idx is None')
parser.add_argument('--target_idx', type = int, default = None, help = 'Target Model Index')


args = parser.parse_args()


#filter feature according to cutoff
def get_performance_filter_index():
    performance_file = np.loadtxt("./resources/performance_filter_file.txt", delimiter = "\t", dtype = str, skiprows = 1)
    filter_cutoff=0.5
    performance_list=performance_file[:,1].astype(float)
    filter_index=np.where(performance_list>filter_cutoff)[0]
    return filter_index

def get_utr_features():
    bed_info = np.loadtxt(args.work_dir + "temp.bed", delimiter = "\t", dtype = str)
    utr_features = np.loadtxt("./resources/utr_features.txt", delimiter = "\t", dtype = str)
    utr_out = []
    for i in range(bed_info.shape[0]):
        gene_name = bed_info[i][6]
        gene_index = np.where(utr_features[:,0] == gene_name)[0][0]
        utr_out.append(utr_features[gene_index,1:17].astype(float))
    utr_out = np.array(utr_out)
    return utr_out

def generate_model_anno(eid, anno_whole, anno_utr):
    model_feature_idx=np.loadtxt(model_feature_idx_dir+"filter_index_"+str(eid)+".txt",delimiter="\t",dtype=int)
    model_feature_idx.tolist()
    anno_model=anno_whole[:,model_feature_idx]
    '''merge features'''
    anno_model=np.hstack((anno_model,anno_utr))
    return anno_model

def _model_predict_fast(eid, model_anno):
    model_param = model_path + "xgb_" + str(eid) + ".json"
    if len(model_anno.shape) == 1:
        model_anno = model_anno.reshape(1, -1)
    xgb_model=xgb.XGBRegressor()
    xgb_model.load_model(model_param)
    exp_tmp = xgb_model.predict(model_anno)
    return exp_tmp

def _model_predict(eid, anno_whole, anno_utr):
    model_param = model_path + "xgb_" + str(eid) + ".json"
    model_anno = generate_model_anno(eid, anno_whole, anno_utr)
    if len(model_anno.shape) == 1:
        model_anno = model_anno.reshape(1, -1)
    xgb_model=xgb.XGBRegressor()
    xgb_model.load_model(model_param)
    exp_tmp = xgb_model.predict(model_anno)
    return exp_tmp



if __name__ == "__main__":
    if args.mode == "fast":
        model_path = "./models/fast_mode/"
    
    print("Loading annotations...")
    anno_acc = np.load(args.work_dir + "annotations/transformed/class_acc_transformed.npy")
    anno_his = np.load(args.work_dir + "annotations/transformed/class_his_transformed.npy")
    anno_tf = np.load(args.work_dir + "annotations/transformed/class_tf_transformed.npy")
    anno_utr = get_utr_features()
    print("Merge features...")
    filter_index = get_performance_filter_index()
    anno_whole = np.concatenate((anno_acc, anno_his, anno_tf), axis = -1)
    anno_whole = anno_whole[:,:,filter_index].reshape(anno_whole.shape[0], -1)
    
    if args.mode == "fast":
        anno_model=np.hstack((anno_whole,anno_utr))
    
    if args.target_idx is None:
        print("Predict expression with all models...")
        total = 218
        pred_exp = np.empty((anno_whole.shape[0], total), dtype = np.float32)   
        for eid in range(total):
            if args.mode == "fast":
                exp_tmp = _model_predict_fast(eid, anno_model)
            pred_exp[:, eid] = exp_tmp
            # print progress bar
            if args.verbose:
                progress = (eid + 1) / total
                print('\r[{0}] {1}%'.format('#'*(int(progress*50)), int(progress*100)), end='')
        np.savetxt(args.output_file, pred_exp, delimiter = "\t")
        print("\n")
    else:
        print("Predict expression with model %d..." % args.target_idx)
        #check args.target_idx is int, between 0 and 217
        if not isinstance(args.target_idx, int):
            raise ValueError("target_idx must be an integer.")
        if args.target_idx < 0 or args.target_idx > 217:
            raise ValueError("target_idx must be an integer between 0 and 217.")
        if args.mode == "fast":
            exp_tmp = _model_predict_fast(args.target_idx, anno_model)
        if len(exp_tmp.shape) == 1:
            exp_tmp = exp_tmp.reshape(-1, 1)
        np.savetxt(args.output_file, exp_tmp, delimiter = "\t")      
    print("Success: expression prediction.")