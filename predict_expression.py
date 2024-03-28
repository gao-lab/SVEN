import argparse
import numpy as np
import xgboost as xgb

parser = argparse.ArgumentParser(description='Predict expression')
parser.add_argument('output_file', type = str, help = 'Output file')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory')


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
        utr_out.append(utr_features[gene_index,1:].astype(float))
    utr_out = np.array(utr_out)
    return utr_out

def generate_model_anno(eid, anno_whole, anno_utr):
    model_feature_idx=np.loadtxt(model_feature_idx_dir+"filter_index_"+str(eid)+".txt",delimiter="\t",dtype=int)
    model_feature_idx.tolist()
    anno_model=anno_whole[:,model_feature_idx]
    '''merge features'''
    anno_model=np.hstack((anno_model,anno_utr))
    return anno_model

if __name__ == "__main__":
    model_path = "./resources/models/fast_mode/"
    model_feature_idx_dir = model_path + "model_feature_index/"

    print("Loading annotations...")
    anno_acc = np.load(args.work_dir + "annotations/transformed/class_acc_transformed.npy")
    anno_his = np.load(args.work_dir + "annotations/transformed/class_his_transformed.npy")
    anno_tf = np.load(args.work_dir + "annotations/transformed/class_tf_transformed.npy")
    anno_utr = get_utr_features()
    print("Merge features...")
    filter_index = get_performance_filter_index()
    anno_whole = np.concatenate((anno_acc, anno_his, anno_tf), axis = -1)
    anno_whole = anno_whole[:,:,filter_index].reshape(anno_whole.shape[0], -1)
    
    print("Predict expression...")
    pred_exp = np.empty((anno_whole.shape[0], 218), dtype = np.float32)
    for eid in range(218):
        model_param = model_path + "xgb_" + str(eid) + ".model"
        model_anno = generate_model_anno(eid, anno_whole, anno_utr)
        xgb_model=xgb.XGBRegressor()
        xgb_model.load_model(model_param)
        exp_tmp = xgb_model.predict(model_anno)
        pred_exp[:, eid] = exp_tmp.reshape(exp_tmp.shape[0],1)
    np.savetxt(args.output_file, pred_exp, delimiter = "\t")
    print("Success: expression prediction.")