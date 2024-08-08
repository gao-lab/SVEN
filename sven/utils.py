# SVEN: https://github.com/gao-lab/SVEN

import numpy as np
import shutil
import ast


# function for replace annotations with outputs of sep models
def replace_anno(category, anno_path, output_prefix):
    # necessary files
    replace_list = np.loadtxt("./resources/" + category + "_select.filtered.txt", delimiter="\t", dtype=str)
    anno_ref_list = np.loadtxt("./resources/list_" + category + ".txt", delimiter="\t", dtype=str, skiprows=1)

    # path
    sep_anno_path = anno_path + category + "_" + output_prefix + "_tmp/"

    # load holistic annotations
    hol_anno = np.load(anno_path + "class_" + category + ".npy")

    output_file = anno_path + output_prefix + "_" + category + ".full.npy"

    # replace annotations
    for x in range(replace_list.shape[0]):
        feature_idx = replace_list[x][0]
        pos_idx = np.where(anno_ref_list[:, 0] == feature_idx)[0][0]
        sep_anno_file = sep_anno_path + category + "_" + feature_idx + ".npy"
        sep_anno = np.load(sep_anno_file)
        sep_anno = sep_anno.reshape((sep_anno.shape[0], -1))
        hol_anno[:, :, pos_idx] = sep_anno
    # save
    np.save(output_file, hol_anno)
    # remove sep annotations
    shutil.rmtree(sep_anno_path)


# function for transform annotations
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


def transform_feature(data_input, decay_list):
    # convert string to list
    decay_list = list(map(float, decay_list.split(',')))
 
    data_num = data_input.shape[0]
    feature_num = data_input.shape[2]
    #placeholder for transformed data
    sum_value = np.empty([data_num, len(decay_list)*2, feature_num], dtype=np.float32)
    for m in range(len(decay_list)):
        decay_c = decay_list[m]
        decay_array = trans_array(decay_c)
        decay_value = data_input*decay_array
        sum_value[:,2*m:2*m+2,:] = np.stack((np.sum(decay_value[:,:448,:], axis=1),np.sum(decay_value[:,448:,:], axis=1)),axis=1)
    return sum_value


# function for filter feature according to cutoff
def get_performance_filter_index(filter_cutoff = 0.5, performance_raw_file = "./resources/performance_filter_file.txt"):
    performance_file = np.loadtxt(performance_raw_file, delimiter = "\t", dtype = str, skiprows = 1)
    performance_list = performance_file[:,1].astype(float)
    filter_index = np.where(performance_list > filter_cutoff)[0]
    return filter_index


# function for generate model annotations, for full mode
def generate_model_anno_full(eid, anno_whole, anno_utr, model_path):
    model_feature_selected_dir = model_path + "model_feature_index/"
    model_feature_selected_list = np.loadtxt(model_feature_selected_dir + "filter_index_" + str(eid) + ".txt", delimiter="\t", dtype=int).tolist()
    anno_model = anno_whole[:, model_feature_selected_list]
    # merge features
    anno_model = np.hstack((anno_model, anno_utr))
    return anno_model


# function for get utr features
def get_utr_features(work_dir):
    bed_info = np.loadtxt(work_dir + "temp.bed", delimiter = "\t", dtype = str)
    utr_features = np.loadtxt("./resources/utr_features.txt", delimiter = "\t", dtype = str)
    utr_out = []
    for i in range(bed_info.shape[0]):
        gene_name = bed_info[i][3]
        gene_index = np.where(utr_features[:, 0] == gene_name)[0][0]
        utr_out.append(utr_features[gene_index, 1:].astype(float))
    utr_out = np.array(utr_out)
    return utr_out

# function for get length of chromosomes
def generate_chr_size_dict():
    chr_size = np.loadtxt("./resources/chromosome_size.GRCh38.txt", delimiter="\t", dtype=str)
    chr_size_dict = {}
    for m in range(chr_size.shape[0]):
        chr_size_dict[chr_size[m][0]] = int(chr_size[m][1])
    return chr_size_dict

# function for get tss of genes
def generate_chr_gene_dict(ignore_rRNA):
    chr_list = ["chr" + str(x) for x in range(1, 23)] + ["chrX", "chrY"]
    tss_gene_info = np.loadtxt("./resources/tss_gene_list.txt", dtype=str, delimiter='\t', skiprows=1)
    chr_gene_dict = {}
    for m in range(len(chr_list)):
        chr_info = chr_list[m]
        if ignore_rRNA == "true":
            chr_gene_pos = np.where((tss_gene_info[:,0] == chr_info) & (tss_gene_info[:,4] != "rRNA"))[0].tolist()
        else:
            chr_gene_pos = np.where(tss_gene_info[:,0] == chr_info)[0].tolist()
        chr_gene_dict[chr_info] = tss_gene_info[chr_gene_pos]
    return chr_gene_dict

# function for pad sequence
def pad_seq(err_info, seq_record, target_len):
    if err_info == "left":
        seq_record = "N"*(target_len - len(seq_record)) + seq_record
    elif err_info == "right":
        seq_record = seq_record + "N"*(target_len - len(seq_record))
    else:
        raise ValueError("Error: wrong error info.")
    return seq_record

# function for get relative center of sv
def get_relative_center(sv_rel_pos, tss_rel_pos, sv_length, sv_type):
    sv_rel_end = sv_rel_pos + sv_length
    spanning_tss = False
    if sv_type == "DEL":
        if sv_rel_end < tss_rel_pos:
            seq_rel_center = tss_rel_pos - sv_length
        elif sv_rel_pos >= tss_rel_pos:
            seq_rel_center = tss_rel_pos
        elif sv_rel_pos < tss_rel_pos and sv_rel_end > tss_rel_pos:
            relative_mod_len = tss_rel_pos - sv_rel_pos
            seq_rel_center = tss_rel_pos - relative_mod_len
            spanning_tss = True
        else:
            raise ValueError("Error: wrong relative position.")
    elif sv_type == "INS":
        if sv_rel_pos < tss_rel_pos:
            seq_rel_center = tss_rel_pos + sv_length
        else:
            seq_rel_center = tss_rel_pos
    else:
        raise ValueError("Error: wrong SV type.")
    return seq_rel_center

# function for calculate log2 fold change
def cal_fold_change(work_dir):
    # files
    ref_exp_file = work_dir + "/output/" + "exp_ref.txt"
    alt_exp_file = work_dir + "/output/" + "exp_alt.txt"
    output_file = work_dir + "/output/" + "exp_log2fc.txt"
    
    # load data
    ref_exp = np.loadtxt(ref_exp_file, delimiter="\t", dtype=str)
    alt_exp = np.loadtxt(alt_exp_file, delimiter="\t", dtype=str)
    
    # reshape for only one cell line
    ref_exp = ref_exp.reshape((ref_exp.shape[0], -1))
    alt_exp = alt_exp.reshape((alt_exp.shape[0], -1))
    output_title = ref_exp[0].reshape(1, -1)
    
    # output placeholder
    log2fc_out = np.empty((ref_exp.shape[0] - 1, ref_exp.shape[1]), dtype=float)
    # calculate log2 fold change
    for x in range(ref_exp.shape[1]):
        ref_exp_tmp = np.power(10, ref_exp[1:, x].astype(float))
        alt_exp_tmp = np.power(10, alt_exp[1:, x].astype(float))
        log2fc = np.log2(alt_exp_tmp / ref_exp_tmp)
        log2fc_out[:, x] = log2fc
    # save to out
    log2fc_out = np.vstack((output_title, log2fc_out))
    np.savetxt(output_file, log2fc_out, delimiter="\t", fmt="%s")
    