# SVEN: https://github.com/gao-lab/SVEN

import argparse
import os
import subprocess
import h5py
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from sven.data import onehot_code
from sven.utils import generate_chr_size_dict, generate_chr_gene_dict, pad_seq, get_relative_center


parser = argparse.ArgumentParser(description='Prepare data for prediction')
parser.add_argument('inputfile', type = str, help = 'Input TSS file')
parser.add_argument('--type', type = str, default = "tss", help = 'Type of input file: tss, sv or snv. Default is tss.')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory, default is ./work_dir/')
parser.add_argument('--bedtools_path', type = str, default = "bedtools", help = 'Path to bedtools, default is bedtools')
parser.add_argument('--seq_len', type = int, default = 131_072, help = 'Sequence length, default is 131_072. Do not change this value.')
parser.add_argument('--ignore_rRNA', type = str, default = "true", help = 'Ignore rRNA genes, default is true. Only work in type sv.')
args = parser.parse_args()


ref_genome = "./resources/hg38.fa"
half_len = int(args.seq_len / 2)
# generate chromosome size dictionary
chr_size_dict = generate_chr_size_dict()

def extract_tss_bed():
    input_file = np.loadtxt(args.inputfile, delimiter="\t", dtype = str, skiprows = 1)
    output = []
    for x in range(input_file.shape[0]):
        chr_info = input_file[x][0]
        tss_pos = int(input_file[x][1]) - 1 # 1-based to 0-based
        strand = input_file[x][2]
        gene_name = input_file[x][3]

        err = "ok"
        pos_start = tss_pos - half_len
        pos_end = tss_pos + half_len
        if pos_start < 0:
            pos_start = 0
            err = "left"
        if pos_end > chr_size_dict[chr_info]:
            pos_end = chr_size_dict[chr_info]
            err = "right"
        output.append([chr_info, pos_start, pos_end, gene_name, strand, err])
    output = np.array(output)
    np.savetxt(args.work_dir + "temp.bed", output, delimiter = "\t", fmt = "%s")
    print("##### Success: extract bed file. #####")

def extract_sv_bed():
    chr_gene_dict = generate_chr_gene_dict(args.ignore_rRNA)
    input_file = np.loadtxt(args.inputfile, delimiter="\t", dtype = str, skiprows = 1)
    ref_bed = [] # list for sv info
    sv_allele = [] # list for ref allele and alt allele

    for x in range(input_file.shape[0]):
        # get basic info
        chr_info = input_file[x][0]
        sv_start = int(input_file[x][1]) - 1 # 1-based to 0-based
        ref_allele = input_file[x][2]
        alt_allele = input_file[x][3]
        sv_info = input_file[x][4]
        
        # get other info
        sv_length = abs(len(ref_allele) - len(alt_allele)) # for insertion and deletion
        if len(ref_allele) > len(alt_allele):
            sv_type = "DEL"
        elif len(ref_allele) < len(alt_allele):
            sv_type = "INS"
        else:
            print("##### Warning: unsupported SV %s, skip. #####" % sv_info)
            continue
        sv_end = sv_start + sv_length
        
        # check if all bases of SV are in 131kb regions
        sv_pos_array = np.array([sv_start, sv_end]).reshape((1, 2))
        tss_pos_list = chr_gene_dict[chr_info][:, 1].astype(int).reshape((-1,1))
        distance_array = np.abs(sv_pos_array - tss_pos_list)
        sv_gene_pos = np.where((distance_array[:,0] < half_len) & (distance_array[:,1] < half_len))[0]
        sv_pair_num = sv_gene_pos.shape[0]
        if sv_pair_num == 0:
            print("##### Warning: SV %s is not in 131kb region of any gene, skip. #####" % sv_info)
            continue
        # get sv-gene pairs
        for y in range(sv_pair_num):
            # get paired gene info
            err = "ok"
            gene_tss_pos = int(chr_gene_dict[chr_info][sv_gene_pos[y], 1])
            gene_strand = chr_gene_dict[chr_info][sv_gene_pos[y], 2]
            gene_name = chr_gene_dict[chr_info][sv_gene_pos[y], 3]

            # get bed info
            ref_pos_start = gene_tss_pos - half_len - sv_length
            ref_pos_end = gene_tss_pos + half_len + sv_length
            
            # check range of position
            if ref_pos_start < 0:
                ref_pos_start = 0
                err = "left"
            if ref_pos_end > chr_size_dict[chr_info]:
                ref_pos_end = chr_size_dict[chr_info]
                err = "right"
            # append bed info
            ref_bed.append([chr_info, ref_pos_start, ref_pos_end, gene_name, gene_strand, err, 
                            sv_type, sv_start, sv_end, sv_length, gene_tss_pos, sv_info])
            sv_allele.append([ref_allele, alt_allele])
    ref_bed = np.array(ref_bed)
    sv_allele = np.array(sv_allele)
    # save files
    np.savetxt(args.work_dir + "temp.bed", ref_bed, delimiter = "\t", fmt = "%s")
    np.savetxt(args.work_dir + "temp_sv_allele.txt", sv_allele, delimiter = "\t", fmt = "%s")
    print("##### Success: extract bed file. #####")

def extract_snv_bed():
    input_file = np.loadtxt(args.inputfile, delimiter="\t", dtype = str, skiprows = 1)
    output = []
    for x in range(input_file.shape[0]):
        chr_info = input_file[x][0]
        snv_pos = int(input_file[x][1]) - 1
        ref_allele = input_file[x][2]
        alt_allele = input_file[x][3]
        snv_info = input_file[x][4]
        # confirm if variant is a SNV in required format
        if len(ref_allele) != 1 or len(alt_allele) != 1:
            print("##### Warning: unsupported SNV %s, skip. #####" % snv_info)
            continue
        err = "ok"
        ref_start = snv_pos - half_len
        ref_end = snv_pos + half_len
        if ref_start < 0:
            ref_start = 0
            err = "left"
        if ref_end > chr_size_dict[chr_info]:
            ref_end = chr_size_dict[chr_info]
            err = "right"
        output.append([chr_info, ref_start, ref_end, ref_allele, alt_allele, err])
    output = np.array(output)
    np.savetxt(args.work_dir + "temp.bed", output, delimiter = "\t", fmt = "%s")
    print("##### Success: extract bed file. #####")

def extract_seq():
    # extract sequences from bed file
    in_bed = args.work_dir + "temp.bed"
    out_fasta = args.work_dir + "temp.fa"
    # ignore strand information here
    cmd = args.bedtools_path + ' getfasta -fi %s -bed %s -fo %s' % (ref_genome, in_bed, out_fasta)
    subprocess.call(cmd, shell=True)
    print("##### Success: extract sequences from bed file. #####")

def snv_to_h5():
    ref_bed = np.loadtxt(args.work_dir + "temp.bed", delimiter = "\t", dtype = str)
    in_fasta = args.work_dir + "temp.fa"
    ref_seq_list = []
    alt_seq_list = []

    sequence_info = open(in_fasta, 'r')
    for x, record in enumerate(SeqIO.parse(sequence_info, "fasta")):
        seq_record = str(record.seq).upper()
        err_info = ref_bed[x][5]
        # check length of sequence
        if len(seq_record) < args.seq_len:
            seq_record = pad_seq(err_info, seq_record, args.seq_len)
        #check allele
        ref_allele = ref_bed[x][3]
        alt_allele = ref_bed[x][4]
        #replace ref allele with alt allele
        ref_seq = seq_record[:half_len] + ref_allele + seq_record[half_len + 1:]
        alt_seq = seq_record[:half_len] + alt_allele + seq_record[half_len + 1:]
        # append
        ref_seq_list.append(ref_seq)
        alt_seq_list.append(alt_seq)
    sequence_info.close()
    os.remove(in_fasta)
    
    seq_num = len(ref_seq_list)
    print("##### Processing %d sequence pairs. #####" % seq_num)
    # convert to one-hot
    ref_seq_code = np.zeros((seq_num, args.seq_len, 4), dtype = np.int32)
    alt_seq_code = np.zeros((seq_num, args.seq_len, 4), dtype = np.int32)
    for j in range(seq_num):
        ref_sequence = ref_seq_list[j]
        alt_sequence = alt_seq_list[j]
        onehot_code(ref_sequence, ref_seq_code, j)
        onehot_code(alt_sequence, alt_seq_code, j)
    # save to h5
    with h5py.File(args.work_dir + "temp.h5", 'w') as hf:
        hf.create_dataset("ref_seq", data = ref_seq_code)
        hf.create_dataset("alt_seq", data = alt_seq_code)
    print("##### Success: sequence one-hot encoding. #####")

def sv_to_h5():
    ref_bed = np.loadtxt(args.work_dir + "temp.bed", delimiter = "\t", dtype = str)
    sv_allele = np.loadtxt(args.work_dir + "temp_sv_allele.txt", delimiter = "\t", dtype = str)
    in_fasta = args.work_dir + "temp.fa"
    ref_seq_list = []
    sv_seq_list = []

    sequence_info = open(in_fasta, 'r')
    for x, record in enumerate(SeqIO.parse(sequence_info, "fasta")):
        seq_record = str(record.seq).upper()
        err_info = ref_bed[x][5]
        sv_length = int(ref_bed[x][9])
        target_length = args.seq_len + 2*sv_length
        if len(seq_record) < target_length:
            seq_record = pad_seq(err_info, seq_record, target_length)

        # get sv info
        sv_start = int(ref_bed[x][7])
        gene_tss_pos = int(ref_bed[x][10])
        sv_type = ref_bed[x][6]
        strand = ref_bed[x][4]

        # calculate sv relative position to TSS
        sv_rel_pos = sv_start - (gene_tss_pos - half_len - sv_length)
        tss_rel_pos = half_len + sv_length

        # get ref and alt allele
        ref_allele = sv_allele[x, 0]
        alt_allele = sv_allele[x, 1]
       
        # replace ref allele with alt allele
        ref_allele_length = len(ref_allele)
        alt_allele_length = len(alt_allele)
        seq_record_ref = seq_record[:sv_rel_pos] + ref_allele + seq_record[sv_rel_pos + ref_allele_length:]
        seq_record_alt = seq_record[:sv_rel_pos] + alt_allele + seq_record[sv_rel_pos + ref_allele_length:]

        # append seq_record_ref
        seq_record_ref = seq_record_ref[tss_rel_pos - half_len : tss_rel_pos + half_len]
        if strand == "-":
            seq_record_ref = str(Seq(seq_record_ref).reverse_complement())
        ref_seq_list.append(seq_record_ref)
               
        # get new tss_rel_pos
        seq_rel_center = get_relative_center(sv_rel_pos, tss_rel_pos, sv_length, sv_type)
        
        # append seq_record_alt
        seq_record_alt = seq_record_alt[seq_rel_center - half_len : seq_rel_center + half_len]
        if strand == "-":
            seq_record_alt = str(Seq(seq_record_alt).reverse_complement())
        sv_seq_list.append(seq_record_alt)
    sequence_info.close()
    os.remove(in_fasta)
 
    seq_num = len(ref_seq_list)
    print("##### Processing %d sequence pairs. #####" % seq_num)
    # convert to one-hot
    ref_seq_code = np.zeros((seq_num, args.seq_len, 4), dtype = np.int32)
    sv_seq_code = np.zeros((seq_num, args.seq_len, 4), dtype = np.int32)
    for j in range(seq_num):
        ref_sequence = ref_seq_list[j]
        sv_sequence = sv_seq_list[j]
        onehot_code(ref_sequence, ref_seq_code, j)
        onehot_code(sv_sequence, sv_seq_code, j)
    # save to h5
    with h5py.File(args.work_dir + "temp.h5", 'w') as hf:
        hf.create_dataset("ref_seq", data = ref_seq_code)
        hf.create_dataset("alt_seq", data = sv_seq_code)
    print("##### Success: sequence one-hot encoding. #####")

def tss_to_h5():
    in_fasta = args.work_dir + "temp.fa"
    bed_info = np.loadtxt(args.work_dir + "temp.bed", delimiter = "\t", dtype = str)
    seq_list = []
    
    sequence_info = open(in_fasta, 'r')
    for x, record in enumerate(SeqIO.parse(sequence_info, "fasta")):
        seq_record = str(record.seq).upper()
        strand = bed_info[x][4]
        err_info = bed_info[x][5]
        # check length of sequence
        if len(seq_record) < args.seq_len:
            seq_record = pad_seq(err_info, seq_record, args.seq_len)
        # reverse complement
        if strand == "-":
            seq_record = str(Seq(seq_record).reverse_complement())
        # save sequence
        seq_list.append(seq_record)
    sequence_info.close()
    os.remove(in_fasta)

    seq_num = len(seq_list)
    print("##### Processing %d sequences. #####" % seq_num)
    # convert to one-hot
    seq_code = np.zeros((seq_num, args.seq_len, 4), dtype = np.int32)
    for j in range(seq_num):
        sequence = seq_list[j]
        onehot_code(sequence, seq_code, j)
    # save to h5
    with h5py.File(args.work_dir + "temp.h5", 'w') as hf:
        hf.create_dataset("seq", data = seq_code)
    print("##### Success: sequence one-hot encoding. #####")

if __name__ == "__main__":
    # create temp dir
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # check if bedtools is installed
    if subprocess.call("command -v bedtools", shell=True) != 0:
        raise Exception("bedtools is not installed. Please install bedtools first.")
    
    
    # extract bed
    if args.type == "tss":
        extract_tss_bed()
    elif args.type == "sv":
        extract_sv_bed()
    elif args.type == "snv":
        extract_snv_bed()
    else:
        raise Exception("Unsupported type of input file: %s. Please use tss, sv or snv." % args.type)
    
    # extract sequence
    extract_seq()
    
    # convert sequence to h5
    if args.type == "tss":
        tss_to_h5()
    elif args.type == "sv":
        sv_to_h5()
    else:
        snv_to_h5()
    