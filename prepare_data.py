import argparse
from Bio import SeqIO
from Bio.Seq import Seq
import h5py
import subprocess
import os

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument('inputbed', type = str, help = 'Input bed file')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory')

args = parser.parse_args()

ref_genome = "./resources/hg38.fa"
# create temp dir
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

def extract_seq(inputbed):
    # cp inputbed to work_dir
    cmd = "cp %s %s" % (inputbed, args.work_dir + "temp.bed")
    subprocess.call(cmd, shell=True)
    # extract sequences from bed file
    in_bed = args.work_dir + "temp.bed"
    out_fasta = args.work_dir + "temp.fa"
    cmd = 'bedtools getfasta -s -fi %s -bed %s -fo %s' % (ref_genome, in_bed, out_fasta)
    subprocess.call(cmd, shell=True)
    print("Success: extract sequences from bed file.")

def code(seq, target, order):
    for i in range(len(seq)):
        if((seq[i]=="A")|(seq[i]=='a')):
            target[order,i,0]=1
        if((seq[i]=='C')|(seq[i]=='c')):
            target[order,i,1]=1
        if((seq[i]=='G')|(seq[i]=='g')):
            target[order,i,2]=1
        if((seq[i]=='T')|(seq[i]=='t')):
            target[order,i,3]=1

def seq_to_h5():
    in_fasta = args.work_dir + "temp.fa"
    sequence_info = open(in_fasta, 'r')
    seq_list = []
    for record in SeqIO.parse(sequence_info, "fasta"):
        seq_record=str(record.seq)
        seq_list.append(seq_record)
    sequence_info.close()
    seq_num=len(seq_list)
    print("Processing %d sequences." % seq_num)
    # convert to one-hot
    seq_code = np.zeros((seq_num, 131_072, 4), dtype = np.int32)
    for j in range(seq_num):
        sequence = seq_list[j]
        code(sequence, seq_code, j)
    # save to h5
    with h5py.File(args.work_dir + "temp.h5", 'w') as hf:
        hf.create_dataset("seq", data = seq_code)
    print("Success: sequence one-hot encoding.")

if __name__ == "__main__":
    extract_seq(args.inputbed)
    seq_to_h5()
