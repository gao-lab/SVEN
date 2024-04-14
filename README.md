# SVEN
This repository contains code for SVEN, a multi-modality sequence-oriented <i>in silico</i> model, for quantifying genetic variants' regulatory impacts in over 200 tissues and cell lines.

The SVEN framework is described in the following manuscript: Yu Wang, Nan Liang and Ge Gao, [Quantify genetic variants' regulatory potential via a hybrid sequence-oriented model](https://www.biorxiv.org/content/10.1101/2024.03.28.587115v1), bioRxiv (2024).


## Installation

#### Important Note: now we provide two modes for prediction: <i>Full mode</i> and <i>Fast mode</i>. For <i>Full mode</i>, you need download ~300G dependent model parameters files; while for <i>Fast mode</i>, you only need to download ~2G dependent model parameters files, with negligible precision loss. For reproducing results from our manuscript, please use <i>Full mode</i>.

Clone the repository then download and extract necessary resource files:
```bash
git clone https://github.com/gao-lab/SVEN.git
cd SVEN

# Download resources, ~1G
sh download_resources.sh
# Download model params for Fast mode, ~2G
sh download_model_params_fast.sh
# For Full mode, coming soon

# Extract dependent files
tar -xf resources.tar.gz
tar -xf model_params.tar.gz
```
Install python (3.8), install TensorFlow (v2.5.0) following instructions from https://www.tensorflow.org/ and bedtools from https://bedtools.readthedocs.io/. Use `pip install -r requirements.txt` to install the other dependencies.

## Usage

This is a quick guide for usage, the full guideline is coming soon.

```bash
# One-hot encoding
python prepare_data.py ./example/test.bed

# Get functional annotations with CPUs in fast mode
python get_annotations.py
# OR Get functional annotations with GPU 0 in fast mode
python get_annotations.py --gpu 0

# Transform annotations
python transform_annotations.py

# Predict gene expression
python predict_expression.py ./test.exp.predict.txt # with all models
python predict_expression.py ./test.exp.predict.txt --target_idx 3 # with target model
```

## Contact
Yu Wang: [wangy@mail.cbi.pku.edu.cn](mailto:wangy@mail.cbi.pku.edu.cn)
