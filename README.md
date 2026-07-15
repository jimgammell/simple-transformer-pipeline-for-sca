# A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets

Anonymous artifact submission to OPTIMIST 2026.

This repository enables training, hyperparameter tuning, and evaluation of transformers for full-key physical side-channel attacks on uncropped power/EM traces.

## Project overview

## Installation

This code was tested using Python 3.11.15. Follow the instructions below to install the project and its dependencies:
1. Create and activate an environment for the project. For example, to do this with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):
```bash
micromamba create --name uncropped-transformers python=3.11
micromamba activate uncropped-transformers
```
2. Clone the project, then install it alongside its dependencies (reviewers: download from anonymous github instead of cloning):
```bash
git clone git@github.com:redacted-username/redacted-reponame
cd redacted-reponame
pip install -e .
```

### Downloading datasets

You may download some or all of the following datasets, and extract them to the project directory as follows:
- ASCADv1 (fixed key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key))
```bash
mkdir -p datasets/ascadv1_fixed
cd datasets/ascadv1_fixed
wget https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e -O ASCAD_data.zip
unzip ASCAD_data.zip
```
- ASCADv1 (variable key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key))
```bash
mkdir -p datasets/ascadv1_variable
cd datasets/ascadv1_variable
wget https://www.data.gouv.fr/api/1/datasets/r/3217dcc0-184f-402b-8914-e31cc120c51c -O atmega8515-raw-traces.h5
```
- CHES-CTF-2018 ([link](https://zenodo.org/records/3733418#.Yc2iq1ko9Pa))
```bash
mkdir -p datasets/ches_ctf_2018
cd datasets/ches_ctf_2018
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.1_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.2_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.3_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.4_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.5_1k_NK_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.6_1k_NK_upload.trs
```

### Downloading pretrained models

## Usage instructions

### Training models

### Evaluating pretrained models

### Tuning hyperparameters

## Additional results

### Full performance metrics

### Computational cost and scaling behavior

## Citation

If you use this code or our work, we would appreciate a citation:
```bibtex
(redacted for anonymous submission)
```

### Contact

If you have questions or want to point out bugs, don't hesitate to open an issue or contact me at `redacted`.