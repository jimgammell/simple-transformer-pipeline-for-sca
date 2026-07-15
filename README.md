# A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets

Anonymous artifact submission to OPTIMIST 2026.

This repository enables training, hyperparameter tuning, and evaluation of transformers for full-key physical side-channel attacks on uncropped power/EM traces.

## Project overview

### Minimum resource requirements

| Dataset | VRAM (required) | RAM (recommended) |
| --- | --- | --- |
| ASCADv1 (fixed key) | 8.78GB | 16GB |
| ASCADv1 (variable key) | 9.07GB | 96GB |
| CHES-CTF-2018 | 4.20GB | 32GB |

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

You may download some or all of the following datasets by running the commands below from the project directory:
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

### One-time dataset validation and preprocessing

The first time a dataset is initialized, our code will run checksum validation, copy traces to a binary file to enable faster dataloading, and cache the per-feature mean and variance on the profiling set. We recommend running the following commands from the project directory to complete this step once before training with multiprocessing. 
- ASCADv1 (fixed key); expected runtime: 30 sec.
```bash
python - <<'PY'
from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset
ASCADv1_NumpyDataset(root='./datasets/ascadv1_fixed', partition='profile', variable_key=False).get_trace_statistics(use_progress_bar=True)
ASCADv1_NumpyDataset(root='./datasets/ascadv1_fixed', partition='attack', variable_key=False)
PY
```
- ASCADv1 (variable key)
```bash
python - <<'PY'
from uncropped_transformers.datasets.ascadv1 import ASCADv1_NumpyDataset
ASCADv1_NumpyDataset(root='./datasets/ascadv1_variable', partition='profile', variable_key=True).get_trace_statistics(use_progress_bar=True)
ASCADv1_NumpyDataset(root='./datasets/ascadv1_variable', partition='attack', variable_key=True)
PY
```
- CHES-CTF-2018; expected runtime: 3 min.
```bash
python - <<'PY'
from uncropped_transformers.datasets.ches_ctf_2018 import CHESCTF2018_NumpyDataset
CHESCTF2018_NumpyDataset(root='./datasets/ches_ctf_2018', partition='profile').get_trace_statistics(use_progress_bar=True)
CHESCTF2018_NumpyDataset(root='./datasets/ches_ctf_2018', partition='attack')
PY
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