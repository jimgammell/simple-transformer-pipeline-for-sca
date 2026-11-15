# A Missing Baseline: Uncropped Full-Key Side-Channel Attacks with Boilerplate Transformers

This repository provides a reproducible PyTorch pipeline for training, hyperparameter tuning, and evaluation of transformers for full-key profiled physical side-channel attacks on uncropped power and electromagnetic traces.

We provide our code and final training configurations here. Pretrained models and detailed experimental results will accompany the public release.

## Overview

We use the standard pre-norm transformer backbone that is common in other domains, adapting only the input and output layers to the side-channel setting. Our code is built on PyTorch, PyTorch Lightning, and Optuna.

In `src/uncropped_transformers` we store pip-installable modular library code (e.g. neural nets, training methods, dataset wrappers, evaluation methods) designed to be imported into other projects. We store our project-specific code (e.g. entrypoints, PyTorch/Matplotlib configuration, directory creation) separately in the `experiments` directory.

## Installation

This code was tested using Python 3.11.15. Create and activate an environment for the project using [uv](https://docs.astral.sh/uv/getting-started/installation/), then install it alongside its dependencies:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Downloading datasets

Dataset paths are configured in `config/directories.yaml`.

You may download some or all of the following datasets by running the commands below from the project directory:
- ASCADv1 (fixed key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key))
```bash
(
    mkdir -p datasets/ascadv1_fixed
    cd datasets/ascadv1_fixed
    wget https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e -O ASCAD_data.zip
    unzip ASCAD_data.zip
)
```
- ASCADv1 (variable key) ([link](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key))
```bash
(
    mkdir -p datasets/ascadv1_variable
    cd datasets/ascadv1_variable
    wget https://www.data.gouv.fr/api/1/datasets/r/3217dcc0-184f-402b-8914-e31cc120c51c -O atmega8515-raw-traces.h5
)
```
- CHES-CTF-2018 ([link](https://zenodo.org/records/3733418))
```bash
(
    mkdir -p datasets/ches_ctf_2018
    cd datasets/ches_ctf_2018
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.1_10k_upload.trs
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.2_10k_upload.trs
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.3_10k_upload.trs
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.4_10k_upload.trs
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.5_1k_NK_upload.trs
    wget https://zenodo.org/records/3733418/files/PinataAcqTask2.6_1k_NK_upload.trs
)
```

- ASCADv2 ([link](https://github.com/ANSSI-FR/ASCAD)): download the eight raw trace files, `ascadv2-stm32-conso-raw-traces1.h5` through `ascadv2-stm32-conso-raw-traces8.h5`, to `datasets/ascadv2/`. We use the raw traces rather than the cropped dataset.
- eShards (non-shuffled) ([link](https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled)): download `Nucleo_AES_masked_non_shuffled.ets` to `datasets/eshards/`. We use the non-shuffled version, not `Nucleo_AES_masked_shuffled.ets`.

### One-time dataset validation and preprocessing

The first time a dataset is initialized, our code will run checksum validation, copy traces to a binary file to enable faster dataloading, and cache the per-feature mean and variance on the profiling set. We recommend running the following commands from the project directory to complete this step once before training/evaluation (otherwise dataloader multiprocessing might cause issues):
```bash
bash scripts/preprocess_datasets.sh ascadv1-fixed
bash scripts/preprocess_datasets.sh ascadv1-variable
bash scripts/preprocess_datasets.sh ches-ctf-2018
bash scripts/preprocess_datasets.sh ascadv2
```

eShards is loaded directly from the `.ets` file and does not need this preprocessing step.

An optional second argument specifies the dataset root, for example:
```bash
bash scripts/preprocess_datasets.sh ascadv2 /path/to/dataset
```

## Usage instructions

### Configuration

Trial configurations should be stored in `./config` as YAML files. We provide reference configurations for ASCADv1-fixed, ASCADv1-variable, CHES-CTF-2018, non-shuffled eShards, and ASCADv2, and you may use these as templates to create new configuration files with modified settings.

### Training from scratch

Models can be trained by running the following command, where `CONFIG_FILE` is replaced with the name of a config file in `./config` without the `.yaml` suffix, and `DEST` is a path to the directory where checkpoints and logs should be saved:
```bash
python experiments/train.py \
    --config-file CONFIG_FILE \
    --dest DEST
```
For example:
```bash
python experiments/train.py --config-file ascadv1_variable --dest ./outputs/ascadv1_variable/demo_run
python experiments/train.py --config-file ascadv2 --dest ./outputs/ascadv2/demo_run
python experiments/train.py --config-file eshards_non_shuffled --dest ./outputs/eshards_non_shuffled/demo_run
```
This command will save the following files to `DEST`:
- `best_{metric_name}.ckpt`: the checkpoint of the best model according to `{metric_name}` (validation rank by default).
- `latest.ckpt`: the model checkpoint after the last epoch before training finished or was interrupted.
- `metrics.csv`: logs of metrics recorded after every training epoch (e.g. train/val loss, rank).
- `config.yaml`: a config file encoding settings/info for reproducibility -- e.g. hyperparameters (after loading the yaml and applying command-line overrides), the Git commit hash, random seed.
- `hparams.yaml`: a config file used internally by PyTorch Lightning for saving/loading checkpoints.

Training and tuning use only the profiling partition and its held-out validation split. Attack evaluation is a separate step.

### Evaluating trained models on the attack set

The following command will compute, cache, and display a trained model's full-key and per-byte cross-entropy loss, accuracy, MTD, and correct-key rank on the test set, where `CKPT_PATH` is a path to a trained model checkpoint:
```bash
python experiments/attack.py \
    --model-ckpt-path CKPT_PATH
```
This command assumes the checkpoint is stored in an experiment directory produced by our training script, which contains a `config.yaml` file. It will save `attack_metrics.npz`, a cache of the performance metrics computed on the attack set, to this directory.

### Plotting training curves

The following command will visualize the logs of train/val loss, acc, rank over time, and cached attack performance metrics, where `RUN_DIR` is a path to an experiment directory produced by our training script:
```bash
python experiments/plot_performance.py \
    --run-dir RUN_DIR
```
This command only reads saved data, skips missing input files, and saves the following files to this directory:
- `training_curves.pdf`: plots of the train/val loss, rank, accuracy vs. training steps.
- `attack_performance.pdf`: a plot of correct-key rank vs. traces seen on the attack set, and a visualization of the per-byte MTD.

### Tuning hyperparameters

Hyperparameter tuning is done from the same entrypoint as training by adding additional command line arguments. Run the following command, where `OPTUNA_STUDY_PATH` specifies where the hyperparameter tuning database will be stored (recommended: `DEST/study.log`), and `OPTUNA_TOTAL_TRIALS` specifies the number of trials to run. The hyperparameters to be tuned and their search spaces are specified in the config file.
```bash
python experiments/train.py \
    --config-file CONFIG_FILE \
    --dest DEST \
    --optuna-study-path OPTUNA_STUDY_PATH \
    --optuna-total-trials OPTUNA_TOTAL_TRIALS
```