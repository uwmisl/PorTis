# PorTis

This repository contains the code for the paper "PorTis: A biological phenotype classification dataset
for benchmarking nanopore signal-based protein analysis". It contains code for loading, analyzing, and
preprocessing the PorTis dataset as well as the tissue classification model training and inference code.

Authors: Cailin Winston, Marc Expòsit, Jeff Nivala

## Structure of the repository

* `/src` contains core functionality for loading/analyzing data and for running benchmarks
  * `constants.py`
  * `kmedoids_mod.py`
  * `knn_classifier.py`
  * `knn_load_data.py`
  * `metrics.py`
  * `models.py`
  * `signal_processing.py`
  * `utils_data.py`
  * `utils_models.py`
* `/notebooks` contains interactive notebooks for the empirical analysis and clustering of the dataset
  * `1.0.DataAnalysis.ipynb`
  * `2.0.DataPrepkNN.ipynb`
  * `2.1.DTWmatrixCalc.ipynb`
  * `3.0.ClustAllSignals.ipynb`
  * `4.0.kNNinform.ipynb`
  * `5.0.InformativeSignalClustering.ipynb`
* `/scripts` contains python scripts to regenerate test indices and to run the CNN classification benchmarks
  * `gen_test_indices.py`
  * `1d_cnn.py`
  * `2d_cnn.py`

## Installation and setup

First clone the repository:

```
git clone git@github.com:uwmisl/PorTis.git
```

Then, setup the Python environment with the packages/dependencies needed to run the code in this repository:

```
cd PorTis
conda env create -f env.yml
```

Then, activate the conda environment:

```
conda activate portis
```

Ensure that you have Jupyter notebook to run the notebooks in `/notebooks`.

## Usage

To run any of the notebooks, open them in Jupyter notebook and follow the instructions/comments in the cells.

To run the benchmarks for the signal event classification (Task 1) and "real-time" sample classification (Task 2), run either `1d_cnn.py` or `2d_cnn.py` from the `/scripts` directory.

Any combination of the flags `--to_train`, `--to_eval`, and `--to_sim` can be specified, as long as at least one is specified. Running with the `--to_train` flag will train a new model. Running with the `--to_eval` flag with evaluate the model on the specified test set; either `--to_train` must also be specified to train a new model to evaluate or a `--model_path` to an already trained model can be provided. Running with the `--to_sim` flag will run the "real-time" sample classification simulation.

```
python 1d_cnn.py [-h] [--base_dir BASE_DIR] [--train_test_split_id TRAIN_TEST_SPLIT_ID]
    [--length_thresh LENGTH_THRESH] [--to_train] [--to_eval] [--to_sim] [--batch_size BATCH_SIZE]
    [--lr LR] [--epochs EPOCHS] [--model_path MODEL_PATH]

python 2d_cnn.py [-h] [--base_dir BASE_DIR] [--train_test_split_id TRAIN_TEST_SPLIT_ID]
    [--length_thresh LENGTH_THRESH] [--to_train] [--to_eval] [--to_sim][--batch_size BATCH_SIZE]
    [--lr LR] [--epochs EPOCHS] [--rescale_dim RESCALE_DIM] [--stack_depth STACK_DEPTH]
    [--model_path MODEL_PATH]
```
