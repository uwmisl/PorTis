# PorTis

This repository contains the code for the paper "PorTis: A biological phenotype classification dataset
for benchmarking nanopore signal-based protein analysis". It contains code for loading, analyzing, and
preprocessing the PorTis dataset as well as the tissue classification model training and inference code.

Authors: Cailin Winston, Marc Expòsit, Jeff Nivala

## Structure of the repository

* `/src` contains core functionality for loading/analyzing data and for running benchmarks
  * `constants.py`: constant definitions to load and save data files
  * `kmedoids_mod.py`: modified version of `scikit-learn-extra`'s k-medoids function to cluster signals and return centroids
  * `knn_classifier.py`: functions required to train and test the kNN classifier
  * `knn_load_data.py`: functions to load and save signals, labels, and the distance matrix for the kNN classifier
  * `metrics.py`: functions to calculate metrics of the CNN models
  * `models.py`: model architecture of the CNN classifiers
  * `signal_processing.py`: functions used to process signals before use in the classifiers
  * `utils_data.py`: functions used to load data for the CNN classifiers
  * `utils_models.py`: helper functions to train the CNN classifiers
* `/notebooks` contains interactive notebooks for the empirical analysis and clustering of the dataset
  * `1.0.DataAnalysis.ipynb`: Exploratory data analysis of signal number and length by replicate and tissue type
  * `2.0.DataPrepkNN.ipynb`: Balancing data across replicates and tissues to use in kNN classification
  * `2.1.DTWmatrixCalc.ipynb`: Calculating the distance matrix between signals using Dynamic Time Warping (DTW)
  * `3.0.ClustAllSignals.ipynb`: t-SNE representation of all signals
  * `4.0.kNNinform.ipynb`: kNN classification and identification of signals unique to each tissue
  * `5.0.InformativeSignalClustering.ipynb`: Analysis and clustering of informative signals
* `/scripts` contains python scripts to regenerate test indices and to run the CNN classification benchmarks
  * `gen_test_indices.py`: generate the indices to divide the data between training and testing sets for use in the CNN classifier
  * `1d_cnn.py`: training and inference of the 1D CNN classifier
  * `2d_cnn.py`: training and inference of the 2D CNN classifier

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
