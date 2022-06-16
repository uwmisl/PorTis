from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch


# Loads data from a npy file and returns a np array.
#   npy_file (str): path to npy file
def load_np_data(npy_file):
    return np.load(npy_file, allow_pickle=True, encoding="latin1")

# Loads data from pkl file and returns it.
#    pkl_file (str): path to pkl file
def load_pkl_data(pkl_file):
    with (open(pkl_file, "rb")) as f:
        data = pickle.load(f, encoding="latin1")
    return data

# Loads metadata files from disk.
#   metadata_dict (dict): dict from class name to list of paths to the pkl files for each sample
#   samples_dict (dict): dict from class name to list of paths to the npy arrays for each sample (used to get names of indices files)
#   data_indices (dict): dict from class name to list of indices lists to select from each file
#   save_dir (str) : path to directory in which indices have been saved
#   use_all_data (bool): specify whether all data should be used, or only certain indices
def load_metadata(metadata_dict, samples_dict, data_indices=None, save_dir=None, load_all_data=False):
    class_names = list(metadata_dict.keys())
    num_classes = len(metadata_dict.keys())
    num_samples = [len(metadata_dict[list(metadata_dict.keys())[i]]) for i in range(len(list(metadata_dict.keys())))]

    print(f"class_names={class_names}")
    print(f"num_classes={num_classes}")
    print(f"num_samples={num_samples}")
    
    # Get data indices paths
    if not load_all_data:
        if data_indices is None:
            indices_dict = get_indices_paths(samples_dict, save_dir)

    # Construct metadata in dataframe
    data = pd.DataFrame()
    for c_id, (data_class, sample_list) in enumerate(tqdm(metadata_dict.items())):
        for s_id, sample_path in enumerate(tqdm(sample_list)):
            # Get the indices; either load from disk or use the provided dict of indices
            if not load_all_data:
                if data_indices is None:
                    data_indices_for_sample = np.load(indices_dict[data_class][s_id])
                else:
                    data_indices_for_sample = data_indices[data_class][s_id]
                sample_data = load_pkl_data(metadata_dict[data_class][s_id]).loc[data_indices_for_sample]
            else:
                sample_data = load_pkl_data(metadata_dict[data_class][s_id])
            sample_data["class"] = data_class
            sample_data["sample_idx"] = s_id
            data = pd.concat([data, sample_data])            
    return data


def load_train_data(samples_dict, train_test_split_id, length_thresh=0, balance=True, random_seed=123):
    np.random.seed(random_seed)  # Random seed to ensure same data is loaded each time
    # Load data: signals for each class, for each sample
    data = []
    labels = []
    samples = []
    for c_id, (data_class, sample_list) in enumerate(tqdm(samples_dict.items())):
        for s_id, sample_path in enumerate(tqdm(sample_list)):
            # Only load samples that are not in the test set (train_test_split_id)
            if s_id != train_test_split_id:
                signals = load_np_data(sample_path)
                data.append(signals)
                labels.append(np.full(len(signals), c_id))
                samples.append(np.full(len(signals), s_id))
    
    # Construct np.array for the data
    data = np.concatenate(np.array(data, dtype='object'))
    # Construct np.array for the labels (1, 1, ..., 2, 2, ..., 3, 3, ...)
    labels = np.concatenate(labels)
    # Construct np.array for the samples (1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, ...)
    samples = np.concatenate(samples)
    
    if (data.shape[0] != labels.shape[0]) or (data.shape[0] != samples.shape[0]):
        raise Exception("Error when creating labels or samples")   
    
    labels_key = {i: label for i, label in enumerate(list(samples_dict.keys()))}
    
    # Select events with length greater than length threshold
    if length_thresh > 0:
        idxs = (np.array([len(signal) for signal in data]) > length_thresh)
        data = data[idxs]
        labels = labels[idxs]
        samples = samples[idxs]

    # Balance classes, so that there are equal number of data points from each class for each sample
    if balance:
        data = data.reshape((-1, 1))
        labels = labels.reshape((-1, 1))
        samples = samples.reshape((-1, 1))
        under_sampler = RandomUnderSampler(random_state=42)
        data, labels = under_sampler.fit_resample(data, ((labels + 1) * 10) + (samples + 1))
        labels, samples = ((labels / 10) - 1).astype(int), (labels % 10) - 1
    
        data = data.reshape((-1,))
        labels = labels.reshape((-1,))
    
    return data, labels, samples, labels_key

def load_test_data(samples_dict, train_test_split_id, length_thresh=0, indices_dict=None, balance=False, random_seed=123):
    np.random.seed(random_seed)  # Random seed to ensure same data is loaded each time
    # Load data: signals for each class, for each sample
    data = []
    labels = []
    for c_id, (data_class, sample_list) in enumerate(tqdm(samples_dict.items())):
        for s_id, sample_path in enumerate(tqdm(sample_list)):
            # Only load samples that are in the test set (train_test_split_id)
            if s_id == train_test_split_id:
                signals = load_np_data(sample_path)
                if indices_dict is not None:
                    indices = np.load(indices_dict[data_class][s_id])
                    signals = signals[indices]
                data.append(signals)
                labels.append(np.full(len(signals), c_id))
    
    # Construct np.array for the data
    data = np.concatenate(np.array(data, dtype='object'))
    # Construct np.array for the labels (1, 1, ..., 2, 2, ..., 3, 3, ...)
    labels = np.concatenate(labels)
    # Construct np.array for samples
    samples = np.full(len(data), train_test_split_id)
    
    # Select events with length greater than length threshold
    if length_thresh > 0:
        idxs = (np.array([len(signal) for signal in data]) > length_thresh)
        data = data[idxs]
        labels = labels[idxs]
        samples = samples[idxs]
    
    # Balance classes, so that there are equal number of data points from each class
    if balance:
        data = data.reshape((-1, 1))
        labels = labels.reshape((-1, 1))
        under_sampler = RandomUnderSampler(random_state=42)
        data, labels = under_sampler.fit_resample(data, labels)
        samples = samples[:len(data)]

        data = data.reshape((-1,))
        labels = labels.reshape((-1,))
    
    if (data.shape[0] != labels.shape[0]):
        raise Exception("Error when creating labels or samples")
    
    return data, labels, samples

def load_sim_data(samples_dict, metadata_dict, train_test_split_id, random_seed=123):
    # First load test data as normal (np random seed gets set here and
    # load all data, so no balancing and length threshold)
    data, labels, samples = load_test_data(samples_dict, train_test_split_id, length_thresh=0, balance=False, random_seed=random_seed)
    num_classes = len(samples_dict)
    # Load metadata
    metadata = load_metadata(
        {k: [v[train_test_split_id]] for k, v in metadata_dict.items()},
        {k: [v[train_test_split_id]] for k, v in samples_dict.items()},
        load_all_data=True
    )
    classes = metadata["class"]
    samples_ids = metadata["sample_idx"]
    end_times = metadata["end_obs"]
    event_lengths = metadata["duration_obs"]
    # Note that there is only one sample for each class, since we loaded test data
    # Loop over each class and sort the sim data based on the end times
    data_sorted = []
    labels_sorted = []
    samples_sorted = []
    end_times_sorted = []
    event_lengths_sorted = []
    
    for c in range(num_classes):
        # For each, (1) select values for this class and (2) sort by end times
        c_inds = (labels == c)
        index_arr = np.argsort(end_times[c_inds])
        data_sorted.append(data[c_inds][index_arr])
        labels_sorted.append(labels[c_inds][index_arr])
        samples_sorted.append(samples[c_inds][index_arr])
        end_times_sorted.append(end_times[c_inds][index_arr])
        event_lengths_sorted.append(event_lengths[c_inds][index_arr])

    return np.concatenate(data_sorted), np.concatenate(labels_sorted), np.concatenate(samples_sorted), np.concatenate(end_times_sorted), np.concatenate(event_lengths_sorted)

def split_train_val(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    return X_train, X_val, y_train, y_val

def get_dataloader(X, y, batch_size, drop_last=True, shuffle=True, num_workers=4):
    return torch.utils.data.DataLoader(
        list(zip(X, y)), batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=num_workers
    )