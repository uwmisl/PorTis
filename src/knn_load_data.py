import os, sys
import numpy as np

# change this to the directory where you want to keep the indices, can be any directory but make sure it exists
INDICES_SAVE_PATH_3CLASS = '/mmfs1/gscratch/ml4ml/cailinw/pore_data/'
# change this to the directory where you want to keep the DTW mdistance matrix, the directory should exist beforehand.
DTW_SAVE_PATH_3CLASS = "/mmfs1/gscratch/ml4ml/cailinw/pore_data/MinION_processed_data/dtw/3class/"

def prefix_data_dir(path, new_base_dir):
    if not new_base_dir.startswith("/"):
        new_base_dir = "/" + new_base_dir
    if not new_base_dir.endswith("/"):
        new_base_dir = new_base_dir + "/"
    if type(path) == dict:
        return {
            class_name: [
                new_base_dir + path
                for path in path_list
            ]
            for class_name, path_list in path.items()
        }
    else:
        raise Exception("Invalid path. Must be a dictionary")
        
# Filters signals that have a length smaller than the given threshold and returns new list.
#   signals (list): list of signals, each signal could be an np array
#   min_size (int): threshold for which signals with smaller length should be discarded.
def cut_signals(signals, min_size, indices):
    return indices[np.array([len(signal) for signal in signals]) > min_size]

# Selects the same number of signals for all tissue samples by randomly sampling
#   data (list): list of np.arrays containing data to balance
#   sample_size(int): minimum size of all samples for which to downsample to
def balance_data(data, sample_size):
    for i in range(len(data)):
        data[i] = np.random.choice(data[i], sample_size, replace=False)
    return data

def load_np_data(npy_file):
    return np.load(npy_file, allow_pickle=True, encoding="latin1")

# Gets the paths to the files storing the indices for each data file. Returns these paths
# in the same dict structure as samples_dict
#   samples_dict (dict): dict from class name to list of paths to the npy arrays for each sample
#   save_dir (str): path to directory to save indices files in
def get_indices_paths(samples_dict, save_dir):
    indices_dict = {}
    for data_class, sample_list in samples_dict.items():
        indices_dict[data_class] = []
        for s_id, sample_path in enumerate(sample_list):
            indices_dict[data_class].append(
                os.path.join(save_dir, os.path.basename(sample_path).split(".")[0] + "_indices.npy")
            )
    return indices_dict


def get_data_indices(samples_dict, min_size, save_dir=None, random_seed=123):
    np.random.seed(random_seed)  # Random seed to ensure same data is loaded each time

    class_names = list(samples_dict.keys())
    num_classes = len(samples_dict.keys())
    num_samples = [len(samples_dict[list(samples_dict.keys())[i]]) for i in range(len(list(samples_dict.keys())))]

    print(f"class_names={class_names}")
    print(f"num_classes={num_classes}")
    print(f"num_replicates={num_samples}")

    # Load signals for each class, for each sample. Select signals with length above threshold
    # Save the indices of the signals we select from each data file
    indices = []
    min_class_sample_size = 0
    print('In progress...')
    for cl_i, (data_class, sample_list) in enumerate(samples_dict.items()):
        print(' ')
        for s_id, sample_path in enumerate(sample_list):
            print(f'Class {data_class}, replicate {s_id+1}/{num_samples[cl_i]}', end='\r')
            signals = load_np_data(sample_path)
            signals_indices = np.arange(signals.size)
            signals_indices = cut_signals(signals, min_size, signals_indices)
            indices.append(signals_indices)
            # Compute the size of the smallest class sample
            min_class_sample_size = (
                len(signals_indices)
                if min_class_sample_size == 0
                else min(min_class_sample_size, len(signals_indices))
            )
            
    # Balance all classes and samples by undersampling
    indices = balance_data(indices, min_class_sample_size)
    indices_dict = {}

    # Put indices back in dictionary similar to samples_dict, and save to disk
    i = 0
    if save_dir is not None:
        indices_path_dict = get_indices_paths(samples_dict, save_dir)
    for data_class, sample_list in samples_dict.items():
        indices_dict[data_class] = []
        for s_id, sample_path in enumerate(sample_list):
            if save_dir is not None:
                indices_save_path = indices_path_dict[data_class][s_id]
                print(f"Saving indices at {indices_save_path}")
                np.save(indices_save_path, indices[i])
            indices_dict[data_class].append(indices[i])
            i += 1
            
    return indices_dict


# Loads raw signals from disk and select certain signal indices, based on previously created indices files.
#   samples_dict (dict): dict from class name to list of paths to the npy arrays for each sample
#   data_indices (dict): dict from class name to list of indices lists to select from each file
#   save_dir (str) : path to directory in which indices have been saved
#   load_all_data (bool): specify whether all data should be used, or only certain indices
#   load_raw_signals (bool): specify whether to load all raw signals or not
def load_data(samples_dict, data_indices=None, save_dir=None, load_all_data=False, load_raw_signals=True):
    """
    If load_raw_signals=True (Default) will return the sample dictionary with the full signals inside
    If set to false, the dictionary will be empty, which can be useful when only labels, samples or indices are needed
    Note: this doesn't work if load_all_data=True, as it needs to load the data to get the size of it
    """

    class_names = list(samples_dict.keys())
    num_classes = len(samples_dict.keys())
    num_samples = [len(samples_dict[list(samples_dict.keys())[i]]) for i in range(len(list(samples_dict.keys())))]

    print(f"class_names={class_names}")
    print(f"num_classes={num_classes}")
    print(f"num_samples={num_samples}")
    
    # Get data indices paths
    if not load_all_data:
        if data_indices is None:
            indices_dict = get_indices_paths(samples_dict, save_dir)

    # Load signals for each class, for each sample.
    data = []
    labels = []
    samples = []
    indices = []
    print('In progress...')            
    for c_id, (data_class, sample_list) in enumerate(samples_dict.items()):
        print(' ')
        for s_id, sample_path in enumerate(sample_list):
            print(f'Class {data_class}, replicate {s_id+1}/{num_samples[c_id]}', end='\r')
            if not load_all_data:
                if data_indices is None:
                    data_indices_for_sample = np.load(indices_dict[data_class][s_id])
                else:
                    data_indices_for_sample = data_indices[data_class][s_id]
                if load_raw_signals:
                    signals = load_np_data(sample_path)[data_indices_for_sample]
                    num_signals = len(signals)
                else:
                    # get the number of signals from the length of indices
                    num_signals = len(data_indices_for_sample)
                indices.append(data_indices_for_sample)
            else:
                # in this case, we need to load the data to know how many signals there are
                # so it is not possible to have load_raw_signals active
                # if load_raw_signals:
                signals = load_np_data(sample_path)
                num_signals = len(signals)
                # this would be the alternative to avoid loading but does not work bc it is a numpy object
                # num_signals = len(np.load(sample_path, mmap_mode='r', allow_pickle=True, encoding="latin1"))
            
            if load_raw_signals:
                data.append(signals)
            labels.append(np.full(num_signals, c_id))
            samples.append(np.full(num_signals, s_id))

    
    # Construct np.array for the data
    if load_raw_signals:
        data = np.concatenate(np.array(data, dtype='object'))
    else:
        data = None
    # Construct np.array for the labels (1, 1, ..., 2, 2, ..., 3, 3, ...)
    labels = np.concatenate(labels)
    # Construct np.array for the samples (1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, ...)
    samples = np.concatenate(samples)
    # Construct np.array for the indices
    if not load_all_data:
        indices = np.concatenate(indices)
    
    if load_raw_signals:
        if (data.shape[0] != labels.shape[0]) or (data.shape[0] != samples.shape[0]):
            raise Exception("Error when creating labels or samples")   
    
    labels_key = {i: label for i, label in enumerate(list(samples_dict.keys()))}
    
    return data, labels, samples, indices, labels_key
