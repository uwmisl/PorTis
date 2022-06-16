import argparse
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from tqdm.auto import tqdm

import sys 
sys.path.append('..')
from src.constants import (
    SAMPLES_DICT_3CLASS, TEST_INDICES_DICT_3CLASS,
    prefix_base_data_dir
)
from src.utils_data import load_train_data, load_test_data


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="./", help="path to directory that contains dataset")
args = parser.parse_args()

# Constants
base_dir = args.base_dir
num_classes = 3
SAMPLES_DICT_3CLASS = prefix_base_data_dir(SAMPLES_DICT_3CLASS, base_dir)
TEST_INDICES_DICT_3CLASS = prefix_base_data_dir(TEST_INDICES_DICT_3CLASS, base_dir)
train_test_split_ids = [0, 1, 2, 3]

# Load data for each replicate, one at a time, and generate indices to achieve a balanced test set
for train_test_split_id in tqdm(train_test_split_ids, desc="Test split"):
    # Load the test replicate
    X_test, y_test, samples_test = load_test_data(
        SAMPLES_DICT_3CLASS, train_test_split_id, length_thresh=0, balance=False
    )
    # Construct the list of indices (for each class)
    indices = []
    for c_id in range(num_classes):
        indices.append(np.arange((y_test == c_id).sum()))
    indices = np.concatenate(indices)

    # Randomly undersample over-represented classes
    X_test = X_test.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    indices = indices.reshape((-1, 1))
    under_sampler = RandomUnderSampler(random_state=42)
    indices, y_test = under_sampler.fit_resample(indices, y_test)
    # Sort indices for each class and save
    for c_id, (data_class, test_indices_list) in enumerate(samples_dict.items()):
        indices_save_path = TEST_INDICES_DICT_3CLASS[data_class][train_test_split_id]
        print(f"Saving indices at {indices_save_path}")
        indices_class = np.sort(indices.reshape((-1))[y_test == c_id])
        np.save(indices_save_path, indices_class)