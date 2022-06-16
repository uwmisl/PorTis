# This dictionary contains a list of the path to the data for each replicate for each class.
# This represents all the data that is currently available for this project.
SAMPLES_DICT = {
    "heart": [
        "PorTis/segmented_peptides_raw_data/replicate1/segmented_peptides_raw_data_replicate1_heart.npy",
        "PorTis/segmented_peptides_raw_data/replicate2/segmented_peptides_raw_data_replicate2_heart.npy",
        "PorTis/segmented_peptides_raw_data/replicate3/segmented_peptides_raw_data_replicate3_heart.npy",
        "PorTis/segmented_peptides_raw_data/replicate4/segmented_peptides_raw_data_replicate4_heart.npy"
    ],
    "adrenal": [
        "PorTis/segmented_peptides_raw_data/replicate1/segmented_peptides_raw_data_replicate1_adrenal.npy",
        "PorTis/segmented_peptides_raw_data/replicate2/segmented_peptides_raw_data_replicate2_adrenal.npy",
        "PorTis/segmented_peptides_raw_data/replicate3/segmented_peptides_raw_data_replicate3_adrenal.npy",
        "PorTis/segmented_peptides_raw_data/replicate4/segmented_peptides_raw_data_replicate4_adrenal.npy"
    ],
    "aorta": [
        "PorTis/segmented_peptides_raw_data/replicate1/segmented_peptides_raw_data_replicate1_aorta.npy",
        "PorTis/segmented_peptides_raw_data/replicate2/segmented_peptides_raw_data_replicate2_aorta.npy",
        "PorTis/segmented_peptides_raw_data/replicate3/segmented_peptides_raw_data_replicate3_aorta.npy",
        "PorTis/segmented_peptides_raw_data/replicate4/segmented_peptides_raw_data_replicate4_aorta.npy"
    ]
}

SAMPLES_DICT_3CLASS = {key:value for (key, value) in SAMPLES_DICT.items() if key in {"heart", "adrenal", "aorta"}}

# This dictionary contains a list of the path to the metadata pkl file for each replicate for each class.
METADATA_DICT = {
    "heart": [
        "PorTis/segmented_peptides_metadata/replicate1/segmented_peptides_metadata_replicate1_heart.pkl",
        "PorTis/segmented_peptides_metadata/replicate2/segmented_peptides_metadata_replicate2_heart.pkl",
        "PorTis/segmented_peptides_metadata/replicate3/segmented_peptides_metadata_replicate3_heart.pkl",
        "PorTis/segmented_peptides_metadata/replicate4/segmented_peptides_metadata_replicate4_heart.pkl"
    ],
    "adrenal": [
        "PorTis/segmented_peptides_metadata/replicate1/segmented_peptides_metadata_replicate1_adrenal.pkl",
        "PorTis/segmented_peptides_metadata/replicate2/segmented_peptides_metadata_replicate2_adrenal.pkl",
        "PorTis/segmented_peptides_metadata/replicate3/segmented_peptides_metadata_replicate3_adrenal.pkl",
        "PorTis/segmented_peptides_metadata/replicate4/segmented_peptides_metadata_replicate4_adrenal.pkl"
    ],
    "aorta": [
        "PorTis/segmented_peptides_metadata/replicate1/segmented_peptides_metadata_replicate1_aorta.pkl",
        "PorTis/segmented_peptides_metadata/replicate2/segmented_peptides_metadata_replicate2_aorta.pkl",
        "PorTis/segmented_peptides_metadata/replicate3/segmented_peptides_metadata_replicate3_aorta.pkl",
        "PorTis/segmented_peptides_metadata/replicate4/segmented_peptides_metadata_replicate4_aorta.pkl"
    ],
}

METADATA_DICT_3CLASS = {key:value for (key, value) in METADATA_DICT.items() if key in {"heart", "adrenal", "aorta"}}


TEST_INDICES_DICT = {
    "heart": [
        "PorTis/segmented_peptides_test_indices/replicate1/segmented_peptides_test_indices_replicate1_heart.npy",
        "PorTis/segmented_peptides_test_indices/replicate2/segmented_peptides_test_indices_replicate2_heart.npy",
        "PorTis/segmented_peptides_test_indices/replicate3/segmented_peptides_test_indices_replicate3_heart.npy",
        "PorTis/segmented_peptides_test_indices/replicate4/segmented_peptides_test_indices_replicate4_heart.npy"
    ],
    "adrenal": [
        "PorTis/segmented_peptides_test_indices/replicate1/segmented_peptides_test_indices_replicate1_adrenal.npy",
        "PorTis/segmented_peptides_test_indices/replicate2/segmented_peptides_test_indices_replicate2_adrenal.npy",
        "PorTis/segmented_peptides_test_indices/replicate3/segmented_peptides_test_indices_replicate3_adrenal.npy",
        "PorTis/segmented_peptides_test_indices/replicate4/segmented_peptides_test_indices_replicate4_adrenal.npy"
    ],
    "aorta": [
        "PorTis/segmented_peptides_test_indices/replicate1/segmented_peptides_test_indices_replicate1_aorta.npy",
        "PorTis/segmented_peptides_test_indices/replicate2/segmented_peptides_test_indices_replicate2_aorta.npy",
        "PorTis/segmented_peptides_test_indices/replicate3/segmented_peptides_test_indices_replicate3_aorta.npy",
        "PorTis/segmented_peptides_test_indices/replicate4/segmented_peptides_test_indices_replicate4_aorta.npy"
    ]
}

TEST_INDICES_DICT_3CLASS = {key:value for (key, value) in TEST_INDICES_DICT.items() if key in {"heart", "adrenal", "aorta"}}

def prefix_base_data_dir(path, base_dir):
    if not base_dir.endswith("/"):
        base_dir += "/"
    if type(path) is str:
        return base_dir + path
    elif type(path) == dict:
        return {
            class_name: [
                base_dir + path
                for path in path_list
            ]
            for class_name, path_list in path.items()
        }
    else:
        raise Exception("Invalid path. Must be either string or dictionary")
