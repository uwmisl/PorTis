import argparse
import pandas as pd
from tqdm.auto import tqdm

import sys 
sys.path.append('..')
from src.constants import (
    SAMPLES_DICT_3CLASS, METADATA_DICT_3CLASS, TEST_INDICES_DICT_3CLASS,
    prefix_base_data_dir
)
from src.metrics import get_test_metrics
from src.models import create_1DCNN
from src.utils_data import get_dataloader, load_train_data, load_sim_data, load_test_data, split_train_val
from src.utils_models import (
    get_device, load_model, train_model, test_model, run_model, save_model, simulate_model, plot_losses, plot_simulation
)
from src.signal_processing import SignalProcessor


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="./", help="path to directory that contains dataset")
parser.add_argument("--train_test_split_id", type=int, help="1-indexed id for test split")
parser.add_argument("--length_thresh", type=int, help="event length threshold")
parser.add_argument("--to_train", default=False, action="store_true", help="run training")
parser.add_argument("--to_eval", default=False, action="store_true", help="run evaluation")
parser.add_argument("--to_sim", default=False, action="store_true", help="run simulation")
parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=150, help="number of epochs to run training")
parser.add_argument("--model_path", type=str, default=None, help="path to model checkpoint for evaluation or simulation")
args = parser.parse_args()

base_dir = args.base_dir
train_test_split_id = args.train_test_split_id
length_thresh = args.length_thresh
to_train = args.to_train
to_eval = args.to_eval
to_sim = args.to_sim
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
model_chkpt_path = args.model_path
if model_chkpt_path is None:
    model_chkpt_path = f"../checkpoints/1d_cnn_ckpt_test={train_test_split_id}_lenthresh={length_thresh}.pt"

# Get constants, data paths
SAMPLES_DICT_3CLASS = prefix_base_data_dir(SAMPLES_DICT_3CLASS, base_dir)
METADATA_DICT_3CLASS = prefix_base_data_dir(METADATA_DICT_3CLASS, base_dir)
TEST_INDICES_DICT_3CLASS = prefix_base_data_dir(TEST_INDICES_DICT_3CLASS, base_dir)
num_classes = 3
metrics_df = pd.DataFrame(columns=["test_idx", "auroc", "aupr", "accuracy", *[f"f1_class{i}" for i in range(num_classes)]])
device = get_device()

# Load train data and balance
if to_train:
    X_train, y_train, samples_train, labels_key = load_train_data(
        SAMPLES_DICT_3CLASS, train_test_split_id, length_thresh
    )

# Load balanced test data
if to_eval:
    X_test, y_test, samples_test = load_test_data(
        SAMPLES_DICT_3CLASS, train_test_split_id, length_thresh=length_thresh, indices_dict=TEST_INDICES_DICT_3CLASS
    )

# Load sim data
if to_sim:
    # Don't specify a length threshold here, because we utilize the thresh later
    X_sim, y_sim, samples_sim, end_times_sim, event_lengths_sim = load_sim_data(
        SAMPLES_DICT_3CLASS, METADATA_DICT_3CLASS, train_test_split_id
    )

# Pre-process data and create dataloaders
signal_length = 1000
signal_processor = SignalProcessor(
    downsample=True,
    downsample_points=signal_length,
    pad=True, pad_len=signal_length,
    add_channels_dim=True
)
if to_train:
    X_train, y_train, samples_train, _ = signal_processor.transform(X_train, y_train, samples_train, None)
    X_train, X_val, y_train, y_val = split_train_val(X_train, y_train)
    trainloader = get_dataloader(X_train, y_train, batch_size)
    valloader = get_dataloader(X_val, y_val, len(X_val), shuffle=False)
if to_eval:
    X_test, y_test, _, _ = signal_processor.transform(X_test, y_test, samples_test, None)
    testloader = get_dataloader(X_test, y_test, 32, shuffle=False)
if to_sim:
    X_sim, y_sim, _, _ = signal_processor.transform(X_sim, y_sim, samples_sim, None)
    # A separate simulation dataloader for each class
    simloaders = [
        get_dataloader(
            X_sim[y_sim == class_idx],
            list(zip(y_sim[y_sim == class_idx], end_times_sim[y_sim == class_idx], event_lengths_sim[y_sim == class_idx])),
            1,  # Load each event one at a time
            drop_last=False,
            shuffle=False, num_workers=1
        )
        for class_idx in range(num_classes)
    ]
print("Finished loading data")

# Create model
input_data_length = X_train.shape[2] if to_train else (X_test.shape[2] if to_eval else X_sim.shape[2])
model = create_1DCNN(input_data_length=input_data_length, num_classes=num_classes)
model.to(device)

#########################################
# Train the model and save a checkpoint #
#########################################
if to_train:
    train_losses, val_losses = train_model(model, trainloader, lr=lr, epochs=epochs, valloader=valloader)
    plot_losses(train_losses, val_losses, model_description="1D Classifier")
    save_model(model, f"../checkpoints/1d_cnn_ckpt_test={train_test_split_id}_lenthresh={length_thresh}.pt")

##################################
# Test the model and log metrics #
##################################
if to_eval:
    # Load model if not just trained
    if not to_train:
        if model_chkpt_path == "":
            raise Exception("Either specify `--to_train` to train a model or please provide `--model_path`")
        else:
            load_model(model, model_chkpt_path)
    y_pred, y_pred_proba, y_test = run_model(model, testloader, device)
    auroc, aupr, accuracy, f1_per_class = get_test_metrics(y_test, y_pred, y_pred_proba)
    print(f"AUROC: {auroc}, AUPR: {aupr}, accuracy: {accuracy}, F1 per class: {f1_per_class}")
    metrics_df = pd.DataFrame(columns=["test_idx", "auroc", "aupr", "accuracy", *[f"f1_class{i}" for i in range(num_classes)]])
    metrics_df = metrics_df.append(
        {
            "test_idx": train_test_split_id,
            "auroc": auroc,
            "aupr": aupr,
            "accuracy": accuracy,
            **{
                f"f1_class{i}": f1_per_class[i] for i in range(num_classes)
            }
        },
        ignore_index=True
    )
    metrics_df.to_csv(f"../res/1d_cnn_metrics_test={train_test_split_id}_lenthresh={length_thresh}.csv")

#################################################
# Run the simulation and save time-series plots #
#################################################
if to_sim:
    # Load model if not just trained
    if not to_train:
        if model_chkpt_path == "":
            raise Exception("Either specify `--to_train` to train a model or please provide `--model_path`")
        else:
            load_model(model, model_chkpt_path)
    # Simulate each class separately
    labels_key = {i: label for i, label in enumerate(list(SAMPLES_DICT_3CLASS.keys()))}
    for (l, simloader) in enumerate(simloaders):
        # Load data in order and run model inference
        y_pred, y_test, class_pred_counts, class_pred_prob_avgs, end_times, pred_proba_all = simulate_model(model, simloader, num_classes, length_thresh=length_thresh)
        # Plot the number of predictions and class probabilites for each class over time
        plot_simulation(end_times, class_pred_counts, class_pred_prob_avgs, pred_proba_all, labels_key, l, num_classes, filename=f"../res/1d_cnn_simulation_test={train_test_split_id}_lenthresh={length_thresh}_class={l}.png")