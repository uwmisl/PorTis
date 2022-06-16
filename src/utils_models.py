import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_device(use_cuda=True):
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def train_model(model, trainloader, lr=0.001, decay=1e-7, epochs=100, n_epochs_stop=6, optim_type="sgd", valloader=None, device=get_device()):
    criterion = nn.CrossEntropyLoss()
    if optim_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=0.9)  #0.7557312793639288)
    elif optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise Exception("Unknown optimizer. Please specify either 'sgd' or 'adam'")
        

    model.train()

    train_losses = []
    val_losses = []
    min_val_loss = -float("inf")
    epochs_no_improve = 0
#     n_epochs_stop = 6

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_size = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).to(dtype=torch.float64)
            labels = labels.to(dtype=torch.long)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_size += inputs.size(0)
            
        epoch_loss = running_loss / running_size
        train_losses.append(epoch_loss)
        print('Finished epoch number ' + str(epoch))
        print('Loss on train set after this epoch ' + str(epoch_loss))
        
        if valloader is not None:
            inc = 5
            if (epoch % inc) == 0:
                accuracy, val_loss = test_model(model, criterion, valloader, device)
                val_losses.append(val_loss)
                print('Loss on the validation set: ' + str(val_loss))
                print('Accuracy of the network on the validation set: %d %%' 
                        % (100 * accuracy))
                accuracy, _ = test_model(model, criterion, trainloader, device)
                print('Accuracy of the network on the training set: %d %%' 
                        % (100 * accuracy))
                model.train()

                # Check if early stopping condition is met
                if val_loss < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_improve += inc  #1
                if (epoch > 10) and (epochs_no_improve >= n_epochs_stop):
                    print(f"Early stopping condition met at epoch {epoch}")
                    break
            
    return train_losses, val_losses


def test_model(model, criterion, testloader, device=get_device()):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.to(device)

            outputs = model(inputs).to(dtype=torch.float64)
            labels = labels.to(dtype=torch.long)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss /= total
    accuracy = correct / total
    return accuracy, test_loss


def run_model(model, testloader, device=get_device()):
    model.eval()
    pred = []
    pred_proba = []
    lab = []
    with torch.no_grad():
        for data_batch in testloader:
            inputs, labels = data_batch
            inputs, labels = inputs.float().to(device), labels.to(device)
            
            outputs = model(inputs).to(dtype=torch.float64)
            labels = labels.to(dtype=torch.long)
            _, pred_batch = torch.max(outputs.data, 1)
            pred_proba_batch = F.softmax(outputs, dim=1)
            pred.append(pred_batch.cpu().numpy())
            pred_proba.append(pred_proba_batch.cpu().numpy())
            lab.append(labels.cpu().numpy())
            
    return np.concatenate(np.array(pred)), np.concatenate(np.array(pred_proba)), np.concatenate(np.array(lab))


def simulate_model(model, simloader, n_classes, length_thresh=0, device=get_device()):
    model.eval()
    pred = []  # All the predictions over time
    pred_proba_all = []
    lab = []  # All the true labels over time
    class_pred_counts_totals = [0 for i in range(n_classes)]  # Total number of predictions of each class
    class_pred_prob_avgs_totals = np.array([0.0 for i in range(n_classes)])  # Average prediction probability for each class
    class_pred_counts = []  # Number of predictions of each class so far at each time point
    class_pred_prob_avgs = []  # Average prediction probability for each class so far at each time point
    num_greater_thresh = 0
    all_end_times = []
    with torch.no_grad():
        # Assumes that the batch size is 1
        for (i, data_batch) in tqdm(enumerate(simloader)):
            inputs, (labels, end_times, event_lengths) = data_batch
            inputs, labels = inputs.float().to(device), labels.to(device)
            labels = labels.to(dtype=torch.long)
            lab.append(labels.item())

            all_end_times.append(end_times.item())

            # Count how many predictions for each class up till this point in time
            # Only account for predictions on events greater than length threshold
            if event_lengths.item() >= length_thresh:
                num_greater_thresh += 1

                outputs = model(inputs).to(dtype=torch.float64)
            
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.item()
                pred.append(predicted)
                pred_proba = F.softmax(outputs, dim=1).reshape((-1,)).cpu().numpy()

                class_pred_counts_totals[predicted] += 1
                class_pred_prob_avgs_totals = (
                    (class_pred_prob_avgs_totals * (num_greater_thresh - 1)) + pred_proba
                ) / num_greater_thresh
                pred_proba_all.append(pred_proba)
            else:
                pred_proba_all.append(np.array([0.0, 0.0, 0.0]))
            class_pred_counts.append(np.copy(class_pred_counts_totals))
            class_pred_prob_avgs.append(np.copy(class_pred_prob_avgs_totals))

    return np.array(pred), np.array(lab), np.array(class_pred_counts), np.array(class_pred_prob_avgs), np.array(all_end_times), np.array(pred_proba_all)

def plot_losses(train_losses, test_losses, model_description="Classifier", filename=None):
    plt.plot(train_losses, label = "Train Loss")
    plt.plot(test_losses, label = "Test Loss")
    plt.title(f"Loss over Epochs for {model_description}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_simulation(end_times, class_pred_counts, class_pred_prob_avgs, pred_proba_all, labels_key, l, num_classes, filename=None):
    end_times = end_times / 10000  # To get seconds
    # Plot counts
    for c in range(num_classes):
        plt.plot(end_times, class_pred_counts[:,c], label=labels_key[c])
    plt.legend()
    plt.title(f"Model Inference Across Time for {labels_key[l].capitalize()} Class")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Events Classified")
    if filename is not None:
        plt.savefig(filename.replace("simulation", "simulation_counts"), dpi=300)
    plt.show()
    # Plot running average probabilities
    for c in range(num_classes):
        plt.plot(end_times, class_pred_prob_avgs[:,c], label=labels_key[c])
    plt.legend()
    plt.title(f"Model Inference Across Time for {labels_key[l].capitalize()} Class")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Running Average Prediction Probability")
    if filename is not None:
        plt.savefig(filename.replace("simulation", "simulation_probs"), dpi=300)
    plt.show()
    # Plot cumulative probabilities
    for c in range(num_classes):
        plt.plot(end_times, np.cumsum(pred_proba_all[:,c]), label=labels_key[c])
    plt.legend()
    plt.title(f"Model Inference Across Time for {labels_key[l].capitalize()} Class")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Cumulative Sum of Model Prediction Confidence")
    if filename is not None:
        plt.savefig(filename.replace("simulation", "simulation_cumprobs"), dpi=300)
    plt.show()
    # Plot probabilites
    for c in range(num_classes):
        plt.plot(end_times, pred_proba_all[:,c], label=labels_key[c])
    plt.legend()
    plt.title(f"Model Inference Across Time for {labels_key[l].capitalize()} Class")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Prediction Probability")
    if filename is not None:
        plt.savefig(filename.replace("simulation", "simulation_probsindiv"), dpi=300)
    plt.show()

def save_model(model, filename):
    # Save model
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    # Load model
    model.load_state_dict(torch.load(filename))