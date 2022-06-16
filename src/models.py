import math as m
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Creates a Random Forest Classifier.

Parameters:
    n_estimators (int) : number of estimators for the sklearn random forest classifier
    max_depth (int) : maximum depth of the tree created by the sklearn random forest classifier
"""
def create_random_forest_classifier(n_estimators, max_depth):
    return RandomForestClassifier(verbose=2, n_estimators=n_estimators, max_depth=max_depth)


"""Creates a 1D CNN Pytorch model.

Parameters:
    in_channels (int) : the number of input channels for the first layer
    num_classes (int) : the number of output classes
    conv_params (list[dict]) : list of dicts specifying parameters for each consecutive conv layer;
                                if not provided, a default (previously determined) set is used
    lin_params (list[int]) : list specifying output channels for each consecutive linear layer
"""
def create_1DCNN(input_data_length, num_classes, in_channels=1, conv_params=None, lin_params=None):
    
    if conv_params is None:
        conv_params = [
            {
                "O": 64, "K": 7, "KP": 3
            },
            {
                "O": 128, "K": 5, "KP": 3
            },
            {
                "O": 256, "K": 3, "KP": 3
            }
        ]

    if lin_params is None:
        lin_params = [512]
        
    conv_linear_out = input_data_length
    for i in range(len(conv_params)):
        conv_linear_out = m.floor((conv_linear_out - conv_params[i]["K"] + 1) / conv_params[i]["KP"])
    conv_linear_out = int(conv_linear_out * conv_params[-1]["O"])

    class CNN1D(nn.Module):

        def __init__(self):

            super(CNN1D, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, conv_params[0]["O"], conv_params[0]["K"]),
                nn.ReLU(),
                nn.MaxPool1d(conv_params[0]["KP"])
            )

            self.conv2 = nn.Sequential(
                nn.Conv1d(conv_params[0]["O"], conv_params[1]["O"], conv_params[1]["K"]),
                nn.ReLU(),
                nn.MaxPool1d(conv_params[1]["KP"]),
                nn.Dropout(0.4)
            )

            self.conv3 = nn.Sequential(
                nn.Conv1d(conv_params[1]["O"], conv_params[2]["O"], conv_params[2]["K"]),
                nn.ReLU(),
                nn.MaxPool1d(conv_params[2]["KP"]),
                nn.Dropout(0.4)
            )

            self.fc1 = nn.Sequential(
                nn.Linear(conv_linear_out, lin_params[0]),
                nn.ReLU(),
                nn.Dropout(0.6)
            )

            self.fc2 = nn.Linear(lin_params[0], num_classes)

            self.relu = nn.ReLU()
            self.lrelu = nn.LeakyReLU()
            self.logsig = nn.Sigmoid()


        def forward(self, x):
            x = x.float()
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            return x
        
    return CNN1D()


"""Creates a 2D CNN Pytorch model.

Parameters:
    in_channels (int) : the number of input channels for the first layer
    num_classes (int) : the number of output classes
    conv_params (list[dict]) : list of dicts specifying parameters for each consecutive conv layer;
                                if not provided, a default (previously determined) set is used
    lin_params (list[int]) : list specifying output channels for each consecutive linear layer
"""
def create_2DCNN(input_data_length, num_classes, in_channels=1, conv_params=None, lin_params=None):

    if conv_params is None:
        conv_params = [
            {
                "O": 120, "K": 2, "KP": 2
            },
            {
                "O": 240, "K": 1, "KP": 1
            },
            {
                "O": 488, "K": 2, "KP": 2
            }
        ]

    if lin_params is None:
        lin_params = [148]
        
    conv_linear_out = input_data_length
    for i in range(len(conv_params)):
        conv_linear_out = m.floor((conv_linear_out - conv_params[i]["K"] + 1) / conv_params[i]["KP"])
    conv_linear_out = int(m.floor(conv_linear_out ** 2) * conv_params[-1]["O"])

    class CNN2D(nn.Module):
        def __init__(self):

            super(CNN2D, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, conv_params[0]["O"], conv_params[0]["K"]),
                nn.LeakyReLU(),
                nn.MaxPool2d(conv_params[0]["KP"]),
                nn.Dropout(0.8)  #0.5)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(conv_params[0]["O"], conv_params[1]["O"], conv_params[1]["K"]),
                nn.LeakyReLU(),
                nn.MaxPool2d(conv_params[1]["KP"]),
                nn.Dropout(0.5)  #4)
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(conv_params[1]["O"], conv_params[2]["O"], conv_params[2]["K"]),
                nn.LeakyReLU(),
                nn.MaxPool2d(conv_params[2]["KP"]),
                nn.Dropout(0.5)  #4)
            )

            self.fc1 = nn.Sequential(
                nn.Linear(conv_linear_out, lin_params[0]),
                nn.LeakyReLU(),
                nn.Dropout(0.8)  #5)
            )

            self.fc2 = nn.Linear(lin_params[0], num_classes)

            self.relu = nn.ReLU()
            self.lrelu = nn.LeakyReLU()
            self.logsig = nn.Sigmoid()


        def forward(self, x):
            x = x.float()
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(len(x), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            return x
        
    return CNN2D()