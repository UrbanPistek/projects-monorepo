import time
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import RegNet_Y_16GF_Weights, EfficientNet_V2_M_Weights
from torchinfo import summary


def load_model(numClasses):
    model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2)
    numFeatures = model.fc.in_features
    model.fc = nn.Linear(numFeatures, numClasses)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    summary(model, input_size=(1, 3, 256, 256), depth=3)
    return model, criterion, optimizer


def load_data(path: str):
    pass


def main():
    
    # Load model
    numClasses = 75
    model, criterion, optimizer = load_model(numClasses)