import time
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import RegNet_X_32GF_Weights
from torchinfo import summary

model = models.regnet_x_32gf(weights=RegNet_X_32GF_Weights.IMAGENET1K_V2)
summary(model, input_size=(1, 3, 256, 256), depth=3)

numFeatures = model.fc.in_features
numClasses = 75
model.fc = nn.Linear(numFeatures, numClasses)
