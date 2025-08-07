import os
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import RegNet_Y_16GF_Weights, EfficientNet_V2_M_Weights
from torchinfo import summary
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data:pd.DataFrame = pd.read_csv(csv_file)
        self.data = self.data["label"].str.title() # for cleaner formatting
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row['label']
        return image, label


def load_model(num_classes):
    model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    summary(model, input_size=(1, 3, 256, 256), depth=3)
    return model, criterion, optimizer


def load_data():
    
    # Define transforms 
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ensure all are this size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    training_dataset = CustomDataset(
        csv_file='./data/Training_set.csv',
        img_dir='./data/train',
        transform=transform
    )
    validation_dataset = CustomDataset(
        csv_file='./data/Testing_set.csv',
        img_dir='./data/test',
        transform=transform
    )

    # Create dataloaders
    trainLoader = DataLoader(training_dataset, batch_size=batch_size)
    valLoader = DataLoader(validation_dataset, batch_size=batch_size)

    return trainLoader, valLoader


def main():
    
    # Load model
    numClasses = 75
    model, criterion, optimizer = load_model(numClasses)

    # Load data
    trainLoader, valLoader = load_data()
