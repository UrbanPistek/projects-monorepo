import os
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import RegNet_Y_16GF_Weights
from torchinfo import summary
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pprint import pprint


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train_set=False):
        self.data: pd.DataFrame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Do a little clean up here
        if is_train_set:
            self.data = self.data["label"].str.title() # for cleaner formatting
            self.classes = np.unique(self.data.values)

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

    return model, criterion, optimizer


def load_data():
    
    # Define transforms 
    # Transforms apply each time the image is loaded
    # Effectively applying a new transform to each image during each new epoch
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # using 224 to start since that is what the model was originally trained on
        
        # Augmentation transforms
        transforms.ColorJitter(
            brightness=0.2,      # ±20% brightness
            contrast=0.2,        # ±20% contrast  
            saturation=0.2,      # ±20% saturation
            hue=0.1             # ±10% hue
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

    # Create dataset and dataloader
    # Dataset is already split, so not need to perform a random split
    training_dataset = CustomDataset(
        csv_file='./data/train_set.csv',
        img_dir='./data/train',
        transform=transform,
        is_train_set=True
    )
    validation_dataset = CustomDataset(
        csv_file='./data/val_set.csv',
        img_dir='./data/val',
        transform=transform
    )

    # Create dataloaders
    classes = training_dataset.classes
    trainLoader = DataLoader(training_dataset, batch_size=batch_size)
    valLoader = DataLoader(validation_dataset, batch_size=batch_size)

    return trainLoader, valLoader, classes


def main():

    # Show original model info
    base_model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2)
    summary(base_model, input_size=(1, 3, 224, 224), depth=3)
    
    # Load model
    ts = time.perf_counter()
    numClasses = 75
    device = "cpu" # using CPU to start
    model, criterion, optimizer = load_model(numClasses)
    te = time.perf_counter()
    elapsed = round(te-ts, 3)
    print(f"model loaded in: {elapsed}s")

    # Load data
    trainLoader, valLoader, classes = load_data()
    print(f"train set size: {len(trainLoader.dataset)}")
    print(f"val set size: {len(valLoader.dataset)}")
    print("classes: ")
    pprint(classes)

    # 

if __name__ == "__main__":
    main()
