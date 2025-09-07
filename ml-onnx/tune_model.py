import os
import gc
import time
import json
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import RegNet_Y_16GF_Weights, RegNet_X_400MF_Weights
from torchinfo import summary
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
from pathlib import Path
from typing import Any

# Training
LABELS_CSV_PATH = "./data/train_set.csv"
IMAGES_PATH = "./data/train"


class TrainingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data: pd.DataFrame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Clean up label formatting & get classes
        self.data["label"] = self.data["label"].str.title() # for cleaner formatting
        self.classes = np.unique(self.data["label"].values).tolist()

        # Create and fit label encoder since classes are
        # initially provided as strings
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        # Convert string labels to numeric
        self.data["label_encoded"] = self.label_encoder.transform(self.data["label"])

        # Save labels
        self.encoded_labels: np.ndarray = self.label_encoder.fit_transform(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return numeric label as tensor
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        return image, label
    
    def export_labels(self):

        # Create bidirectional mapping
        encoded_labels = self.encoded_labels.tolist() # need to convert to JSON compatible format
        class_to_int = dict(zip(self.classes, encoded_labels))
        int_to_class = dict(zip(encoded_labels, self.classes))

        # Save to json to use 
        with open('./models/labels_mapping.json', 'w') as f:
            json.dump({
                'class_to_int': class_to_int,
                'int_to_class': {str(k): v for k, v in int_to_class.items()}  # JSON keys must be strings
            }, f)


def load_model(num_classes, base_model: models.RegNet):
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.0001)

    return base_model, criterion, optimizer


def load_transforms():

    train_transform = transforms.Compose([
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
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # using 224 to start since that is what the model was originally trained on
                
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])

    return train_transform, val_transform


def load_data():
    
    # Define transforms 
    # Transforms apply each time the image is loaded
    # Effectively applying a new transform to each image during each new epoch
    batch_size = 32
    split_ratio = 0.8
    train_transform, val_transform = load_transforms()

    # Convert to absolute paths using pathlib
    labels_abs = Path(LABELS_CSV_PATH).resolve()
    images_abs = Path(IMAGES_PATH).resolve()

    # Create dataset and dataloader
    # Dataset is already split, so no need to perform a random split
    # Create full dataset with training transforms initially
    full_dataset = TrainingDataset(
        csv_file=labels_abs,
        img_dir=images_abs,
        transform=train_transform
    )

    # Save labels for later reference
    full_dataset.export_labels()
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(split_ratio * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Update validation dataset transforms
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    classes = full_dataset.classes

    return train_loader, val_loader, classes


def epoch_train(
        model: models.RegNet, 
        train_dataloader: DataLoader[Any], 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int
    ):

    # Training params
    predictions = []
    labels = []
    train_loss = 0
    ts = time.perf_counter()

    # Initiate training
    model.train()

    # Main training loop
    batch_idx = 0
    for images, targets in train_dataloader:

        # Loader tensors
        images: torch.Tensor = images.to(device)
        targets: torch.Tensor = targets.to(device)

        # Train
        optimizer.zero_grad()
        outputs: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate training metrics
        train_loss += loss.item()
        pred = torch.argmax(outputs, dim=-1).tolist()
        predictions.extend(pred)
        labels.extend(targets.tolist())
        batch_idx += 1

        # Explicity de-allocate memory
        del outputs, loss, pred

    # Calculate more metrics for tracking
    train_loss = train_loss / len(train_dataloader)
    te = time.perf_counter()
    epoch_time = round(te-ts, 3)
    train_acc = accuracy_score(labels, predictions)

    print(f"[Train Epoch]: {epoch}, Accuracy: {train_acc}, Loss: {train_loss}, Time: {epoch_time}s")
    epoch_metrics = {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "train_time": epoch_time
    }

    return epoch_metrics


def epoch_val(
        model: models.RegNet, 
        val_dataloader: DataLoader[Any], 
        criterion: nn.Module, 
        device: torch.device,
        epoch: int
    ):

    # Training params
    predictions = []
    labels = []
    val_loss = 0
    ts = time.perf_counter()

    # Initiate training
    model.train()

    # Main training loop
    batch_idx = 0
    for images, targets in val_dataloader:

        # Loader tensors
        with torch.no_grad():
            images: torch.Tensor = images.to(device)
            targets: torch.Tensor = targets.to(device)

            # Val
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, targets)

            # Calculate training metrics
            val_loss += loss.item()
            pred = torch.argmax(outputs, dim=-1).tolist()
            predictions.extend(pred)
            labels.extend(targets.tolist())
            batch_idx += 1

            # Explicity de-allocate memory
            del outputs, loss, pred

    # Calculate more metrics for tracking
    val_loss = val_loss / len(val_dataloader)
    te = time.perf_counter()
    epoch_time = round(te-ts, 3)
    val_acc = accuracy_score(labels, predictions)

    print(f"[Validation Epoch]: {epoch}, Accuracy: {val_acc}, Loss: {val_loss}, Time: {epoch_time}s")
    epoch_metrics = {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "val_time": epoch_time
    }

    return epoch_metrics


def training_loop(
        num_epochs: int, 
        model: models.RegNet, 
        train_dataloader: DataLoader[Any], 
        val_dataloader: DataLoader[Any],
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> pd.DataFrame:

    ts = time.perf_counter()
    metrics = []
    for epoch in range(num_epochs):
        train_epoch_metrics = epoch_train(model, train_dataloader, criterion, optimizer, device, epoch)
        val_epoch_metrics = epoch_val(model, val_dataloader, criterion, device, epoch)

        # Merge into combined dictionary, python 3.9+ method
        combined = train_epoch_metrics | val_epoch_metrics
        metrics.append(combined)

    te = time.perf_counter()
    elapsed = round(te-ts, 3)
    print(f"Training completed in: {elapsed}s")

    # Create a metrics dataframe to view the results later
    metrics_df = pd.DataFrame(metrics)

    return metrics_df


def main():

    # Show original model info
    base_model = models.regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2)
    # base_model = models.regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V2) # Smaller model for quicker testing

    # Print and show summmary of the model architecture
    model_input_size = (1, 3, 224, 224)
    summary(base_model, input_size=model_input_size, depth=3)
    
    # Load model
    ts = time.perf_counter()
    num_classes = 75
    device = torch.device("cpu") # using CPU to start
    model, criterion, optimizer = load_model(num_classes, base_model)
    te = time.perf_counter()
    elapsed = round(te-ts, 3)
    print(f"model loaded in: {elapsed}s")

    # Load data
    train_loader, val_loader, classes = load_data()
    print(f"train set size: {len(train_loader.dataset)}")
    print(f"val set size: {len(val_loader.dataset)}")
    print("classes: ")
    pprint(classes)

    # Train model
    num_epochs = 3
    metrics_df = training_loop(
        num_epochs,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device
    )

    print(metrics_df.head(num_epochs))

    # Save model as .pth
    model.to("cpu")
    model_params_string = "y_16gf"
    model_save_path = f"./models/{base_model._get_name()}_{model_params_string}_tuned.pth"
    torch.save(model, model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Save model as onnx
    example_inputs = (torch.randn(model_input_size))
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    model_onnx_save_path = f"./models/{base_model._get_name()}_{model_params_string}_tuned.onnx"
    onnx_program.save(model_onnx_save_path)
    print(f"ONNX Model saved to: {model_onnx_save_path}")

if __name__ == "__main__":
    main()
