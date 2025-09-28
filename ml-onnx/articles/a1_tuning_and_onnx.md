# Fine Tuning Models in Pytorch & Using ONNX for Inference in Python & Rust

## [1] Overview

I want to mostly focus on the Inference side of things and how ONNX can be leveraged, but to get there I want to cover fine tuning a model to serve as a starting point and as a working example. Then I will dive into the specifics of how to use ONNX for inference in both rust and python, also covering how some of the pre-processing transforms can be replicated without having to rely on the transforms used in pytorch. 

### Dataset

Dataset: [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

I decided to use this image dataset because I thought it was a little more interesting and wanted to work with image based data for this example. If you wish to follow along, download it from the link above.

### Model Selection & Data Preparation

So why tuning a existing model here? The biggest reasons is efficiency - both in terms of cost and time in addition to getting faster convergence. By taking a more general existing model its weights are already adjusted to a point where it has a strong foundational knowledge in that general task. So tuning it requires much less training time to have adjust to your specific task. There are cases where training from sratch makes more sense, however I do think in a lot of senarios fine tuning just makes a lot of sense, and can give you great results with much less effort - overall more efficient. 

> [Introduction to Fine-Tuning in Machine Learning](https://www.oracle.com/ca-en/artificial-intelligence/fine-tuning/)

So we've selected a image dataset to use for this example and the general problem we are trying to address here is a classification task. So we'll want to select a image classification foundation model to use for our base, from [PyTorch Vision](https://docs.pytorch.org/vision/main/models.html), there are quite a few models to select from for this problem. Naturally, from the table there are so many models to choose from I was not sure what I wanted to select - model selection can be very nuanced but for this sample, I wanted to use a rough critieria to select something decent. 

I started by using some basic web-scraping to pull the table for image classification pre-trained models from [here](https://docs.pytorch.org/vision/main/models.html#table-of-all-available-classification-weights) - if you wish to see how I did this see this Article [](https://medium.com/@urban.pistek/visual-compairsons-of-pytorch-pre-trained-models-resnet-deeplabv3-mvit-others-5f833606776a) I made outlining all the comparisons, methods and links to the raw data used to make all the visuals. From this article I am going to reference one image I made to show how I chose a model to fine-tune.

![Image_Classification_Models_(IMAGENET1K_V1)_Params_(M)_v_Acc@1](./media/Image_Classification_Models_(IMAGENET1K_V1)_Params_(M)_v_Acc@1.png)

Using this, and also comparing the accuracy to the floating point operations, I chose a model that had a higher accuracy then most but still did not skew too high for the number of model parameters and GFLOPS - balancing accuracy and performance. I also chose a low params model to prototype and build with as I was initially testing things out to be able to execute the runtime quickly as I was verifying things while developing. 

To start, I used the following models to fine tune:

```
RegNet_Y_16GF
RegNet_X_400MF
```

#### Loading the Model

A convienant package to get all the layers and some info on a model is [torchinfo](https://pypi.org/project/torchinfo/), using this we can load the model and print out some info as follows: 

```python
from torchvision import models
from torchvision.models import RegNet_X_400MF_Weights
from torchinfo import summary

base_model = models.regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V2)

# Print and show summmary of the model architecture
model_input_size = (1, 3, 224, 224)
summary(base_model, input_size=model_input_size, depth=3)
```

#### Loading the Data

Following PyTorch convension, a class is created to maange the loading of the data. Some additional steps are that a label encoder is created to take the string based labels and map them to a numerical form; the mappings are also exported to a json so we can easily review the mappings as well and use them in other applications. 

```python
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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
```

Next, the transforms are defined. Note that the train transforms have an additional step to alter some of the training inputs, yet the validation does not required these transforms. 

```python
from torchvision import transforms

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
```

With these, we can combine these steps to load the csv's, split the data and define the data loader. For this dataset, there is a csv that maps the filename to the label, and then the directory of all the images - we'll need both.

```python
from torch.utils.data import DataLoader, random_split
from pathlib import Path

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
```

## [2] Model Tuning

One of the first things to adjust - when needed - while tuning a model is some of the layers, often the final layer. Since the model was initially trained on a different dataset, with a different number of output classes

## [3] Exporting to ONNX

## [4] ONNX Inference in Python

## [5] ONNX Inference in Rust

## [6] Summary

## [7] References

1. https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
2. https://www.oracle.com/ca-en/artificial-intelligence/fine-tuning/
