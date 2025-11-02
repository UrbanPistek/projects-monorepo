# Fine Tuning Models in Pytorch & Using ONNX for Inference in Python & Rust

## [1] Overview

I want to mostly focus on the Inference side of things and how ONNX can be leveraged, but to get there I want to cover fine tuning a model to serve as a starting point and as a working example. Then I will dive into the specifics of how to use ONNX for inference in both rust and python, also covering how some of the pre-processing transforms can be replicated without having to rely on the transforms used in PyTorch. 

I give some quick hints as to why ONNX can be advantageous - which will be expanded on more. 

1. ONNX Runtime is optimized for inference - uses techniques like Kernel fusion & Graph optimization
2. ONNX is an open standard for model representation
3. ONNX models and run-times tend to be lighter than PyTorch

### Dataset

Dataset: [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

I decided to use this image dataset because I thought it was a little more interesting and wanted to work with image based data for this example. If you wish to follow along, download it from the link above.

### Model Selection & Data Preparation

So why tuning a existing model here? The biggest reasons is efficiency - both in terms of cost and time in addition to getting faster convergence. By taking a more general existing model its weights are already adjusted to a point where it has a strong foundational knowledge in that general task. So tuning it requires much less training time to have adjust to your specific task. There are cases where training from scratch makes more sense, however I do think in a lot of scenarios fine tuning just makes a lot of sense, and can give you great results with much less effort - overall more efficient. 

> [Introduction to Fine-Tuning in Machine Learning](https://www.oracle.com/ca-en/artificial-intelligence/fine-tuning/)

So we've selected a image dataset to use for this example and the general problem we are trying to address here is a classification task. So we'll want to select a image classification foundation model to use for our base, from [PyTorch Vision](https://docs.PyTorch.org/vision/main/models.html), there are quite a few models to select from for this problem. Naturally, from the table there are so many models to choose from I was not sure what I wanted to select - model selection can be very nuanced but for this sample, I wanted to use a rough criteria to select something decent. 

I started by using some basic web-scraping to pull the table for image classification pre-trained models from [here](https://docs.PyTorch.org/vision/main/models.html#table-of-all-available-classification-weights) - if you wish to see how I did this see this Article [Visual Compairsons of PyTorch Pre-trained Models: Resnet, DeepLabV3, MViT & Others](https://medium.com/@urban.pistek/visual-compairsons-of-PyTorch-pre-trained-models-resnet-deeplabv3-mvit-others-5f833606776a) I made outlining all the comparisons, methods and links to the raw data used to make all the visuals. From this article I am going to reference one image I made to show how I chose a model to fine-tune.

![Image_Classification_Models_(IMAGENET1K_V1)_Params_(M)_v_Acc@1](./media/Image_Classification_Models_(IMAGENET1K_V1)_Params_(M)_v_Acc@1.png)

Using this, and also comparing the accuracy to the floating point operations, I chose a model that had a higher accuracy then most but still did not skew too high for the number of model parameters and GFLOPS - balancing accuracy and performance. I also chose a lower parameter model to prototype and build with as I was initially testing things out to be able to execute the runtime quickly as I was verifying things while developing. 

To start, I used the following models to fine tune:

```
RegNet_Y_16GF
RegNet_X_400MF
```

#### Loading the Model

A convenient package to get all the layers and some info on a model is [torchinfo](https://pypi.org/project/torchinfo/), using this we can load the model and print out some info as follows: 

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

Following PyTorch convension, a class is created to maange the loading of the data. Some additional steps are that a label encoder is created to take the string based labels and map them to a numerical form; the mappings are also exported to a JSON so we can easily review the mappings as well and use them in other applications. 

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

One of the first things to adjust - when needed - while tuning a model is some of the layers, often the final layer. Since the model was initially trained on a different dataset, with a different number of output classes the final linear layer needs to be modified to output the same number of classes. Additionally, the optimizer and criterion are defined. 

```python
def load_model(num_classes, base_model: models.RegNet):
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.0001)

    return base_model, criterion, optimizer
```

From here the the steps are the same as with training a model, define the training and validation steps and run the training loop. I will put my functions for that here for reference. 

**Training:**

```python
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
```

**Validation:**

```python
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
    model.eval()

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
```

**Main Loop:**

```python
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
```

## [3] Exporting to ONNX

There are a few different ways to save and export a PyTorch model but I will focus on using the [ONNX](https://onnx.ai/) format to export - so why ONNX? In short it is a open format for machine learning models, thus allowing better interoperability. Also, it offers a decent amount of optimizations and inference capabilities that can be advantageous - these I will explore more later. 

```python
# Save model as onnx
example_inputs = (torch.randn(model_input_size))
onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
model_onnx_save_path = f"./models/{base_model._get_name()}_{model_params_string}_tuned.onnx"
onnx_program.save(model_onnx_save_path)
print(f"ONNX Model saved to: {model_onnx_save_path}")
```

In the practical lens, in python the ONNX runtime provides a much smaller package for running inference, and also can provide much better performance.

Using `du -sh venv/lib/python3.10/site-packages/* | sort -h` we can see the package sizes in my local environment, we can see the difference is quite large between onnx and PyTorch. For PyTorch, currently there is not a separate runtime you can install, you'll need all of PyTorch unless you want to go through the manual process of striping down the package.

```
49M	venv/lib/python3.10/site-packages/onnxruntime
1.6G	venv/lib/python3.10/site-packages/torch
```

To see the full tuning script, [see the source here](../tune_model.py).

## [4] ONNX Inference in Python

A full sample script for running ONNX inference in python is [here](../onnx_inference.py). In general I'd say that the more nauced part is just replicating the transforms without using the PyTorch library. Luckily for this specific case the actual transforms where not anything too complicated or advanced. 

The transforms: 

```python
# Need to perform the base image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
```

Can be written using just numpy as follows:

```python
def center_crop_numpy(image_array: np.ndarray, crop_size: int) -> np.ndarray:

    # Get image shape
    h, w, c = image_array.shape
    
    # Calculate crop coordinates
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2

    # Perform the crop
    return image_array[start_y:start_y + crop_size, start_x:start_x + crop_size, :]

image = image.resize((256, 256), Image.LANCZOS) # equivalent to transforms.Resize(256)
image = np.asarray(image) # convert to numpy array
image = center_crop_numpy(image, 224) # equivalent to transforms.CenterCrop(224)

# Normalize 
image = np.transpose(image / 255.0, (2, 0, 1)) # equivalent to transforms.Normalize()
mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
image = (image - mean) / std

image = image.astype(np.float32) # equivalent to transforms.ToTensor(), needs to be a float for ORT 
image = np.expand_dims(image, axis=0) # Add batch dimension
```

The rest of the core ONNX inference steps are as follows:

```python
# Load the onnx model with the runtime session
onnx_path = Path(ONNX_MODEL_PATH).resolve()
ort_session = onnxruntime.InferenceSession(onnx_path)

# Format for onnxruntime
onnx_runtime_input = {
    "x": image
}
onnx_runtime_outputs = ort_session.run(None, onnx_runtime_input)
logits = onnx_runtime_outputs[0]

# Apply softmax to get probabilities
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Apply argmax to get predicted class
predicted_class = np.argmax(logits, axis=1)[0]
confidence = probabilities[0][predicted_class]
```

The full script is as follows:

```python
import json
import time
import onnxruntime 
import numpy as np
from PIL import Image
from pathlib import Path

MODEL_NAME = "RegNet_x_400mf"
SAMPLE_IMAGE_PATH = "data/test/Image_7.jpg"
ONNX_MODEL_PATH = f"models/{MODEL_NAME}_tuned.onnx"
LABELS_MAPPING_PATH = "models/labels_mapping.json"

def center_crop_numpy(image_array: np.ndarray, crop_size: int) -> np.ndarray:

    # Get image shape
    h, w, c = image_array.shape
    
    # Calculate crop coordinates
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2

    # Perform the crop
    return image_array[start_y:start_y + crop_size, start_x:start_x + crop_size, :]

def load_labels_mapping():

    # Load from a JSON file
    class_to_int = {}
    int_to_class = {}
    with open(LABELS_MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
        class_to_int = mapping['class_to_int']
        int_to_class = {int(k): v for k, v in mapping['int_to_class'].items()}

    return class_to_int, int_to_class

def main():

    ts = time.perf_counter()
    print(f"Running Inference on {MODEL_NAME}")

    # Use a sample image
    sample_img_path = Path(SAMPLE_IMAGE_PATH).resolve()
    image = Image.open(sample_img_path).convert("RGB")

    # Load the onnx model with the runtime session
    onnx_path = Path(ONNX_MODEL_PATH).resolve()
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Need to perform the base image transformations
    image = image.resize((256, 256), Image.LANCZOS) # equivalent to transforms.Resize(256)
    image = np.asarray(image) # convert to numpy array
    image = center_crop_numpy(image, 224) # equivalent to transforms.CenterCrop(224)
    
    # Normalize 
    image = np.transpose(image / 255.0, (2, 0, 1)) # equivalent to transforms.Normalize()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std

    image = image.astype(np.float32) # equivalent to transforms.ToTensor(), needs to be a float for ORT 
    image = np.expand_dims(image, axis=0) # Add batch dimension

    # Load labels mapping
    class_to_int, int_to_class = load_labels_mapping()

    # Format for onnxruntime
    onnx_runtime_input = {
        "x": image
    }
    # onnx_runtime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), [image])} # alternate format

    ts_inf = time.perf_counter()
    onnx_runtime_outputs = ort_session.run(None, onnx_runtime_input)
    logits = onnx_runtime_outputs[0]

    # Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Apply argmax to get predicted class
    predicted_class = np.argmax(logits, axis=1)[0]
    confidence = probabilities[0][predicted_class]
    te_inf = time.perf_counter()
    elapsed_ms_inf = (te_inf-ts_inf)*1000
    print(f"Inference Completed in {elapsed_ms_inf}ms")

    print("\n=== Inference Results ===")
    print(f"Predicted class index: {predicted_class}")
    print(f"Predicted class name: {int_to_class[predicted_class]}")
    print(f"Confidence: {confidence}")

    te = time.perf_counter()
    elapsed_ms = (te-ts)*1000
    print(f"Completed in {elapsed_ms}ms")

if __name__ == "__main__":
    main()
```

## [5] ONNX Inference in Rust

A full sample script for running ONNX inference in python is [here](../src/main.rs).

With rust there are a couple of function we need to define ourselves - that at least at the time of writing this I don't know of a numpy equivalent library that has these - regardless, they are pretty straightforward functions. 

```rust
pub fn argmax(array: &[f32]) -> usize {
    array
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

pub fn center_crop(image: &RgbImage, crop_size: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    
    let start_x = (width - crop_size) / 2;
    let start_y = (height - crop_size) / 2;
    
    let mut cropped = ImageBuffer::new(crop_size, crop_size);
    
    for y in 0..crop_size {
        for x in 0..crop_size {
            let src_x = start_x + x;
            let src_y = start_y + y;
            cropped.put_pixel(x, y, *image.get_pixel(src_x, src_y));
        }
    }
    
    cropped
}
```

Then, we can create a pre-processing function that replicates the transform functionality from before. 

```rust
pub fn preprocess_image(image_path: &str) -> Result<Array4<f32>> {
    // Load and convert image to RGB
    let image = image::open(image_path)?;
    let image = image.to_rgb8();
    
    // Resize to 256x256 (equivalent to transforms.Resize(256))
    let image = image::imageops::resize(&image, 256, 256, image::imageops::FilterType::Lanczos3);
    
    // Center crop to 224x224 (equivalent to transforms.CenterCrop(224))
    let image = center_crop(&image, 224);
    
    // Convert to ndarray and normalize
    let mut array = Array4::<f32>::zeros((1, 3, 224, 224));
    
    // ImageNet normalization values
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    
    for y in 0..224 {
        for x in 0..224 {
            let pixel = image.get_pixel(x, y);
            
            // Convert to [0, 1] range and apply ImageNet normalization
            for c in 0..3 {
                let normalized = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                array[[0, c, y as usize, x as usize]] = normalized;
            }
        }
    }
    
    Ok(array)
}
```

With this the key inference steps look something like this: 

```rust
let onnx_model_path: String = format!("/home/urban/urban/projects/projects-monorepo/ml-onnx/models/{}_tuned.onnx", MODEL_NAME);
let mut model = Session::builder()?
.with_optimization_level(GraphOptimizationLevel::Level3)?
.with_intra_threads(4)?
.commit_from_file(onnx_model_path)?;

// Load and preprocess the image
let array_input = onnx_inference::common::preprocess_image(SAMPLE_IMAGE_PATH).unwrap();

// Run inference
let outputs: SessionOutputs = model.run(ort::inputs!["x" => TensorRef::from_array_view(&array_input)?])?;

// Extract predictions into usable format
let predictions = outputs[0].try_extract_array::<f32>()?;
let predictions_slice = predictions.as_slice().unwrap();

// Find the predicted class
let predicted_class_idx = onnx_inference::common::argmax(predictions_slice);

// Calculate probabilities using softmax
let probabilities = onnx_inference::common::softmax(predictions_slice);
let confidence = probabilities[predicted_class_idx];
```

The full script is can be something like this: 

```rust
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::{value::TensorRef};
use std::time::Instant;

// Constants
const MODEL_NAME: &str = "RegNet_x_400mf";
const SAMPLE_IMAGE_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/data/test/Image_7.jpg";
const LABELS_MAPPING_PATH: &str = "/home/urban/urban/projects/projects-monorepo/ml-onnx/models/labels_mapping.json";

// Struct to hold the label mappings
#[derive(Debug, Serialize, Deserialize)]
struct LabelMapping {
    class_to_int: HashMap<String, i32>,
    int_to_class: HashMap<String, String>,
}

// Util functions
fn load_labels_mapping(path: &str) -> Result<HashMap<i32, String>> {
    let file_content = std::fs::read_to_string(path)?;
    let mapping: LabelMapping = serde_json::from_str(&file_content)?;
    
    // Convert string keys to integers for int_to_class mapping
    let mut int_to_class = HashMap::new();
    for (key, value) in mapping.int_to_class {
        let int_key: i32 = key.parse()?;
        int_to_class.insert(int_key, value);
    }
    
    Ok(int_to_class)
}

fn main() -> Result<()> {

    // Start timer
    let start = Instant::now();
    println!("Running Inference on: {:?}", MODEL_NAME);

    let onnx_model_path: String = format!("/home/urban/urban/projects/projects-monorepo/ml-onnx/models/{}_tuned.onnx", MODEL_NAME);
    let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file(onnx_model_path)?;
    
    // Load and preprocess the image
    let array_input = onnx_inference::common::preprocess_image(SAMPLE_IMAGE_PATH).unwrap();

    // Run inference
    let start_inf = Instant::now();
    let outputs: SessionOutputs = model.run(ort::inputs!["x" => TensorRef::from_array_view(&array_input)?])?;

    // Extract predictions into usable format
    let predictions = outputs[0].try_extract_array::<f32>()?;
    let predictions_slice = predictions.as_slice().unwrap();
    
    // Find the predicted class
    let predicted_class_idx = onnx_inference::common::argmax(predictions_slice);
    
    // Calculate probabilities using softmax
    let probabilities = onnx_inference::common::softmax(predictions_slice);
    let confidence = probabilities[predicted_class_idx];
    let duration_inf = start_inf.elapsed();
    println!("Inference done in: {:?}", duration_inf);
    
    // Load labels mapping
    let int_to_class = load_labels_mapping(LABELS_MAPPING_PATH)?;
    
    // Get the predicted class name
    let default = String::from("Unknown");
    let predicted_class_name = int_to_class
        .get(&(predicted_class_idx as i32))
        .unwrap_or(&default);
    
    // Print results
    println!("\n=== Inference Results ===");
    println!("Predicted class index: {}", predicted_class_idx);
    println!("Predicted class name: {}", predicted_class_name);
    println!("Confidence: {:.6}", confidence);

    let duration = start.elapsed();
    println!("Completed in: {:?}", duration);
    
    Ok(())
}
```

## [6] Summary

I hope this gives helpful insight into tuning existing models, then after training exporting to ONNX formatting and seeing how to run a ONNX model in both python and rust for inference. I have already covered some of the benefits of using the ONNX runtime, but I have not elaborated on any benefits of using rust - that is something I'll dive into deeper detail for another article. In short, other then all the benefits you have heard with using Rust, here is a small practical example. 

Even though it is known that rust executable can be larger then other languages - trading off a larger size for other benefits - we can check the size of the executable with: `ls -l target/release/onnx-inference`

> -rwxrwxr-x 2 urban urban 37147936 Sep 17 07:28 target/release/onnx-inference

Which is 37MB - already smaller then the `onnxruntime` python library just by itself; in this case the executable is all you need to run. So, there is a glimpse into a aspect of the efficiencies you gain just right there. 

For performance & further efficiencies - that is another deep dive. 

## [7] References

1. https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
2. https://www.oracle.com/ca-en/artificial-intelligence/fine-tuning/
