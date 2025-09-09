import json
import time
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms, models

MODEL_NAME = "RegNet_x_400mf"
SAMPLE_IMAGE_PATH = "data/test/Image_7.jpg"
PYTORCH_MODEL_PATH = f"models/{MODEL_NAME}_tuned.pth"
LABELS_MAPPING_PATH = "models/labels_mapping.json"

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
    model: models.RegNet = torch.load(PYTORCH_MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()

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

    # Load labels mapping
    class_to_int, int_to_class = load_labels_mapping()

    ts_inf = time.perf_counter()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
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
