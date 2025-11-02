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
