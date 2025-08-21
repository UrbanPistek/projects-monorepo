import json
import onnxruntime 
import numpy as np
from PIL import Image
from pathlib import Path

SAMPLE_IMAGE_PATH = "data/test/Image_7.jpg"
ONNX_MODEL_PATH = "models/RegNet_tuned.onnx"
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
    image = image.astype(np.float32) # equivalent to transforms.ToTensor()
    image = np.transpose(image / 255.0, (2, 0, 1)) # equivalent to transforms.Normalize()
    image = (image - 0.5) / 0.5
    image = np.expand_dims(image, axis=0) # Add batch dimension

    # Load labels mapping
    class_to_int, int_to_class = load_labels_mapping()

    # Format for onnxruntime
    onnx_runtime_input = {
        "x": image
    }
    # onnx_runtime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), [image])} # alternate format
    print(onnx_runtime_input)

    onnx_runtime_outputs = ort_session.run(None, onnx_runtime_input)
    res = np.argmax(onnx_runtime_outputs[0])
    print(res, int_to_class[res])

if __name__ == "__main__":
    main()
