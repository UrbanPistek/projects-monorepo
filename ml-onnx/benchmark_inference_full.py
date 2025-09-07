import os
import time
import json
import glob
import torch
import random
import onnxruntime as ort
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from typing import List, Dict, Any, Tuple

MODEL_NAME = "RegNet_y_16gf"

class ModelBenchmark:

    def __init__(self, pytorch_model_path: str, onnx_model_path: str, labels_mapping_path: str):
        self.pytorch_model_path = pytorch_model_path
        self.onnx_model_path = onnx_model_path
        self.labels_mapping_path = labels_mapping_path
        
        # Load label mappings
        with open(labels_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.int_to_class = {int(k): v for k, v in self.label_mapping['int_to_class'].items()}
        
        # Initialize models
        self.pytorch_model = None
        self.onnx_session = None
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_pytorch_model(self) -> None:
        print("Loading PyTorch model...")
        ts = time.perf_counter()
        
        self.pytorch_model = torch.load(self.pytorch_model_path, map_location='cpu', weights_only=False)
        self.pytorch_model.eval()
        
        load_time = time.perf_counter() - ts
        print(f"PyTorch model loaded in {load_time:.4f}s")
    
    def load_onnx_model(self) -> None:
        print("Loading ONNX model...")
        ts = time.perf_counter()
        
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use CPU provider for fair comparison
        providers = ['CPUExecutionProvider']
        
        self.onnx_session = ort.InferenceSession(
            self.onnx_model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        load_time = time.perf_counter() - ts
        print(f"ONNX model loaded in {load_time:.4f}s")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:

        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def pytorch_inference(self, image_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:

        with torch.no_grad():
            outputs = self.pytorch_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return predicted_class, confidence, probabilities.numpy()
    
    def onnx_inference(self, image_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:

        # Convert to numpy for ONNX Runtime
        input_array = image_tensor.numpy()
        
        # Get input name from ONNX model
        input_name = self.onnx_session.get_inputs()[0].name
        
        # Run inference
        outputs = self.onnx_session.run(None, {input_name: input_array})
        logits = outputs[0]
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        predicted_class = np.argmax(logits, axis=1)[0]
        confidence = probabilities[0][predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def benchmark_batch_images(self, image_paths: List[str], num_runs: int = 10) -> Tuple[dict, pd.DataFrame]:

        print(f"\nBenchmarking batch of {len(image_paths)} images with {num_runs} runs...")
        all_results = []
        pytorch_times = []
        onnx_times = []

        for loop_num in range(num_runs):

            # Get shuffle list
            random.shuffle(image_paths)
            
            # Loop with tqdm
            for image_path in tqdm(image_paths, desc=f"Loop {loop_num + 1}"):
                image_name = Path(image_path).name

                # Time PyTorch inference
                ts = time.perf_counter()
                image_tensor = self.preprocess_image(image_path)
                pytorch_pred, pytorch_conf, _ = self.pytorch_inference(image_tensor)
                pytorch_time = (time.perf_counter() - ts)
                pytorch_times.append(pytorch_time * 1000) # convert to ms
                
                # Time ONNX inference
                ts = time.perf_counter()
                image_tensor = self.preprocess_image(image_path)
                onnx_pred, onnx_conf, _ = self.onnx_inference(image_tensor)
                onnx_time = (time.perf_counter() - ts)
                onnx_times.append(onnx_time * 1000) # convert to ms
                
                all_results.append({
                    'image': image_name,
                    'pytorch_pred': self.int_to_class[pytorch_pred],
                    'pytorch_conf': pytorch_conf,
                    'onnx_pred': self.int_to_class[onnx_pred],
                    'onnx_conf': onnx_conf,
                    'pytorch_time_ms': pytorch_time * 1000, # convert to ms
                    'onnx_time_ms': onnx_time * 1000, # convert to ms
                    'predictions_match': pytorch_pred == onnx_pred
                })
        
        batch_results = {
            'num_images': len(image_paths),
            'pytorch_avg_time_ms': np.mean(pytorch_times),
            'onnx_avg_time_ms': np.mean(onnx_times),
            'pytorch_total_time_ms': np.sum(pytorch_times),
            'onnx_total_time_ms': np.sum(onnx_times),
            'avg_speedup': np.mean(pytorch_times) / np.mean(onnx_times),
            'prediction_accuracy': sum(r['predictions_match'] for r in all_results) / len(all_results),
            'individual_results': all_results
        }
        results_df = pd.DataFrame(all_results)
        
        return batch_results, results_df
    
    def get_model_info(self) -> Dict[str, Any]:
        
        info = {}

        # PyTorch model info
        if self.pytorch_model:
            pytorch_size = os.path.getsize(self.pytorch_model_path) / (1024 * 1024)  # MB
            num_params = sum(p.numel() for p in self.pytorch_model.parameters())
            trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
            
            info['pytorch'] = {
                'model_size_mb': pytorch_size,
                'num_parameters': num_params,
                'trainable_parameters': trainable_params,
                'model_type': type(self.pytorch_model).__name__
            }
        
        # ONNX model info
        if self.onnx_session:
            onnx_size = os.path.getsize(self.onnx_model_path) / (1024 * 1024)  # MB
            
            info['onnx'] = {
                'model_size_mb': onnx_size,
                'providers': self.onnx_session.get_providers(),
                'input_shape': [input.shape for input in self.onnx_session.get_inputs()],
                'output_shape': [output.shape for output in self.onnx_session.get_outputs()]
            }
        
        return info
    
    def save_results(self, results: Dict[str, Any], output_path: str):

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        if 'pytorch' in results and 'onnx' in results:
            # Single image results
            pytorch_time = results['pytorch']['avg_inference_time_ms']
            onnx_time = results['onnx']['avg_inference_time_ms']
            speedup = results['onnx_speedup']
            
            print(f"PyTorch avg inference time: {pytorch_time:.3f}ms")
            print(f"ONNX avg inference time:    {onnx_time:.3f}ms")
            print(f"ONNX speedup:              {speedup:.2f}x")
            print(f"Predictions match:         {results['pytorch']['predicted_class'] == results['onnx']['predicted_class']}")
            
        elif 'num_images' in results:
            # Batch results
            pytorch_time = results['pytorch_avg_time_ms']
            onnx_time = results['onnx_avg_time_ms']
            speedup = results['avg_speedup']
            accuracy = results['prediction_accuracy']
            
            print(f"Number of images:          {results['num_images']}")
            print(f"PyTorch avg time:          {pytorch_time:.3f}ms")
            print(f"ONNX avg time:             {onnx_time:.3f}ms")
            print(f"Average speedup:           {speedup:.2f}x")
            print(f"Prediction accuracy:       {accuracy:.2%}")


def main():

    # Configuration
    pytorch_model_path = f"./models/{MODEL_NAME}_tuned.pth"
    onnx_model_path = f"./models/{MODEL_NAME}_tuned.onnx"
    labels_mapping_path = "./models/labels_mapping.json"
    images_path = "./data/test"
    num_runs = 1

    # Get full absolute paths
    pytorch_model_path_abs = Path(pytorch_model_path).resolve()
    onnx_model_path_abs = Path(onnx_model_path).resolve()
    labels_mapping_path_abs = Path(labels_mapping_path).resolve()
    images_path_abs = Path(images_path).resolve()
    test_images = glob.glob(f"{images_path_abs}/*.jpg")
    
    # Filter to only existing images
    test_images = [img for img in test_images if os.path.exists(img)]
    if not test_images:
        print("No test images found! Please update the test_images list with actual image paths.")
        return
    
    # Initialize benchmark
    benchmark = ModelBenchmark(pytorch_model_path_abs, onnx_model_path_abs, labels_mapping_path_abs)
    
    # Load models
    benchmark.load_pytorch_model()
    benchmark.load_onnx_model()
    
    # Get model information
    model_info = benchmark.get_model_info()
    print("\nModel Information:")
    print(json.dumps(model_info, indent=2, default=str))
    
    # Batch benchmark
    batch_results, results_df = benchmark.benchmark_batch_images(test_images, num_runs=num_runs)

    # Save / print resutls
    if not os.path.exists("./data/benchmarks"):
        os.makedirs("./data/benchmarks")
    
    benchmark.print_summary(batch_results)
    benchmark.save_results(batch_results, f"./data/benchmarks/{MODEL_NAME}_benchmark_batch_full.json")
    results_df.to_csv(f"./data/benchmarks/{MODEL_NAME}_benchmark_batch_full.csv")
    
    print("\nBenchmarking completed!")

if __name__ == "__main__":
    main()
