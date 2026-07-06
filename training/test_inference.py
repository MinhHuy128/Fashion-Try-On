import torch
from pipelines.inference import FitAIInferencePipeline
from PIL import Image

def test_inference_mock():
    print("Testing SOTA Diffusion Inference Pipeline...")
    
    # Initialize pipeline
    pipeline = FitAIInferencePipeline(architecture="architecture_d", optimize_fp16=False)
    
    # Create mock inputs
    person_img = Image.new('RGB', (384, 512), color='blue')
    garment_img = Image.new('RGB', (384, 512), color='red')
    preservation_mask = Image.new('L', (384, 512), color=255) # White mask
    
    print("Running prediction (this will take a few seconds simulating Diffusion)...")
    result = pipeline.predict(person_img, garment_img, preservation_mask)
    
    print(f"Prediction successful! Latency: {result['latency_ms']:.2f} ms")
    print(f"Output Image Size: {result['output_image'].size}")
    
if __name__ == "__main__":
    test_inference_mock()
