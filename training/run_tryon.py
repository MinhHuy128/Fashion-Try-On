import argparse
from PIL import Image
import os
import sys

# Ensure training directory and parent directory are in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if TRAINING_DIR not in sys.path:
    sys.path.append(TRAINING_DIR)

from pipelines.inference import FitAIInferencePipeline

def main():
    parser = argparse.ArgumentParser(description="FitAI Virtual Try-On CLI Runner")
    parser.add_argument("--person", type=str, default="images/person.jpg", help="Path to person image")
    parser.add_argument("--garment", type=str, default="images/cloth.jpg", help="Path to garment image")
    parser.add_argument("--prompt", type=str, default="A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution")
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="output_tryon_demo.png")

    args = parser.parse_args()

    if not os.path.exists(args.person) or not os.path.exists(args.garment):
        print(f"Error: Input images not found ({args.person}, {args.garment})")
        return

    print("Initializing FitAI Inference Pipeline...")
    pipeline = FitAIInferencePipeline()

    print("Running inference...")
    person_img = Image.open(args.person).convert("RGB")
    garment_img = Image.open(args.garment).convert("RGB")

    result = pipeline.predict(person_img, garment_img, prompt=args.prompt, ip_scale=args.ip_scale)
    
    result["output_image"].save(args.output)
    print(f"[SUCCESS] Saved try-on result to '{args.output}' (Latency: {result['latency_ms']:.2f} ms)")

if __name__ == "__main__":
    main()

