import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
import argparse
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

def generate_auto_mask(person_img, device):
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
    
    inputs = processor(images=person_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=person_img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    mask_np = (pred_seg == 4).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_np, mode='L')
    return mask_img

def main():
    parser = argparse.ArgumentParser(description="FitAI Virtual Try-On Inference")
    parser.add_argument("--person", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment", type=str, required=True, help="Path to garment image")
    parser.add_argument("--mask", type=str, default="auto", help="Mask image path or 'auto'")
    parser.add_argument("--prompt", type=str, default="A high quality, photorealistic image of a model wearing this exact shirt, perfect texture, natural lighting", help="Text prompt")
    parser.add_argument("--ip_scale", type=float, default=1.0, help="IP-Adapter image scale")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA weights directory")
    parser.add_argument("--output", type=str, default="outputs/output_tryon_demo.png", help="Output file path")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Stable Diffusion Inpainting pipeline on {device}...")
    
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    
    # Load IP-Adapter
    try:
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipeline.set_ip_adapter_scale(args.ip_scale)
    except Exception as e:
        print(f"Note on IP-Adapter: {e}")
        
    # Check and load custom trained LoRA weights
    lora_dir = args.checkpoint or "training/models/checkpoints"
    if os.path.exists(lora_dir):
        if os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(lora_dir, "adapter_model.bin")):
            try:
                pipeline.load_lora_weights(lora_dir)
                print(f"Loaded LoRA weights from {lora_dir}")
            except Exception as e:
                print(f"Warning loading LoRA: {e}")
    
    try:
        print(f"Processing person image: {args.person}")
        print(f"Processing garment image: {args.garment}")
        person_img = Image.open(args.person).convert("RGB").resize((512, 512))
        garment_img = Image.open(args.garment).convert("RGB").resize((224, 224))
        
        if args.mask.lower() == "auto":
            print("Auto-generating body mask with Segformer...")
            mask_img = generate_auto_mask(person_img, device).resize((512, 512))
        else:
            mask_img = Image.open(args.mask).convert("L").resize((512, 512))
        
        print("Generating try-on image...")
        output = pipeline(
            prompt=args.prompt,
            image=person_img,
            mask_image=mask_img,
            ip_adapter_image=garment_img,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        output.save(args.output)
        print(f"[SUCCESS] Saved output image to '{args.output}'")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
