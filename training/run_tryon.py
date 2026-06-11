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

def test_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    
    try:
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipeline.set_ip_adapter_scale(args.ip_scale)
    except Exception as e:
        print(e)
    
    try:
        person_img = Image.open(args.person).convert("RGB").resize((512, 512))
        garment_img = Image.open(args.garment).convert("RGB").resize((224, 224))
        
        if args.mask.lower() == "auto":
            mask_img = generate_auto_mask(person_img, device).resize((512, 512))
        else:
            mask_img = Image.open(args.mask).convert("L").resize((512, 512))
        
        output = pipeline(
            prompt=args.prompt,
            image=person_img,
            mask_image=mask_img,
            ip_adapter_image=garment_img,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        output_path = "output_tryon_demo.png"
        output.save(output_path)
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, default="data/ACGPN_raw/test_img/000001_0.jpg")
    parser.add_argument("--garment", type=str, default="data/ACGPN_raw/test_color/000001_1.jpg")
    parser.add_argument("--mask", type=str, default="auto")
    parser.add_argument("--prompt", type=str, default="A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution")
    parser.add_argument("--ip_scale", type=float, default=1.0)
    
    args = parser.parse_args()
    test_inference(args)
