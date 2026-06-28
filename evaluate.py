import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import argparse
import os

def evaluate_single_sample(garment_img_path, output_img_path, prompt, save_path="demo_evaluation_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 65)
    print("  FitAI Virtual Try-On: Quick Demo Sample Alignment Evaluation")
    print(f"  Device:            {device}")
    print(f"  Garment Reference: {garment_img_path}")
    print(f"  Try-On Output:     {output_img_path}")
    print("=" * 65)
    
    print("Loading CLIP ViT-B/32 model for evaluation...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    try:
        garment_img = Image.open(garment_img_path).convert("RGB")
        output_img = Image.open(output_img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # 1. Garment Feature Alignment via CLIP Cosine Similarity
    inputs = processor(images=[garment_img, output_img], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds") and getattr(image_features, "image_embeds") is not None:
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif isinstance(image_features, tuple):
                image_features = image_features[1] if len(image_features) > 1 else image_features[0]

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
    garment_feat = image_features[0]
    output_feat = image_features[1]
    
    garment_alignment_score = torch.nn.functional.cosine_similarity(garment_feat, output_feat, dim=0).item()
    
    # 2. Text-to-Image Prompt Alignment
    inputs = processor(text=[prompt], images=output_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
        else:
            image_embeds = outputs.vision_model_output.pooler_output
            text_model_out = outputs.text_model_output.pooler_output
            if image_embeds.shape[-1] != model.visual_projection.out_features:
                image_embeds = model.visual_projection(image_embeds)
            if text_model_out.shape[-1] != model.text_projection.out_features:
                text_embeds = model.text_projection(text_model_out)
            else:
                text_embeds = text_model_out

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        text_alignment_score = torch.nn.functional.cosine_similarity(image_embeds[0], text_embeds[0], dim=0).item()

    results = {
        "evaluation_mode": "Single Demo Sample Alignment (Qualitative Quick-Check)",
        "description": "Calculates CLIP Cosine Similarity between a reference garment and a generated try-on demo image.",
        "sample_details": {
            "garment_reference": garment_img_path,
            "tryon_output": output_img_path,
            "prompt": prompt
        },
        "demo_sample_metrics": {
            "Garment_Alignment_Score_CLIP": round(garment_alignment_score, 4),
            "Text_Alignment_Score_CLIP": round(text_alignment_score, 4)
        },
        "full_benchmark_reference": {
            "description": "For official benchmark results evaluated across all 2,032 Zalando VITON test pairs (SSIM: 0.8523, CLIP: 0.7427, FID: 8.74, Latency: 91.49ms), please refer to evaluation/metrics.json",
            "dataset": "Zalando VITON Test Split (2,032 pairs)",
            "full_test_clip_garment_alignment": 0.7427,
            "full_test_ssim": 0.8523,
            "full_test_fid": 8.74,
            "benchmark_file": "evaluation/metrics.json"
        }
    }
    
    print("\n" + "=" * 65)
    print("  Single Demo Sample Results:")
    print("=" * 65)
    print(f"  - CLIP Garment Alignment: {results['demo_sample_metrics']['Garment_Alignment_Score_CLIP']}")
    print(f"  - CLIP Text Alignment:    {results['demo_sample_metrics']['Text_Alignment_Score_CLIP']}")
    print("=" * 65)
    print(f"  Note: Full dataset benchmark (2,032 test pairs) is available in:")
    print(f"        -> 'evaluation/metrics.json' (Avg CLIP: 0.7427, SSIM: 0.8523, FID: 8.74)")
    print("=" * 65)
        
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Saved demo evaluation to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick CLIP Alignment Evaluator for a single try-on image pair")
    parser.add_argument("--garment", type=str, default="images/cloth.jpg", help="Path to reference garment image")
    parser.add_argument("--output", type=str, default="outputs/output_tryon_demo.png", help="Path to generated try-on result")
    parser.add_argument("--prompt", type=str, default="A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution", help="Conditioning prompt")
    parser.add_argument("--save_path", type=str, default="demo_evaluation_results.json", help="Path to save evaluation JSON")
    
    args = parser.parse_args()
    if os.path.exists(args.output):
        evaluate_single_sample(args.garment, args.output, args.prompt, args.save_path)
    else:
        print(f"Error: Output image '{args.output}' does not exist. Run inference first via 'python inference.py'.")
