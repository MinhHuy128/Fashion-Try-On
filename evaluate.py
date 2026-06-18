import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import argparse
import os

def evaluate(garment_img_path, output_img_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading CLIP model for evaluation...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    try:
        garment_img = Image.open(garment_img_path).convert("RGB")
        output_img = Image.open(output_img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
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
    
    inputs = processor(text=[prompt], images=output_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
        else:
            image_embeds = outputs.vision_model_output.pooler_output
            text_embeds = outputs.text_model_output.pooler_output
            if image_embeds.shape[-1] != model.visual_projection.out_features:
                image_embeds = model.visual_projection(image_embeds)
            if text_embeds.shape[-1] != model.text_projection.out_features:
                text_embeds = model.text_projection(text_embeds)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        text_alignment_score = torch.nn.functional.cosine_similarity(image_embeds[0], text_embeds[0], dim=0).item()

    results = {
        "Garment_Alignment_Score_CLIP": round(garment_alignment_score, 4),
        "Text_Alignment_Score_CLIP": round(text_alignment_score, 4),
        "evaluated_output": output_img_path,
        "garment_reference": garment_img_path,
        "prompt": prompt
    }
    
    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f" - {k}: {v}")
        
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved results to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--garment", type=str, default="images/cloth.jpg")
    parser.add_argument("--output", type=str, default="output_tryon_demo.png")
    parser.add_argument("--prompt", type=str, default="A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution")
    
    args = parser.parse_args()
    if os.path.exists(args.output):
        evaluate(args.garment, args.output, args.prompt)
    else:
        print(f"Error: Output image '{args.output}' does not exist. Run inference first.")
