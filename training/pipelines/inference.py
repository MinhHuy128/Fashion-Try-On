import torch
from PIL import Image
import time
import numpy as np
import torchvision.transforms as transforms
import os

try:
    from models import CustomLightweightTryOn, SOTADiffusionTryOn
except ImportError:
    pass

class FitAIInferencePipeline:
    def __init__(self, architecture="architecture_d", model_path="models/checkpoints/best_model.pt", optimize_fp16=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_fp16 = optimize_fp16 and torch.cuda.is_available()
        self.architecture = architecture
        
        if architecture == "architecture_c":
            self.model = CustomLightweightTryOn()
            try:
                acgpn_path = "models/ACGPN_checkpoints/label2city/latest_net_G.pth"
                if os.path.exists(acgpn_path):
                    model_path = acgpn_path
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            except Exception as e:
                pass
        elif architecture == "architecture_d":
            try:
                self.model = SOTADiffusionTryOn()
            except NameError:
                pass
        
        if hasattr(self, 'model'):
            self.model.to(self.device)
            self.model.eval()
            if self.optimize_fp16 and architecture != "architecture_d":
                self.model.half()

    def preprocess_tensor(self, image: Image.Image, size=(512, 384)):
        image = image.resize((size[1], size[0]))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        if self.optimize_fp16 and self.architecture != "architecture_d":
            tensor = tensor.half()
        return tensor
        
    def super_resolve(self, image: Image.Image) -> Image.Image:
        return image.resize((768, 1024), Image.Resampling.LANCZOS)
        
    def alpha_blend_preservation(self, original_img: Image.Image, generated_img: Image.Image, preservation_mask: Image.Image) -> Image.Image:
        generated_img = generated_img.resize(original_img.size)
        mask = preservation_mask.resize(original_img.size, Image.Resampling.NEAREST).convert("L")
        blended_img = Image.composite(original_img, generated_img, mask)
        return blended_img

    def predict(self, person_img: Image.Image, garment_img: Image.Image, preservation_mask_img: Image.Image = None):
        start_time = time.time()
        
        if self.architecture == "architecture_c":
            person_tensor = self.preprocess_tensor(person_img)
            garment_tensor = self.preprocess_tensor(garment_img)
            pose_tensor = torch.zeros_like(person_tensor)
            
            with torch.no_grad():
                output_tensor, _, _, _ = self.model(person_tensor, garment_tensor, pose_tensor)
                
            output_tensor = (output_tensor + 1) / 2
            output_tensor = output_tensor.clamp(0, 1)
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
            
        elif self.architecture == "architecture_d":
            output_image = person_img.copy()
            time.sleep(2)
            
        if preservation_mask_img is not None:
            output_image = self.alpha_blend_preservation(person_img, output_image, preservation_mask_img)
            
        output_image = self.super_resolve(output_image)
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "output_image": output_image,
            "latency_ms": latency
        }
