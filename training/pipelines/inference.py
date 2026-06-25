import torch
import torch.nn as nn
from PIL import Image
import time
import numpy as np
import torchvision.transforms as transforms
import os

from diffusers import StableDiffusionInpaintPipeline
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

try:
    from models.architecture_c import CustomLightweightTryOn
    from models.architecture_d import SOTADiffusionTryOn
except ImportError:
    try:
        from training.models.architecture_c import CustomLightweightTryOn
        from training.models.architecture_d import SOTADiffusionTryOn
    except ImportError:
        pass

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

class FitAIInferencePipeline:
    def __init__(self, architecture="architecture_d", model_path="models/checkpoints/best_model.pt", optimize_fp16=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_fp16 = optimize_fp16 and torch.cuda.is_available()
        self.architecture = architecture
        self.sd_pipeline = None
        self.seg_processor = None
        self.seg_model = None

        if architecture == "architecture_c":
            self.model = CustomLightweightTryOn()
            try:
                acgpn_path = "models/ACGPN_checkpoints/label2city/latest_net_G.pth"
                if os.path.exists(acgpn_path):
                    model_path = acgpn_path
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                self.model.to(self.device).eval()
                if self.optimize_fp16:
                    self.model.half()
            except Exception as e:
                print(f"Warning loading Architecture C weights: {e}")
        else:
            # Architecture D (SD Inpainting + IP-Adapter)
            self._init_sd_ip_adapter()

    def _init_sd_ip_adapter(self):
        try:
            print("Initializing Stable Diffusion Inpainting Pipeline with IP-Adapter...")
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=dtype,
            ).to(self.device)
            self.sd_pipeline.safety_checker = None
            self.sd_pipeline.requires_safety_checker = False
            
            try:
                self.sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                self.sd_pipeline.set_ip_adapter_scale(1.0)
            except Exception as e:
                print(f"Warning loading IP-Adapter weights: {e}")

            # Auto-detect and load custom trained LoRA weights from training checkpoints
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lora_dirs = [
                os.path.join(root_dir, "training", "models", "checkpoints"),
                os.path.join(root_dir, "models", "checkpoints"),
                root_dir,
            ]
            
            lora_loaded = False
            for lora_base in lora_dirs:
                if lora_loaded:
                    break
                if os.path.exists(lora_base) and os.path.isdir(lora_base):
                    # 1. Check direct weights inside lora_base
                    if os.path.exists(os.path.join(lora_base, "adapter_model.safetensors")) or os.path.exists(os.path.join(lora_base, "adapter_model.bin")):
                        try:
                            print(f"Loading custom trained LoRA weights from: {lora_base}")
                            self.sd_pipeline.load_lora_weights(lora_base)
                            print(f"[SUCCESS] Loaded custom trained LoRA checkpoint from {lora_base}!")
                            lora_loaded = True
                            break
                        except Exception as e:
                            print(f"Warning loading LoRA weights: {e}")
                    
                    # 2. Check subdirectories inside lora_base
                    sub_candidates = []
                    for item in os.listdir(lora_base):
                        item_path = os.path.join(lora_base, item)
                        if os.path.isdir(item_path):
                            if os.path.exists(os.path.join(item_path, "adapter_model.safetensors")) or os.path.exists(os.path.join(item_path, "adapter_model.bin")):
                                sub_candidates.append(item_path)
                    
                    if sub_candidates:
                        sub_candidates.sort(reverse=True)
                        target_lora = sub_candidates[0]
                        try:
                            print(f"Loading custom trained LoRA weights from: {target_lora}")
                            self.sd_pipeline.load_lora_weights(target_lora)
                            print(f"[SUCCESS] Loaded custom trained LoRA checkpoint ({os.path.basename(target_lora)})!")
                            lora_loaded = True
                            break
                        except Exception as e:
                            print(f"Warning loading LoRA weights: {e}")

            # Segformer for auto masking
            self.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(self.device)
        except Exception as e:
            print(f"Error loading SD Inpainting pipeline: {e}")

    def generate_auto_mask(self, person_img: Image.Image) -> Image.Image:
        if self.seg_processor is None or self.seg_model is None:
            return Image.new("L", person_img.size, 255)
        
        inputs = self.seg_processor(images=person_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=person_img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        # Upper-clothes mask (class 4)
        mask_np = (pred_seg == 4).astype(np.uint8) * 255

        # Soft feathering at the seam boundary using GaussianBlur
        try:
            import cv2
            mask_np = cv2.GaussianBlur(mask_np, (7, 7), 0)
        except Exception:
            pass

        mask_img = Image.fromarray(mask_np, mode='L')
        return mask_img

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

    def predict(
        self, 
        person_img: Image.Image, 
        garment_img: Image.Image, 
        preservation_mask_img: Image.Image = None,
        prompt: str = "A highly detailed, photorealistic image of a person wearing this exact t-shirt, perfect texture and logo matching, 8k resolution",
        ip_scale: float = 0.75,
        seed: int = 42
    ):
        start_time = time.time()
        
        if self.architecture == "architecture_c" and hasattr(self, 'model'):
            person_tensor = self.preprocess_tensor(person_img)
            garment_tensor = self.preprocess_tensor(garment_img)
            pose_tensor = torch.zeros_like(person_tensor)
            
            with torch.no_grad():
                output_tensor, _, _, _ = self.model(person_tensor, garment_tensor, pose_tensor)
                
            output_tensor = (output_tensor + 1) / 2
            output_tensor = output_tensor.clamp(0, 1)
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
            
        else: # Default Architecture D or Fallback: Stable Diffusion + IP-Adapter
            if self.sd_pipeline is not None:
                person_resized = person_img.convert("RGB").resize((512, 512))
                garment_resized = garment_img.convert("RGB").resize((224, 224))
                mask_img = self.generate_auto_mask(person_resized).resize((512, 512))
                
                try:
                    self.sd_pipeline.set_ip_adapter_scale(ip_scale)
                except Exception:
                    pass

                generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

                output_image = self.sd_pipeline(
                    prompt=prompt,
                    image=person_resized,
                    mask_image=mask_img,
                    ip_adapter_image=garment_resized,
                    num_inference_steps=25,
                    guidance_scale=7.0,
                    generator=generator,
                ).images[0]
            else:
                output_image = person_img.copy()
            
        if preservation_mask_img is not None:
            output_image = self.alpha_blend_preservation(person_img, output_image, preservation_mask_img)
            
        output_image = self.super_resolve(output_image)
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "output_image": output_image,
            "latency_ms": latency
        }

