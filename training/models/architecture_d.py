import torch
from torch import nn
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class SOTADiffusionTryOn(nn.Module):
    """
    Architecture D: SOTA Diffusion-based Try-On (IDM-VTON / OOTDiffusion inspired)
    Utilizes Stable Diffusion Inpainting + LoRA + Garment Feature Injection.
    """
    def __init__(self, base_model_id="runwayml/stable-diffusion-inpainting"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load core Diffusers components
        self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        
        # Garment Feature Extractor (CLIP)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")
        
        # Freeze VAE and Image Encoder (we only train the UNet via LoRA or natively)
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        
        # Projection layer to match UNet cross-attention dim (1024 -> 768)
        self.proj = nn.Linear(1024, 768)
        
        # Optimization
        if torch.cuda.is_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print("xformers not installed, skipping memory efficient attention")
            self.unet.enable_gradient_checkpointing()
            
    def encode_images(self, pixel_values):
        """Encodes pixel values to latent space using VAE"""
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
        
    def encode_garment_features(self, garment_images):
        """Extracts garment semantics using CLIP"""
        garment_resized = F.interpolate(garment_images, size=(224, 224), mode='bilinear', align_corners=False)
        return self.image_encoder(garment_resized).image_embeds
        
    def forward(self, person_latents, mask_latents, masked_image_latents, garment_embeds, timesteps, noise):
        """
        Forward pass for training.
        Stable Diffusion Inpainting UNet expects 9 channels:
        4 (latent noise) + 1 (mask) + 4 (masked image latents)
        """
        noisy_latents = self.noise_scheduler.add_noise(person_latents, noise, timesteps)
        
        # Concatenate for inpainting: [noisy_latents, mask, masked_image]
        latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)
        
        # Project from 1024 to 768, cast input to match proj weight dtype
        garment_embeds_proj = self.proj(garment_embeds.to(self.proj.weight.dtype))
        
        # Typically shape is (batch_size, sequence_length, hidden_size)
        encoder_hidden_states = garment_embeds_proj.unsqueeze(1).repeat(1, 77, 1) 
        
        noise_pred = self.unet(
            latent_model_input.to(self.unet.dtype),
            timesteps,
            encoder_hidden_states=encoder_hidden_states.to(self.unet.dtype)
        ).sample
        
        return noise_pred
