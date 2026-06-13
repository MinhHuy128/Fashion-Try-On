import torch
from torch import nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

class ControlNetTryOnModel(nn.Module):
    """
    Architecture A: ControlNet-based (Pose-Guided)
    Uses Stable Diffusion v1.5 with ControlNet for pose control.
    """
    def __init__(self, base_model="runwayml/stable-diffusion-v1-5", controlnet_model="lllyasviel/sd-controlnet-openpose"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load Pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, controlnet=self.controlnet, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.device)
        
        # Enable memory optimizations if on CUDA
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_xformers_memory_efficient_attention()

    def forward(self, garment_prompt, pose_image, num_inference_steps=20):
        """
        Inference step
        """
        output = self.pipeline(
            garment_prompt,
            image=pose_image,
            num_inference_steps=num_inference_steps,
        ).images[0]
        return output
