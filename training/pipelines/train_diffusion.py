import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from models import SOTADiffusionTryOn
from peft import LoraConfig, get_peft_model

def train_diffusion(model, dataloader, num_epochs=10, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    model.unet = get_peft_model(model.unet, lora_config)
    model.unet.train()
    
    optimizer = AdamW(model.unet.parameters(), lr=config.get("learning_rate", 1e-4), weight_decay=1e-2)
    
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(project="fitai-diffusion-sota", config=config)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            agnostic = batch["agnostic"].to(device)
            inpaint_mask = batch["inpaint_mask"].to(device)
            
            with torch.no_grad():
                person_latents = model.encode_images(person)
                masked_image_latents = model.encode_images(agnostic)
                garment_embeds = model.encode_garment_features(garment)
                mask_latents = F.interpolate(inpaint_mask, size=person_latents.shape[-2:], mode="nearest")
                
            noise = torch.randn_like(person_latents)
            bsz = person_latents.shape[0]
            timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                noise_pred = model(
                    person_latents=person_latents,
                    mask_latents=mask_latents,
                    masked_image_latents=masked_image_latents,
                    garment_embeds=garment_embeds,
                    timesteps=timesteps,
                    noise=noise
                )
                
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
            
            if use_wandb and batch_idx % 50 == 0:
                wandb.log({"batch_diffusion_loss": loss.item()})
                
        save_path = os.path.join(config.get("save_dir", "models/checkpoints"), f"diffusion_lora_epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        model.unet.save_pretrained(save_path)
        
    if use_wandb:
        wandb.finish()
