import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import mlflow
import wandb
from tqdm import tqdm
import time
from models import PatchGANDiscriminator

# -------------------------------------------------------------
# Research-Grade Losses
# -------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27], vgg[27:36]
        ]).to(device)
        for param in self.parameters():
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

def gan_loss_g(D_fake):
    """LSGAN Generator Loss"""
    return 0.5 * torch.mean((D_fake - 1.0) ** 2)

def gan_loss_d(D_real, D_fake):
    """LSGAN Discriminator Loss"""
    loss_real = 0.5 * torch.mean((D_real - 1.0) ** 2)
    loss_fake = 0.5 * torch.mean(D_fake ** 2)
    return loss_real + loss_fake

def tps_grid_loss(grid):
    """Second-Order Smoothness Loss to prevent TPS grid tearing"""
    dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    dyy = dy[:, 1:, :, :] - dy[:, :-1, :, :]
    dxx = dx[:, :, 1:, :] - dx[:, :, :-1, :]
    dxy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    return torch.mean(dxx**2) + torch.mean(dyy**2) + torch.mean(dxy**2)

# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------

def train_model(model, dataloader, val_dataloader, num_epochs=5, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize Discriminator for GAN Training
    discriminator = PatchGANDiscriminator(in_channels=3).to(device)
    
    # Optimizers
    opt_G = AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4), betas=(0.5, 0.999))
    opt_D = AdamW(discriminator.parameters(), lr=config.get("learning_rate", 1e-4), betas=(0.5, 0.999))
    
    sched_G = CosineAnnealingLR(opt_G, T_max=num_epochs)
    sched_D = CosineAnnealingLR(opt_D, T_max=num_epochs)
    
    # Loss functions
    l1_loss_fn = nn.L1Loss()
    vgg_loss_fn = VGGPerceptualLoss(device)
    
    scaler_G = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scaler_D = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_loss = float('inf')
    
    use_wandb = config.get("use_wandb", False)
    if use_wandb:
        wandb.init(project="fitai-tryon-research", config=config)

    # Weights for losses
    lambda_l1 = 10.0
    lambda_vgg = 10.0
    lambda_gan = 1.0
    lambda_tps = 0.1

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        
        train_g_loss_total = 0.0
        train_d_loss_total = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(progress_bar):
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            pose = batch["pose"].to(device)
            
            # --- 1. Train Discriminator ---
            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if hasattr(model, 'person_encoder'):
                    # Architecture C returns 4 outputs
                    output, warped_gmt, layout, grid = model(person, garment, pose)
                else:
                    output = model(person, garment)
                
                D_real = discriminator(person)
                D_fake = discriminator(output.detach())
                loss_D = gan_loss_d(D_real, D_fake)
                
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # --- 2. Train Generator ---
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                D_fake_for_G = discriminator(output)
                loss_gan = gan_loss_g(D_fake_for_G) * lambda_gan
                loss_l1 = l1_loss_fn(output, person) * lambda_l1
                loss_vgg = vgg_loss_fn(output, person) * lambda_vgg
                
                loss_G = loss_l1 + loss_vgg + loss_gan
                
                if hasattr(model, 'person_encoder'): # Add TPS Grid Loss
                    loss_G += tps_grid_loss(grid) * lambda_tps
                    
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            
            train_g_loss_total += loss_G.item()
            train_d_loss_total += loss_D.item()
            
            progress_bar.set_postfix({"L_G": loss_G.item(), "L_D": loss_D.item()})
            
            if use_wandb and batch_idx % 100 == 0:
                wandb.log({"batch_loss_G": loss_G.item(), "batch_loss_D": loss_D.item()})
                
        sched_G.step()
        sched_D.step()
        
        # Validation Loop omitted for brevity but similar to train
        # Saving checkpoints
        save_path = os.path.join(config.get("save_dir", "models/checkpoints"), "best_model.pt")
        torch.save(model.state_dict(), save_path)
                
    if use_wandb:
        wandb.finish()
        
    return best_val_loss
