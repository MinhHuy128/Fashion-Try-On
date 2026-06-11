import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from tqdm import tqdm
from models import PatchGANDiscriminator

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27], vgg[27:36]]).to(device)
        for param in self.parameters():
            param.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss += F.l1_loss(x, y)
        return loss

def gan_loss_g(D_fake):
    return 0.5 * torch.mean((D_fake - 1.0) ** 2)

def gan_loss_d(D_real, D_fake):
    return 0.5 * torch.mean((D_real - 1.0) ** 2) + 0.5 * torch.mean(D_fake ** 2)

def train_model(model, dataloader, val_dataloader, num_epochs=5, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    discriminator = PatchGANDiscriminator(in_channels=3).to(device)
    opt_G = AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4), betas=(0.5, 0.999))
    opt_D = AdamW(discriminator.parameters(), lr=config.get("learning_rate", 1e-4), betas=(0.5, 0.999))
    l1_loss_fn = nn.L1Loss()
    vgg_loss_fn = VGGPerceptualLoss(device)
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            pose = batch["pose"].to(device)
            output, _, _, _ = model(person, garment, pose)
            opt_D.zero_grad()
            loss_D = gan_loss_d(discriminator(person), discriminator(output.detach()))
            loss_D.backward()
            opt_D.step()
            opt_G.zero_grad()
            loss_G = l1_loss_fn(output, person)*10.0 + vgg_loss_fn(output, person)*10.0 + gan_loss_g(discriminator(output))
            loss_G.backward()
            opt_G.step()
    return 0.0
