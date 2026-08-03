import torch
from torch import nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, 128, 3, padding=1), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(128, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

class SemanticGenerationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(6, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64, 1, 3, padding=1))
        
    def forward(self, pose, garment):
        if pose.shape[1] > 3:
            pose = pose[:, :3, :, :]
        elif pose.shape[1] < 3:
            pose = pose.repeat(1, 3, 1, 1)
        x = torch.cat([pose, garment], dim=1)
        return self.dec2(self.dec1(self.enc2(self.enc1(x))) + self.enc1(x))

class ClothesWarpingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 6))
        
    def forward(self, garment, semantic_layout):
        b, c, h, w = garment.shape
        x = torch.cat([garment, semantic_layout], dim=1)
        theta = self.regressor(self.extract(x).view(b, -1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, garment.size(), align_corners=False)
        return F.grid_sample(garment, grid, align_corners=False), grid

class ContentFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Conv2d(7, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 3, 3, padding=1), nn.Tanh())
    def forward(self, person, warped_garment, layout):
        return self.fusion(torch.cat([person, warped_garment, layout], dim=1))

class CustomLightweightTryOn(nn.Module):
    def __init__(self):
        super().__init__()
        self.sgm = SemanticGenerationModule()
        self.cwm = ClothesWarpingModule()
        self.cfm = ContentFusionModule()
    def forward(self, person, garment, pose):
        layout = self.sgm(pose, garment)
        warped, grid = self.cwm(garment, layout)
        output = self.cfm(person, warped, layout)
        return output, warped, layout, grid

# Support dynamic pose channel adaptation (RGB 3-channel vs OpenPose 18-channel)
