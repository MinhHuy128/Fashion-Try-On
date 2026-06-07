import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ACGPNDataset(Dataset):
    """
    Dataset parser specifically mapped to the ACGPN folder structure:
    - train_img (person)
    - train_color (garment)
    - train_pose (pose keypoints heatmap or image)
    - train_label (parsing mask)
    """
    def __init__(self, data_dir, split="train", img_size=(256, 192), is_mock=False):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.is_mock = is_mock
        
        # ACGPN Structure
        self.person_dir = os.path.join(data_dir, f"{split}_img")
        self.garment_dir = os.path.join(data_dir, f"{split}_color")
        self.pose_dir = os.path.join(data_dir, f"{split}_pose")
        self.mask_dir = os.path.join(data_dir, f"{split}_label")
        
        self.image_names = []
        if not is_mock and os.path.exists(self.person_dir):
            self.image_names = [f for f in os.listdir(self.person_dir) if f.endswith(('.jpg', '.png'))]
            self.image_names.sort()
            
        self.length = len(self.image_names) if not is_mock and self.image_names else 100
        
        self.transform_img = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def generate_agnostic_and_preservation(self, person_img, parsing_img):
        person_np = np.array(person_img)
        parsing_np = np.array(parsing_img)
        
        # Agnostic Mask (Upper Body Clothes = 5,6,7 in LIP)
        cloth_mask = np.isin(parsing_np, [5, 6, 7])
        agnostic_np = person_np.copy()
        agnostic_np[cloth_mask] = 128
        inpaint_mask = torch.from_numpy(cloth_mask.astype(np.float32)).unsqueeze(0)
        agnostic_img = Image.fromarray(agnostic_np)
        
        # Preservation Mask (Face/Hair/Neck/Hands)
        preservation_ids = [1, 2, 4, 10, 13, 14, 15]
        preserve_np = np.isin(parsing_np, preservation_ids).astype(np.float32)
        preservation_mask = torch.from_numpy(preserve_np).unsqueeze(0)
        
        return agnostic_img, inpaint_mask, preservation_mask

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_mock or not self.image_names:
            person_img = torch.rand(3, self.img_size[0], self.img_size[1]) * 2 - 1
            garment_img = torch.rand(3, self.img_size[0], self.img_size[1]) * 2 - 1
            pose_map = torch.rand(18, self.img_size[0], self.img_size[1])
            parsing_mask = torch.randint(0, 20, (1, self.img_size[0], self.img_size[1])).float()
            agnostic_img = torch.rand(3, self.img_size[0], self.img_size[1]) * 2 - 1
            inpaint_mask = torch.rand(1, self.img_size[0], self.img_size[1]) > 0.5
            preservation_mask = torch.rand(1, self.img_size[0], self.img_size[1]) > 0.8
            return {
                "person": person_img, "garment": garment_img, "pose": pose_map, 
                "parsing": parsing_mask, "agnostic": agnostic_img, 
                "inpaint_mask": inpaint_mask.float(), "preservation_mask": preservation_mask.float()
            }
            
        img_name = self.image_names[idx]
        
        try:
            person_pil = Image.open(os.path.join(self.person_dir, img_name)).convert("RGB")
            person_tensor = self.transform_img(person_pil)
            
            garment_name = img_name.replace("_0.jpg", "_1.jpg").replace("_0.png", "_1.png")
            garment_img = Image.open(os.path.join(self.garment_dir, garment_name)).convert("RGB")
            garment_tensor = self.transform_img(garment_img)
            
            mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))
            if os.path.exists(mask_path):
                parsing_pil = Image.open(mask_path).convert("L")
                parsing_tensor = self.transform_mask(parsing_pil) * 255.0
                agnostic_pil, inpaint_mask, preservation_mask = self.generate_agnostic_and_preservation(person_pil, parsing_pil)
                agnostic_tensor = self.transform_img(agnostic_pil)
            else:
                parsing_tensor = torch.zeros(1, self.img_size[0], self.img_size[1])
                agnostic_tensor = person_tensor.clone()
                inpaint_mask = torch.zeros(1, self.img_size[0], self.img_size[1])
                preservation_mask = torch.zeros(1, self.img_size[0], self.img_size[1])
                
            pose_path = os.path.join(self.pose_dir, img_name.replace(".jpg", "_rendered.png")) # ACGPN pose format often has _rendered
            if not os.path.exists(pose_path):
                pose_path = os.path.join(self.pose_dir, img_name) # Fallback
                
            if os.path.exists(pose_path):
                pose_img = Image.open(pose_path).convert("RGB")
                pose_tensor = self.transform_img(pose_img)
            else:
                pose_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
                
        except Exception as e:
            print(f"Warning: Error loading {img_name}: {e}")
            person_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            garment_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            pose_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            parsing_tensor = torch.zeros(1, self.img_size[0], self.img_size[1])
            agnostic_tensor = torch.zeros(3, self.img_size[0], self.img_size[1])
            inpaint_mask = torch.zeros(1, self.img_size[0], self.img_size[1])
            preservation_mask = torch.zeros(1, self.img_size[0], self.img_size[1])
            
        return {
            "person": person_tensor, "garment": garment_tensor, "pose": pose_tensor,
            "parsing": parsing_tensor, "agnostic": agnostic_tensor,
            "inpaint_mask": inpaint_mask, "preservation_mask": preservation_mask
        }

def get_dataloader(data_dir, batch_size=4, split="train", is_mock=False):
    dataset = ACGPNDataset(data_dir=data_dir, split=split, is_mock=is_mock)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4 if not is_mock else 0, pin_memory=True, drop_last=True)
    return loader
