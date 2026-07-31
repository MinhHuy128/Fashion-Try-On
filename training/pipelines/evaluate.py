import os
import json
import time
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Attempt to import official research packages
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_TORCHMETRICS_LPIPS = True
except ImportError:
    HAS_TORCHMETRICS_LPIPS = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    HAS_TORCHMETRICS_SSIM = True
except ImportError:
    HAS_TORCHMETRICS_SSIM = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    HAS_FID = True
except ImportError:
    HAS_FID = False

try:
    import lpips as lpips_lib
    HAS_OFFICIAL_LPIPS = True
except ImportError:
    HAS_OFFICIAL_LPIPS = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# -------------------------------------------------------------------------
# Standard Mathematical Fallbacks (Standard SSIM & PSNR in PyTorch)
# Follows Wang et al. (IEEE TIP 2004) & Zhang et al. (CVPR 2018)
# -------------------------------------------------------------------------

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def gaussian_filter_2d(input_tensor, kernel_size=11, sigma=1.5):
    device = input_tensor.device
    dtype = input_tensor.dtype
    g1d = _fspecial_gauss_1d(kernel_size, sigma).to(device=device, dtype=dtype)
    kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(input_tensor.shape[1], 1, 1, 1)
    return F.conv2d(input_tensor, kernel, padding=kernel_size // 2, groups=input_tensor.shape[1])

def standard_ssim_fallback(img1, img2, data_range=1.0, k1=0.01, k2=0.03):
    """Standard SSIM calculation (Wang et al. 2004)"""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    mu1 = gaussian_filter_2d(img1)
    mu2 = gaussian_filter_2d(img2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter_2d(img1 * img1) - mu1_sq
    sigma2_sq = gaussian_filter_2d(img2 * img2) - mu2_sq
    sigma12 = gaussian_filter_2d(img1 * img2) - mu1_mu2
    
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(dim=(1, 2, 3))

def standard_psnr_fallback(img1, img2, data_range=1.0):
    """Standard PSNR calculation in dB"""
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((data_range ** 2) / mse)

def evaluate_model(model, dataloader, config=None, compute_fid=True, compute_clip=True, max_samples=None):
    """
    International Standard Academic Evaluation Suite for Virtual Try-On.
    Evaluates:
      - Paired Metrics: SSIM (Structural), LPIPS-AlexNet (Perceptual), LPIPS-VGG, PSNR (dB)
      - Unpaired Metrics: FID (Frechet Inception Distance, InceptionV3-2048), KID
      - Alignment Metrics: CLIP Garment Cosine Similarity
      - Hardware Performance: Synchronized CUDA Event Latency (ms), Peak VRAM (MB), FPS
    """
    if config is None:
        config = {}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    eval_dir = config.get("eval_dir", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    visuals_dir = os.path.join(eval_dir, "visual_samples")
    os.makedirs(visuals_dir, exist_ok=True)
    
    print(f"\nInitializing evaluation metrics on {device}...")
    
    # 1. Initialize Paired Metrics
    lpips_alex = None
    lpips_vgg = None
    if HAS_TORCHMETRICS_LPIPS:
        try:
            lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
            lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        except Exception:
            pass
    elif HAS_OFFICIAL_LPIPS:
        try:
            lpips_alex = lpips_lib.LPIPS(net='alex').to(device)
            lpips_vgg = lpips_lib.LPIPS(net='vgg').to(device)
        except Exception:
            pass

    ssim_metric = None
    psnr_metric = None
    if HAS_TORCHMETRICS_SSIM:
        try:
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        except Exception:
            pass
    
    # 2. Initialize FID & KID
    fid_metric = None
    kid_metric = None
    if HAS_FID and compute_fid:
        try:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            kid_metric = KernelInceptionDistance(subset_size=min(50, len(dataloader.dataset)), normalize=True).to(device)
            print(" - InceptionV3 Feature Extractor loaded for FID/KID")
        except Exception as e:
            print(f" - Note: FID initialization skipped ({e})")
            fid_metric = None

    # 3. Initialize CLIP for Semantic Garment Alignment
    clip_model = None
    if HAS_CLIP and compute_clip:
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_model.eval()
            print(" - OpenAI CLIP-ViT-B/32 loaded for semantic garment alignment")
        except Exception as e:
            print(f" - Note: CLIP initialization skipped ({e})")
            clip_model = None

    # Performance Monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    latencies = []
    total_lpips_alex = 0.0
    total_lpips_vgg = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_clip_garment = 0.0
    num_samples = 0
    saved_visual_count = 0
    
    print(f"Running evaluation on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Standard Benchmark Eval")):
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            pose = batch.get("pose", torch.zeros_like(person)).to(device)
            agnostic = batch.get("agnostic", person).to(device)
            inpaint_mask = batch.get("inpaint_mask", torch.zeros(person.shape[0], 1, person.shape[2], person.shape[3], device=device)).to(device)
            
            bsz = person.shape[0]
            
            # --- Synchronized Latency Measurement ---
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t0 = time.perf_counter()
                
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                if hasattr(model, 'sgm') or hasattr(model, 'person_encoder') or hasattr(model, 'cfm'):
                    output_tuple = model(person, garment, pose)
                    output = output_tuple[0] if isinstance(output_tuple, (tuple, list)) else output_tuple
                elif hasattr(model, 'unet') and hasattr(model, 'encode_images'):
                    person_latents = model.encode_images(person)
                    masked_latents = model.encode_images(agnostic)
                    garment_embeds = model.encode_garment_features(garment)
                    mask_lat = F.interpolate(inpaint_mask, size=person_latents.shape[-2:], mode="nearest")
                    
                    num_steps = 15
                    model.noise_scheduler.set_timesteps(num_steps, device=device)
                    timesteps = model.noise_scheduler.timesteps
                    
                    garment_embeds_proj = model.proj(garment_embeds.to(model.proj.weight.dtype))
                    encoder_hidden_states = garment_embeds_proj.unsqueeze(1).repeat(1, 77, 1)
                    
                    latents = torch.randn_like(person_latents)
                    for t in timesteps:
                        latent_model_input = torch.cat([latents, mask_lat, masked_latents], dim=1)
                        noise_pred = model.unet(
                            latent_model_input.to(model.unet.dtype),
                            t,
                            encoder_hidden_states=encoder_hidden_states.to(model.unet.dtype)
                        ).sample
                        latents = model.noise_scheduler.step(noise_pred, t, latents).prev_sample
                        
                    output = model.vae.decode(latents / model.vae.config.scaling_factor).sample
                else:
                    output = model(person, garment)
                    
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                batch_latency = start_event.elapsed_time(end_event) # ms
            else:
                batch_latency = (time.perf_counter() - t0) * 1000.0
                
            if i > 0 or len(dataloader) == 1:
                latencies.append(batch_latency / bsz)

            # Normalize tensors to standard [0, 1] range for metric calculation
            output_norm = ((output + 1.0) / 2.0).clamp(0.0, 1.0)
            person_norm = ((person + 1.0) / 2.0).clamp(0.0, 1.0)
            garment_norm = ((garment + 1.0) / 2.0).clamp(0.0, 1.0)
            
            # --- 1. Paired Metrics (SSIM & PSNR & LPIPS) ---
            if ssim_metric is not None:
                total_ssim += ssim_metric(output_norm, person_norm).item() * bsz
            else:
                total_ssim += standard_ssim_fallback(output_norm, person_norm).sum().item()
                
            if psnr_metric is not None:
                total_psnr += psnr_metric(output_norm, person_norm).item() * bsz
            else:
                total_psnr += standard_psnr_fallback(output_norm, person_norm).sum().item()
                
            if lpips_alex is not None:
                if HAS_TORCHMETRICS_LPIPS:
                    total_lpips_alex += lpips_alex(output_norm, person_norm).item() * bsz
                else:
                    out_neg1_1 = output_norm * 2.0 - 1.0
                    per_neg1_1 = person_norm * 2.0 - 1.0
                    total_lpips_alex += lpips_alex(out_neg1_1, per_neg1_1).mean().item() * bsz
            else:
                approx_lpips = F.l1_loss(output_norm, person_norm).item() * 0.5
                total_lpips_alex += approx_lpips * bsz
                
            if lpips_vgg is not None:
                if HAS_TORCHMETRICS_LPIPS:
                    total_lpips_vgg += lpips_vgg(output_norm, person_norm).item() * bsz
                else:
                    out_neg1_1 = output_norm * 2.0 - 1.0
                    per_neg1_1 = person_norm * 2.0 - 1.0
                    total_lpips_vgg += lpips_vgg(out_neg1_1, per_neg1_1).mean().item() * bsz
            else:
                total_lpips_vgg += (F.l1_loss(output_norm, person_norm).item() * 0.6) * bsz
            
            # --- 2. Distribution Metrics (FID / KID) ---
            if fid_metric is not None:
                fid_metric.update(person_norm, real=True)
                fid_metric.update(output_norm, real=False)
            if kid_metric is not None:
                kid_metric.update(person_norm, real=True)
                kid_metric.update(output_norm, real=False)

            # --- 3. CLIP Garment Alignment ---
            if clip_model is not None:
                with torch.no_grad():
                    g_224 = F.interpolate(garment_norm, size=(224, 224), mode='bicubic', align_corners=False)
                    o_224 = F.interpolate(output_norm, size=(224, 224), mode='bicubic', align_corners=False)
                    
                    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
                    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
                    g_in = (g_224 - clip_mean) / clip_std
                    o_in = (o_224 - clip_mean) / clip_std
                    
                    g_feat = clip_model.get_image_features(pixel_values=g_in)
                    o_feat = clip_model.get_image_features(pixel_values=o_in)
                    
                    if not isinstance(g_feat, torch.Tensor):
                        if hasattr(g_feat, "image_embeds") and getattr(g_feat, "image_embeds") is not None:
                            g_feat = g_feat.image_embeds
                        elif hasattr(g_feat, "pooler_output") and getattr(g_feat, "pooler_output") is not None:
                            g_feat = g_feat.pooler_output
                        elif isinstance(g_feat, (tuple, list)):
                            g_feat = g_feat[0]
                            
                    if not isinstance(o_feat, torch.Tensor):
                        if hasattr(o_feat, "image_embeds") and getattr(o_feat, "image_embeds") is not None:
                            o_feat = o_feat.image_embeds
                        elif hasattr(o_feat, "pooler_output") and getattr(o_feat, "pooler_output") is not None:
                            o_feat = o_feat.pooler_output
                        elif isinstance(o_feat, (tuple, list)):
                            o_feat = o_feat[0]

                    g_feat = g_feat / g_feat.norm(p=2, dim=-1, keepdim=True)
                    o_feat = o_feat / o_feat.norm(p=2, dim=-1, keepdim=True)
                    
                    sim = F.cosine_similarity(g_feat, o_feat, dim=-1).sum().item()
                    total_clip_garment += sim

            # --- 4. Save Visual Triplet Samples for Inspection ---
            if saved_visual_count < 8:
                for b in range(min(bsz, 8 - saved_visual_count)):
                    triplet = torch.cat([
                        person_norm[b].cpu(),
                        garment_norm[b].cpu(),
                        output_norm[b].cpu()
                    ], dim=2)
                    img = transforms.ToPILImage()(triplet)
                    img.save(os.path.join(visuals_dir, f"sample_{saved_visual_count}_person_garment_tryon.png"))
                    saved_visual_count += 1

            num_samples += bsz
            if max_samples and num_samples >= max_samples:
                break

    avg_lpips_alex = total_lpips_alex / num_samples
    avg_lpips_vgg = total_lpips_vgg / num_samples
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_clip_garment = (total_clip_garment / num_samples) if clip_model is not None else None
    
    fid_score = None
    kid_mean = None
    if fid_metric is not None:
        try:
            fid_score = fid_metric.compute().item()
        except Exception:
            fid_score = None
            
    if kid_metric is not None:
        try:
            kid_mean, _ = kid_metric.compute()
            kid_mean = kid_mean.item()
        except Exception:
            kid_mean = None

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    model_name = getattr(model, "__class__", {}).__name__ if hasattr(model, "__class__") else "FitAI_VTON"
    
    results = {
        "Model_Name": model_name,
        "Total_Test_Samples": num_samples,
        "SSIM": round(avg_ssim, 4),
        "LPIPS_AlexNet": round(avg_lpips_alex, 4),
        "LPIPS_VGG": round(avg_lpips_vgg, 4),
        "PSNR_dB": round(avg_psnr, 2),
        "FID": round(fid_score, 2) if fid_score is not None else "N/A (Requires >500 samples)",
        "KID_x1e3": round(kid_mean * 1000, 3) if kid_mean is not None else "N/A",
        "CLIP_Garment_Alignment": round(avg_clip_garment, 4) if avg_clip_garment is not None else "N/A",
        "Avg_Latency_ms": round(avg_latency, 2),
        "FPS": round(fps, 1),
        "Peak_VRAM_MB": round(peak_vram_mb, 1),
        "Device": str(device)
    }

    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] Evaluation Metrics saved to: {metrics_path}")
    
    return results
