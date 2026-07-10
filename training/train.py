import yaml
import sys
import os
import torch

# Fix for RTX 5060 Ti (sm_120) crashing on FlashAttention kernels built for older archs
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from pipelines.data_processing import get_dataloader
from pipelines.train import train_model
from pipelines.train_diffusion import train_diffusion
from pipelines.evaluate import evaluate_model
from models import ControlNetTryOnModel, IPAdapterTryOnModel, CustomLightweightTryOn, SOTADiffusionTryOn

def load_config(config_path="configs/best_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_model(architecture_name):
    if architecture_name == "architecture_a":
        return ControlNetTryOnModel()
    elif architecture_name == "architecture_b":
        return IPAdapterTryOnModel()
    elif architecture_name == "architecture_c":
        return CustomLightweightTryOn()
    elif architecture_name == "architecture_d":
        return SOTADiffusionTryOn()
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

def main():
    print("=== PHASE 1: Initialization ===")
    config = load_config()
    print(f"Loaded config: {config}")
    
    print("\n=== PHASE 2: Architecture Selection ===")
    architecture = config.get("architecture", "architecture_c")
    print(f"Selected Architecture: {architecture}")
    model = get_model(architecture)
    
    print("\n=== PHASE 3: Data Preprocessing ===")
    is_mock = config.get("is_mock", True)
    batch_size = config.get("batch_size", 4)
    data_dir = config.get("data_dir", "data")
    
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(data_dir, batch_size=batch_size, split="train", is_mock=is_mock)
    val_loader = get_dataloader(data_dir, batch_size=batch_size, split="val", is_mock=is_mock)
    test_loader = get_dataloader(data_dir, batch_size=batch_size, split="test", is_mock=is_mock)
    print("[OK] Phase 3 completed")

    print("\n=== PHASE 4: Training Pipeline with Monitoring ===")
    num_epochs = config.get("num_epochs", 2)
    print(f"Starting training for {num_epochs} epochs...")
    try:
        if architecture == "architecture_d":
            # Call SOTA Diffusion training loop
            train_diffusion(model, train_loader, num_epochs=num_epochs, config=config)
            print("[OK] Phase 4 completed. (Diffusion LoRA Saved)")
            best_val_loss = "N/A (MSE Noise Loss Mode)"
        else:
            # Call GAN/TPS training loop
            best_val_loss = train_model(model, train_loader, val_loader, num_epochs=num_epochs, config=config)
            print("[OK] Phase 4 completed. Best Val Loss:", best_val_loss)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

    print("\n=== PHASE 5 & 6: Evaluation and Optimization ===")
    print("Evaluating model...")
    try:
        results = evaluate_model(model, test_loader, config)
        print("[OK] Phase 5 & 6 completed")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

    print("\n=== PHASE 7: Autonomous Execution & Reporting ===")
    print("[OK] Execution completed successfully.")
    
    print("\n================ SUMMARY REPORT ================")
    print(f"- Model selected: {model.__class__.__name__}")
    print(f"- Final metrics: LPIPS={results['LPIPS']:.4f}, FID={results['FID']:.4f}, Latency={results['Avg_Latency_ms']:.2f}ms")
    print("- Optimization applied: FP16 Mixed Precision Training & Inference")
    print("- Recommended use case: E-commerce rapid virtual try-on")
    print("- Files location:")
    print(f"  - Checkpoint: {config.get('save_dir')}/")
    print("  - Evaluation Report: evaluation/report.html")
    print("  - Metrics: evaluation/metrics.json")
    print("================================================")

if __name__ == "__main__":
    main()
