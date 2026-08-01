import argparse
import os
import sys
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from training.pipelines.data_processing import get_dataloader
from training.pipelines.evaluate import evaluate_model

def load_config(config_path="configs/best_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    parser = argparse.ArgumentParser(description="Academic International Benchmark Evaluation Suite for Fashion-Try-On")
    parser.add_argument("--config", type=str, default="configs/best_config.yaml")
    parser.add_argument("--architecture", type=str, default=None, choices=["architecture_a", "architecture_b", "architecture_c", "architecture_d"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--eval_dir", type=str, default="evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    arch = args.architecture or config.get("architecture", "architecture_c")
    data_dir = args.data_dir or config.get("data_dir", "data/ACGPN_raw")
    checkpoint = args.checkpoint or os.path.join(config.get("save_dir", "models/checkpoints"), "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  FitAI Virtual Try-On: Evaluation Benchmark")
    print(f"  Device:       {device}")
    print(f"  Architecture: {arch}")
    print("=" * 60)
    if arch == "architecture_c":
        from training.models.architecture_c import CustomLightweightTryOn
        model = CustomLightweightTryOn()
    elif arch == "architecture_d":
        from training.models.architecture_d import SOTADiffusionTryOn
        model = SOTADiffusionTryOn()
    else:
        from training.models.architecture_c import CustomLightweightTryOn
        model = CustomLightweightTryOn()

    # Load custom trained checkpoints if provided
    if checkpoint and os.path.exists(checkpoint):
        try:
            if os.path.isfile(checkpoint) and arch == "architecture_c":
                state_dict = torch.load(checkpoint, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {checkpoint}")
            elif os.path.isdir(checkpoint) and arch == "architecture_d":
                from peft import PeftModel
                model.unet = PeftModel.from_pretrained(model.unet, checkpoint)
                print(f"Loaded LoRA weights from {checkpoint}")
        except Exception as e:
            print(f"Warning: could not load weights ({e}), using initialized weights.")
    elif checkpoint:
        print(f"Checkpoint '{checkpoint}' not found, running on base weights.")

    is_mock = not os.path.exists(data_dir)
    dataloader = get_dataloader(data_dir=data_dir, batch_size=args.batch_size, split=args.split, is_mock=is_mock)
    eval_config = dict(config)
    eval_config["eval_dir"] = args.eval_dir
    results = evaluate_model(model=model, dataloader=dataloader, config=eval_config, compute_fid=not args.no_fid, compute_clip=not args.no_clip, max_samples=args.max_samples)
    print("\n" + "=" * 60)
    print("  Evaluation Results:")
    print("=" * 60)
    print(f"  - SSIM:             {results['SSIM']}")
    print(f"  - LPIPS (AlexNet):  {results['LPIPS_AlexNet']}")
    print(f"  - LPIPS (VGG):      {results['LPIPS_VGG']}")
    print(f"  - PSNR:             {results['PSNR_dB']} dB")
    print(f"  - FID:              {results['FID']}")
    print(f"  - CLIP Score:       {results['CLIP_Garment_Alignment']}")
    print(f"  - Latency:          {results['Avg_Latency_ms']} ms ({results['FPS']} FPS)")
    print(f"  - Peak VRAM:        {results['Peak_VRAM_MB']} MB")
    print("=" * 60)
    print(f"  Saved metrics to: {os.path.join(args.eval_dir, 'metrics.json')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
