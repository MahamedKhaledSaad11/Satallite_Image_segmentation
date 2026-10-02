"""
Evaluation Entry Point for Flood Water Segmentation.
Runs inference on test (or val) split using optimal validation threshold,
computes strict unbiased test metrics, and exports qualitative visual reports.
Usage:
    python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_model.pt --split test --threshold 0.50
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import json
import torch
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.data.dataset import MultispectralWaterDataset
from src.data.preprocessing import load_train_statistics
from src.models.unet import UNet
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.utils.visualization import plot_prediction_comparison

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Water Segmentation Model")
    parser.add_argument("--config", type=str, default="water_segmentation/configs/final_model_engineered.yaml", help="Path to config")
    parser.add_argument("--checkpoint", type=str, default="water_segmentation/checkpoints/final_model_engineered/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (defaults to validation optimal or 0.5)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--save_viz", action="store_true", default=True, help="Save qualitative prediction error plots")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluation running on: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Determine threshold
    thresh = args.threshold
    if thresh is None:
        summary_file = ckpt_path.parent / "training_summary.json"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                sum_data = json.load(f)
                thresh = sum_data.get("optimal_threshold", 0.5)
        else:
            thresh = 0.5
    print(f"Evaluation Decision Threshold: {thresh:.2f}")

    # Data setup
    data_cfg = config["data"]
    img_dir = data_cfg["image_dir"]
    lbl_dir = data_cfg["label_dir"]
    split_dir = Path(data_cfg["split_dir"])
    split_csv = split_dir / f"{args.split}.csv"

    stats = load_train_statistics(data_cfg["stats_file"])
    band_indices = data_cfg.get("band_indices", None)

    use_feat_eng = data_cfg.get("feature_engineering", False)
    eng_stats = None
    if use_feat_eng:
        eng_stats_file = Path(data_cfg.get("engineered_stats_file", "E:\\Cellula_Technologies\\Project2\\water_segmentation\\data\\statistics\\engineered_train_stats.json"))
        if eng_stats_file.exists():
            with open(eng_stats_file, "r", encoding="utf-8") as f:
                eng_stats = json.load(f)

    norm_method = data_cfg.get("norm_method", "zscore")

    dataset = MultispectralWaterDataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        image_ids=split_csv,
        band_means=stats["band_means"],
        band_stds=stats["band_stds"],
        nodata_fill=stats["nodata_fill"],
        augmentation=None,
        band_indices=band_indices,
        use_feature_engineering=use_feat_eng,
        norm_method=norm_method,
        engineered_stats=eng_stats
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Model setup
    from src.models.unet import build_model
    model_cfg = config["model"]
    in_channels = len(band_indices) if band_indices is not None else model_cfg.get("in_channels", 12)
    model_cfg["in_channels"] = in_channels
    model = build_model(model_cfg).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {model_cfg.get('name', 'unet')} (in_channels={in_channels}, feature_engineering={use_feat_eng})")
    if use_feat_eng:
        print("  -> Evaluating Engineered Hydrological Stack (12): 7 Optical + Copernicus DEM + 4 Water Indices (MNDWI, NDWI, AWEI_sh, NDVI)")
    else:
        print("  -> Evaluating Naive Raw Satellite Baseline Stack")

    # Evaluation
    metrics_calc = SegmentationMetrics(threshold=thresh)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metrics_calc.update(logits, masks)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > thresh).astype(np.uint8)
            all_preds.extend(preds[:, 0])
            all_targets.extend(masks.cpu().numpy()[:, 0])

    res = metrics_calc.compute()

    print("=" * 60)
    print(f"      EVALUATION RESULTS (Split: {args.split.upper()})")
    print("=" * 60)
    print(f"Images Evaluated:       {len(dataset)}")
    print(f"Decision Threshold:     {thresh:.2f}")
    print(f"Water IoU (Global):     {res['water_iou_global']:.4f}")
    print(f"Water IoU (Mean Image): {res['water_iou_mean']:.4f}")
    print(f"Precision (Global):     {res['precision_global']:.4f}")
    print(f"Recall (Global):        {res['recall_global']:.4f}")
    print(f"F1-Score (Global):      {res['f1_global']:.4f}")
    print(f"Empty-Image FPR:        {res['empty_image_fpr']:.4%}")
    print("=" * 60)

    # Save metrics JSON
    reports_dir = Path("water_segmentation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{args.split}_metrics.json"
    save_payload = {
        "split": args.split,
        "threshold": thresh,
        "metrics": res,
        "checkpoint": str(args.checkpoint)
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=4)
    print(f"Metrics saved to: {report_file}")

    # Generate qualitative visualizations if requested
    if args.save_viz:
        viz_dir = reports_dir / f"visualizations_{args.split}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        df_split = pd.read_csv(split_csv)
        img_ids = df_split["image_id"].tolist()
        
        # Select first 6 images from split for report
        for i in range(min(6, len(dataset))):
            img_id = img_ids[i]
            # Load raw image for RGB visualization
            raw_tif = tifffile.imread(str(Path(img_dir) / f"{img_id}.tif"))
            raw_chw = np.transpose(raw_tif, (2, 0, 1))

            target = all_targets[i]
            pred = all_preds[i]

            img_iou = metrics_calc.per_image_ious[i]
            plot_prediction_comparison(
                image_raw_chw=raw_chw,
                target_hw=target,
                pred_hw=pred,
                title=f"Sample {img_id} (Split: {args.split}, IoU: {img_iou:.3f})",
                save_path=str(viz_dir / f"sample_{img_id}_comparison.png")
            )
        print(f"Qualitative visualization plots saved to: {viz_dir}")

if __name__ == "__main__":
    main()
