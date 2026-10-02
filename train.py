"""
Main Training Entry Point for Water Segmentation.
Usage:
    python train.py --config configs/baseline.yaml
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
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.dataset import MultispectralWaterDataset
from src.data.augmentation import SpatialAugmentation
from src.data.preprocessing import load_train_statistics, compute_train_statistics, save_train_statistics
from src.models.unet import UNet
from src.losses.segmentation_loss import get_loss_function
from src.metrics.segmentation_metrics import find_optimal_threshold
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Flood Water Segmentation U-Net")
    parser.add_argument("--config", type=str, default="water_segmentation/configs/final_model_engineered.yaml", help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda or cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # 2. Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 3. Data paths and statistics
    data_cfg = config["data"]
    img_dir = data_cfg["image_dir"]
    lbl_dir = data_cfg["label_dir"]
    split_dir = Path(data_cfg["split_dir"])
    stats_file = Path(data_cfg["stats_file"])
    band_indices = data_cfg.get("band_indices", None)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if not stats_file.exists():
        print(f"Stats file {stats_file} not found. Computing train statistics strictly from {train_csv}...")
        import pandas as pd
        train_ids = pd.read_csv(train_csv)["image_id"].tolist()
        stats = compute_train_statistics(
            image_dir=img_dir,
            train_ids=train_ids,
            nodata_values=config.get("preprocessing", {}).get("nodata_values", {8: -9999.0}),
            seed=seed
        )
        save_train_statistics(stats, str(stats_file))
    else:
        stats = load_train_statistics(str(stats_file))

    band_means = stats["band_means"]
    band_stds = stats["band_stds"]
    nodata_fill = stats["nodata_fill"]

    # 4. Augmentation (Train-only)
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("enabled", True):
        train_aug = SpatialAugmentation(
            hflip_prob=aug_cfg.get("horizontal_flip", 0.5),
            vflip_prob=aug_cfg.get("vertical_flip", 0.5),
            rot90_prob=aug_cfg.get("random_rotation_90", 0.5)
        )
    else:
        train_aug = None

    # Check if feature engineering is enabled
    use_feat_eng = data_cfg.get("feature_engineering", False)
    eng_stats = None
    if use_feat_eng:
        eng_stats_file = Path(data_cfg.get("engineered_stats_file", "water_segmentation/data/statistics/engineered_train_stats.json"))
        if eng_stats_file.exists():
            with open(eng_stats_file, "r", encoding="utf-8") as f:
                eng_stats = json.load(f)

    norm_method = data_cfg.get("norm_method", "zscore")

    # 5. Datasets & Dataloaders
    train_dataset = MultispectralWaterDataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        image_ids=train_csv,
        band_means=band_means,
        band_stds=band_stds,
        nodata_fill=nodata_fill,
        augmentation=train_aug,
        band_indices=band_indices,
        use_feature_engineering=use_feat_eng,
        norm_method=norm_method,
        engineered_stats=eng_stats
    )

    val_dataset = MultispectralWaterDataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        image_ids=val_csv,
        band_means=band_means,
        band_stds=band_stds,
        nodata_fill=nodata_fill,
        augmentation=None,
        band_indices=band_indices,
        use_feature_engineering=use_feat_eng,
        norm_method=norm_method,
        engineered_stats=eng_stats
    )

    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    print(f"Data ready: Train={len(train_dataset)} images ({len(train_loader)} batches), Val={len(val_dataset)} images ({len(val_loader)} batches) | Feature Engineering: {use_feat_eng} ({norm_method})")
    if use_feat_eng:
        print("  -> Channel Stack (12): [B0:Coastal, B1:Blue, B2:Green, B3:Red, B4:NIR, B5:SWIR1, B6:SWIR2, Copernicus DEM, MNDWI, NDWI, AWEI_sh, NDVI]")
    else:
        print(f"  -> Channel Stack ({in_channels}): Raw satellite bands ({'all 12 raw bands' if band_indices is None else band_indices})")

    # 6. Model definition
    from src.models.unet import build_model
    model_cfg = config["model"]
    in_channels = len(band_indices) if band_indices is not None else model_cfg.get("in_channels", 12)
    model_cfg["in_channels"] = in_channels
    model = build_model(model_cfg).to(device)

    print(f"Model initialized: {model_cfg.get('name', 'unet')} (in_channels={in_channels}, base={model_cfg.get('base_channels', 32)}, norm={model_cfg.get('norm_type', 'group')}) with {model.count_parameters():,} parameters")

    # 7. Loss, Optimizer, Scheduler
    criterion = get_loss_function(config.get("loss", {}))

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    sched_name = train_cfg.get("scheduler", "reduce_on_plateau").lower()
    if sched_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=train_cfg.get("early_stopping_mode", "max"),
            patience=train_cfg.get("scheduler_patience", 5),
            factor=train_cfg.get("scheduler_factor", 0.5),
            min_lr=float(train_cfg.get("scheduler_min_lr", 1e-6))
        )
    else:
        scheduler = None

    # 8. Training loop
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    fit_results = trainer.fit()

    # 9. Post-Training Optimal Threshold Search on Validation Set
    print("\nRunning post-training threshold optimization on Validation set...")
    # Load best weights
    best_ckpt = torch.load(trainer.best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    cand_threshs = config.get("evaluation", {}).get(
        "threshold_search",
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    )
    best_thresh, opt_metrics = find_optimal_threshold(
        model=model,
        val_loader=val_loader,
        device=device,
        thresholds=cand_threshs
    )

    print("=" * 60)
    print("      POST-TRAINING VALIDATION THRESHOLD SEARCH")
    print("=" * 60)
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Validation IoU:   {opt_metrics['water_iou_global']:.4f}")
    print(f"Validation F1:    {opt_metrics['f1_global']:.4f}")
    print(f"Validation P:     {opt_metrics['precision_global']:.4f}")
    print(f"Validation R:     {opt_metrics['recall_global']:.4f}")
    print("=" * 60)

    # Save training summary
    summary_path = trainer.save_dir / "training_summary.json"
    summary_data = {
        "best_epoch": fit_results["best_epoch"],
        "optimal_threshold": best_thresh,
        "validation_metrics_at_optimum": opt_metrics,
        "config": config
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
