"""
Classical Remote Sensing Spectral Water Index Baselines.
Evaluates NDWI (McFeeters 1996) and MNDWI (Xu 2006) baselines without machine learning.
Optimizes the detection threshold strictly on the validation set, then evaluates once on test.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR + 1e-8)"""
    return (green - nir) / (green + nir + 1e-8)

def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1 + 1e-8)"""
    return (green - swir1) / (green + swir1 + 1e-8)

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Computes global TP, FP, FN, IoU, Precision, Recall, and F1."""
    p_bool = (pred == 1)
    t_bool = (target == 1)
    
    tp = int(np.sum(p_bool & t_bool))
    fp = int(np.sum(p_bool & (~t_bool)))
    fn = int(np.sum((~p_bool) & t_bool))
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def evaluate_index_split(
    image_dir: str,
    label_dir: str,
    split_csv: str,
    index_name: str = "mndwi",
    threshold: float = 0.0
) -> Dict[str, float]:
    from src.utils.paths import resolve_file
    split_path = resolve_file(split_csv)
    df = pd.read_csv(split_path)
    img_dir_path = resolve_file(image_dir)
    lbl_dir_path = resolve_file(label_dir)
    
    all_preds = []
    all_targets = []
    
    for img_id in df["image_id"]:
        img = tifffile.imread(str(img_dir_path / f"{img_id}.tif")).astype(np.float32)
        lbl = np.array(Image.open(str(lbl_dir_path / f"{img_id}.png")), dtype=np.uint8)
        
        green = img[:, :, 2]
        nir = img[:, :, 4]
        swir1 = img[:, :, 5]
        
        if index_name.lower() == "ndwi":
            idx_map = compute_ndwi(green, nir)
        elif index_name.lower() == "mndwi":
            idx_map = compute_mndwi(green, swir1)
        else:
            raise ValueError(f"Unknown index {index_name}")
            
        pred = (idx_map > threshold).astype(np.uint8)
        all_preds.append(pred)
        all_targets.append(lbl)
        
    return compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))

def search_optimal_threshold(
    image_dir: str,
    label_dir: str,
    val_csv: str,
    index_name: str = "mndwi",
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, float]]:
    if thresholds is None:
        thresholds = np.arange(-0.4, 0.6, 0.05)
        
    best_thresh = 0.0
    best_iou = -1.0
    best_metrics = {}
    
    for t in thresholds:
        metrics = evaluate_index_split(image_dir, label_dir, val_csv, index_name, threshold=t)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_thresh = t
            best_metrics = metrics
            
    return float(best_thresh), best_metrics

if __name__ == "__main__":
    img_dir = "water_segmentation/data/raw/images"
    lbl_dir = "water_segmentation/data/raw/labels"
    val_csv = "water_segmentation/data/splits/val.csv"
    test_csv = "water_segmentation/data/splits/test.csv"
    
    print("=" * 55)
    print("       CLASSICAL SPECTRAL WATER INDEX BASELINES")
    print("=" * 55)
    
    for idx_name in ["ndwi", "mndwi"]:
        opt_thresh, val_res = search_optimal_threshold(img_dir, lbl_dir, val_csv, idx_name)
        test_res = evaluate_index_split(img_dir, lbl_dir, test_csv, idx_name, threshold=opt_thresh)
        
        print(f"\n--- {idx_name.upper()} Baseline ---")
        print(f"Optimal Val Threshold: {opt_thresh:+.2f}")
        print(f"Val Performance : IoU={val_res['iou']:.4f} | F1={val_res['f1']:.4f} | P={val_res['precision']:.4f} | R={val_res['recall']:.4f}")
        print(f"Test Performance: IoU={test_res['iou']:.4f} | F1={test_res['f1']:.4f} | P={test_res['precision']:.4f} | R={test_res['recall']:.4f}")
    print("=" * 55)
