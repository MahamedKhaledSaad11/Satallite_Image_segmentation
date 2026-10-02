"""
Dataset Audit Module
Performs comprehensive auditing of satellite images and binary masks.
"""

from pathlib import Path
from typing import Dict, Any, List
import tifffile
from PIL import Image
import numpy as np
import pandas as pd

def run_dataset_audit(
    image_dir: str,
    label_dir: str,
    expected_count: int = 306
) -> Dict[str, Any]:
    """
    Audit image and label directories.
    
    Verifies:
      1. Every pair {0..N-1} exists and loads cleanly.
      2. Image shape is (128, 128, 12), dtype is int16.
      3. Label shape is (128, 128), dtype is uint8, values in {0, 1}.
      4. Tracks NoData (-9999) per band.
      5. Tracks negative values (valid atmospheric correction in bands 0-6).
      6. Computes class distribution and water ratios.
    """
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)

    # Automatic fallback if running from inside notebooks/ directory
    if not (img_path.exists() and (img_path / "0.tif").exists()):
        from src.utils.paths import get_data_dirs
        default_img, _, _, _, _ = get_data_dirs()
        img_path = default_img

    if not (lbl_path.exists() and (lbl_path / "0.png").exists()):
        from src.utils.paths import get_data_dirs
        _, default_lbl, _, _, _ = get_data_dirs()
        lbl_path = default_lbl

    if not img_path.exists():
        raise FileNotFoundError(
            f"Image directory not found: '{image_dir}'. (Resolved: '{img_path.resolve()}'). "
            "Please ensure you pass a valid path to the directory containing 0.tif...305.tif."
        )
    if not lbl_path.exists():
        raise FileNotFoundError(
            f"Label directory not found: '{label_dir}'. (Resolved: '{lbl_path.resolve()}'). "
            "Please ensure you pass a valid path to the directory containing 0.png...305.png."
        )

    records = []
    missing_images = []
    missing_labels = []
    corrupted_files = []
    
    # Check X_Y.png duplicate files in labels dir
    all_label_files = list(lbl_path.glob("*.png"))
    extra_xy_files = [f.name for f in all_label_files if "_" in f.stem]
    
    band_names = [
        "Coastal Aerosol", "Blue", "Green", "Red",
        "NIR", "SWIR1", "SWIR2", "QA Band",
        "Merit DEM", "Copernicus DEM", "ESA WorldCover", "Water Occurrence"
    ]
    
    # Per-band accumulators for global stats
    band_mins = [float("inf")] * 12
    band_maxs = [float("-inf")] * 12
    band_sums = [0.0] * 12
    band_sq_sums = [0.0] * 12
    band_nodata_counts = [0] * 12
    band_valid_pixels = [0] * 12
    
    water_ratios = []
    empty_water_count = 0
    majority_water_count = 0
    
    for idx in range(expected_count):
        cur_img_path = img_path / f"{idx}.tif"
        cur_lbl_path = lbl_path / f"{idx}.png"
        
        if not cur_img_path.exists():
            missing_images.append(idx)
            continue
        if not cur_lbl_path.exists():
            missing_labels.append(idx)
            continue
            
        try:
            img = tifffile.imread(str(cur_img_path))
            lbl = np.array(Image.open(str(cur_lbl_path)))
        except Exception as e:
            corrupted_files.append((idx, str(e)))
            continue
            
        # Shape & dtype verification
        assert img.shape == (128, 128, 12), f"Image {idx} invalid shape: {img.shape}"
        assert img.dtype == np.int16, f"Image {idx} invalid dtype: {img.dtype}"
        assert lbl.shape == (128, 128), f"Label {idx} invalid shape: {lbl.shape}"
        assert lbl.dtype == np.uint8, f"Label {idx} invalid dtype: {lbl.dtype}"
        
        unique_vals = set(np.unique(lbl))
        assert unique_vals.issubset({0, 1}), f"Label {idx} contains non-binary values: {unique_vals}"
        
        water_ratio = float((lbl == 1).mean())
        water_ratios.append(water_ratio)
        if water_ratio == 0.0:
            empty_water_count += 1
        elif water_ratio > 0.5:
            majority_water_count += 1
            
        # Per band statistics accumulation
        for b in range(12):
            band_data = img[:, :, b]
            nodata_mask = (band_data == -9999)
            nodata_count = int(np.sum(nodata_mask))
            band_nodata_counts[b] += nodata_count
            
            valid_data = band_data[~nodata_mask]
            n_valid = len(valid_data)
            band_valid_pixels[b] += n_valid
            
            if n_valid > 0:
                b_min = float(valid_data.min())
                b_max = float(valid_data.max())
                if b_min < band_mins[b]:
                    band_mins[b] = b_min
                if b_max > band_maxs[b]:
                    band_maxs[b] = b_max
                band_sums[b] += float(valid_data.sum())
                band_sq_sums[b] += float((valid_data.astype(np.float64) ** 2).sum())
                
        # Elevation for DEM grouping
        cop_mean = float(img[:, :, 9].mean())
        merit = img[:, :, 8]
        valid_merit = merit[merit != -9999]
        merit_mean = float(valid_merit.mean()) if len(valid_merit) > 0 else np.nan
        
        records.append({
            "image_id": idx,
            "water_ratio": water_ratio,
            "copernicus_mean": cop_mean,
            "merit_mean": merit_mean,
            "group_10m": int(round(cop_mean / 10) * 10),
            "group_20m": int(round(cop_mean / 20) * 20),
        })
        
    if not records:
        raise RuntimeError(
            f"No valid image-label pairs could be found! Checked image_dir='{img_path}' and label_dir='{lbl_path}'."
        )

    df_audit = pd.DataFrame(records)
    
    # Compute global mean & std for audit reporting
    band_means = [band_sums[b] / max(1, band_valid_pixels[b]) for b in range(12)]
    band_stds = [
        np.sqrt(max(0.0, (band_sq_sums[b] / max(1, band_valid_pixels[b])) - (band_means[b] ** 2)))
        for b in range(12)
    ]
    
    summary = {
        "total_expected": expected_count,
        "valid_pairs": len(df_audit),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "corrupted_files": corrupted_files,
        "extra_xy_labels_count": len(extra_xy_files),
        "mean_water_ratio": float(np.mean(water_ratios)) if water_ratios else 0.0,
        "median_water_ratio": float(np.median(water_ratios)) if water_ratios else 0.0,
        "empty_water_images": empty_water_count,
        "majority_water_images": majority_water_count,
        "band_names": band_names,
        "band_mins": band_mins,
        "band_maxs": band_maxs,
        "band_means": band_means,
        "band_stds": band_stds,
        "band_nodata_counts": band_nodata_counts,
    }
    
    return summary, df_audit

if __name__ == "__main__":
    import json
    summary, df_audit = run_dataset_audit(
        "water_segmentation/data/raw/images",
        "water_segmentation/data/raw/labels"
    )
    print("=" * 45)
    print("           DATASET AUDIT REPORT")
    print("=" * 45)
    print(f"Total expected:       {summary['total_expected']}")
    print(f"Valid pairs loaded:   {summary['valid_pairs']}")
    print(f"Missing images:       {len(summary['missing_images'])}")
    print(f"Missing labels:       {len(summary['missing_labels'])}")
    print(f"Corrupted files:      {len(summary['corrupted_files'])}")
    print(f"Extra X_Y.png files:  {summary['extra_xy_labels_count']} (IGNORED)")
    print(f"Mean water ratio:     {summary['mean_water_ratio']:.3%}")
    print(f"Median water ratio:   {summary['median_water_ratio']:.3%}")
    print(f"Zero-water images:    {summary['empty_water_images']}")
    print(f"Majority-water imgs:  {summary['majority_water_images']}")
    print("\nBand Statistics (full dataset overview):")
    for b in range(12):
        print(f"  B{b:02d} ({summary['band_names'][b]}): min={summary['band_mins'][b]:.1f}, max={summary['band_maxs'][b]:.1f}, mean={summary['band_means'][b]:.1f}, std={summary['band_stds'][b]:.1f}, NoData(-9999) count={summary['band_nodata_counts'][b]}")
    print("=" * 45)
