"""
Multispectral Preprocessing & Train-Only Normalization.
Computes per-band normalization statistics STRICTLY from the training split,
handles NoData values (-9999 in Band 8), and applies z-score normalization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import tifffile
import numpy as np
import pandas as pd

BAND_NAMES = [
    "coastal_aerosol",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "qa",
    "merit_dem",
    "copernicus_dem",
    "worldcover",
    "water_occurrence"
]

def compute_train_statistics(
    image_dir: str,
    train_ids: List[int],
    nodata_values: Optional[Dict[int, float]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute per-band mean and standard deviation ONLY from training images.
    NoData values (e.g. -9999 in Band 8) are strictly masked out before calculating stats.
    """
    if nodata_values is None:
        nodata_values = {8: -9999.0}
        
    img_dir_path = Path(image_dir)
    
    # Running accumulators for numerical stability
    band_sums = np.zeros(12, dtype=np.float64)
    band_sq_sums = np.zeros(12, dtype=np.float64)
    band_valid_counts = np.zeros(12, dtype=np.int64)
    
    for img_id in train_ids:
        img_path = img_dir_path / f"{img_id}.tif"
        img = tifffile.imread(str(img_path))  # (128, 128, 12), int16
        
        for b in range(12):
            band_data = img[:, :, b].astype(np.float64)
            if b in nodata_values:
                valid_mask = (band_data != nodata_values[b])
            else:
                valid_mask = np.ones(band_data.shape, dtype=bool)
                
            valid_pixels = band_data[valid_mask]
            count = len(valid_pixels)
            if count > 0:
                band_sums[b] += np.sum(valid_pixels)
                band_sq_sums[b] += np.sum(valid_pixels ** 2)
                band_valid_counts[b] += count

    band_means = band_sums / np.maximum(band_valid_counts, 1)
    variance = (band_sq_sums / np.maximum(band_valid_counts, 1)) - (band_means ** 2)
    band_stds = np.sqrt(np.maximum(variance, 1e-8))
    
    # Band 8 fill value is the training mean of Band 8
    fill_values = {}
    for b in nodata_values:
        fill_values[str(b)] = float(band_means[b])
        
    stats = {
        "band_names": BAND_NAMES,
        "band_means": [float(m) for m in band_means],
        "band_stds": [float(s) for s in band_stds],
        "nodata_fill": fill_values,
        "computed_from": "train_split_only",
        "n_train_images": len(train_ids),
        "split_seed": seed,
        "note": "Computed strictly from the train split. Applied identically to train, val, and test."
    }
    return stats

def save_train_statistics(stats: Dict[str, Any], output_path: str) -> None:
    """Save training statistics to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

def load_train_statistics(stats_path: str) -> Dict[str, Any]:
    """Load training statistics from a JSON file."""
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_image(
    image: np.ndarray,
    band_means: np.ndarray,
    band_stds: np.ndarray,
    nodata_fill: Optional[Dict[str, float]] = None,
    nodata_values: Optional[Dict[int, float]] = None
) -> np.ndarray:
    """
    Transforms raw (128, 128, 12) int16 image into normalized (12, 128, 128) float32.
    
    1. Casts to float32.
    2. Replaces NoData values (-9999 in Band 8) with train mean fill.
    3. Normalizes each band: (pixel - train_mean) / train_std.
    4. Transposes from (H, W, C) to (C, H, W) for PyTorch.
    """
    if nodata_values is None:
        nodata_values = {8: -9999.0}
    if nodata_fill is None:
        nodata_fill = {"8": float(band_means[8])}
        
    img = image.astype(np.float32)
    
    # Handle NoData
    for b_idx, nd_val in nodata_values.items():
        key = str(b_idx)
        fill_val = nodata_fill.get(key, float(band_means[b_idx]))
        mask = (img[:, :, b_idx] == nd_val)
        img[mask, b_idx] = fill_val
        
    # Z-score normalization
    means = np.asarray(band_means, dtype=np.float32)
    stds = np.asarray(band_stds, dtype=np.float32)
    
    normalized = (img - means) / stds
    # Transpose (H, W, C) -> (C, H, W)
    transposed = np.transpose(normalized, (2, 0, 1))
    return transposed.astype(np.float32)

if __name__ == "__main__":
    train_csv = Path("water_segmentation/data/splits/train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Run split.py first.")
        
    df_train = pd.read_csv(train_csv)
    train_ids = df_train["image_id"].tolist()
    
    stats = compute_train_statistics(
        image_dir="water_segmentation/data/raw/images",
        train_ids=train_ids,
        nodata_values={8: -9999.0},
        seed=42
    )
    
    out_file = "water_segmentation/data/statistics/train_stats.json"
    save_train_statistics(stats, out_file)
    
    print("=" * 55)
    print("   TRAIN-ONLY BAND STATISTICS (SAVED TO JSON)")
    print("=" * 55)
    print(f"Computed from: {stats['n_train_images']} training images strictly.")
    for b in range(12):
        name = stats["band_names"][b]
        m = stats["band_means"][b]
        s = stats["band_stds"][b]
        print(f"  B{b:02d} ({name:18s}): mean={m:9.2f}, std={s:9.2f}")
    print(f"Band 8 NoData Fill Value: {stats['nodata_fill']['8']:.2f}")
    print("=" * 55)
