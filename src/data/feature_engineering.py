"""
Multispectral Feature Engineering for Water Segmentation.
Computes domain-specific remote sensing indices and builds an optimized feature stack:
1. MNDWI (Modified Normalized Difference Water Index) - Xu (2006)
2. NDWI (Normalized Difference Water Index) - McFeeters (1996)
3. AWEI_sh (Automated Water Extraction Index with Shadow Elimination) - Feyisa et al. (2014)
4. NDVI (Normalized Difference Vegetation Index) - Rouse et al. (1974)
5. Topographic Elevation (Copernicus DEM)
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import numpy as np
import pandas as pd
import tifffile

def safe_normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes (A - B) / (A + B) safely clamped to [-1.0, 1.0]."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    denom = a_f + b_f
    # Avoid division by zero or negative denominator anomalies
    denom = np.where(np.abs(denom) < 1e-4, 1e-4 * np.sign(denom + 1e-7), denom)
    ratio = (a_f - b_f) / denom
    return np.clip(ratio, -1.0, 1.0)

def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1) clamped to [-1.0, 1.0]."""
    return safe_normalized_diff(green, swir1)

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR) clamped to [-1.0, 1.0]."""
    return safe_normalized_diff(green, nir)

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red) clamped to [-1.0, 1.0]."""
    return safe_normalized_diff(nir, red)

def compute_awei_sh(
    blue: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray
) -> np.ndarray:
    """
    AWEI_sh: Automated Water Extraction Index designed to eliminate mountain and urban shadows.
    Formula: Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2
    """
    return (blue.astype(np.float32) + 2.5 * green.astype(np.float32) - 
            1.5 * (nir.astype(np.float32) + swir1.astype(np.float32)) - 
            0.25 * swir2.astype(np.float32))

ENGINEERED_CHANNEL_NAMES = [
    "coastal_aerosol",  # B00: shallow water & turbidity
    "blue",             # B01: clear water penetration
    "green",            # B02: water reflectance peak
    "red",              # B03: sediment & turbidity sensitivity
    "nir",              # B04: strong water absorption edge
    "swir1",            # B05: built-up vs water discrimination
    "swir2",            # B06: saturated soil moisture
    "copernicus_dem",   # B09: topographic gravity flow
    "mndwi",            # Feature 1: Modified NDWI
    "ndwi",             # Feature 2: Classical NDWI
    "awei_sh",          # Feature 3: Shadow-elimination AWEI
    "ndvi"              # Feature 4: Vegetation suppression
]

def build_engineered_features(raw_image: np.ndarray) -> np.ndarray:
    """
    Constructs a 12-channel physically engineered hydrological feature stack
    from the raw 12-channel satellite image.
    
    Args:
        raw_image: (128, 128, 12) int16 array
    Returns:
        (128, 128, 12) float32 array with engineered features
    """
    img = raw_image.astype(np.float32)
    
    coastal = img[:, :, 0]
    blue    = img[:, :, 1]
    green   = img[:, :, 2]
    red     = img[:, :, 3]
    nir     = img[:, :, 4]
    swir1   = img[:, :, 5]
    swir2   = img[:, :, 6]
    cop_dem = img[:, :, 9]  # Clean 30m DEM
    
    mndwi   = compute_mndwi(green, swir1)
    ndwi    = compute_ndwi(green, nir)
    awei_sh = compute_awei_sh(blue, green, nir, swir1, swir2)
    ndvi    = compute_ndvi(nir, red)
    
    stack = np.stack([
        coastal, blue, green, red, nir, swir1, swir2, cop_dem,
        mndwi, ndwi, awei_sh, ndvi
    ], axis=-1)
    
    return stack.astype(np.float32)

def compute_engineered_train_statistics(
    image_dir: str,
    train_ids: List[int],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes dataset-wide per-channel statistics (mean, std, min, max, p01, p99)
    STRICTLY from the training set for the engineered feature stack.
    """
    img_dir_path = Path(image_dir)
    n_channels = len(ENGINEERED_CHANNEL_NAMES)
    
    sums = np.zeros(n_channels, dtype=np.float64)
    sq_sums = np.zeros(n_channels, dtype=np.float64)
    mins = np.full(n_channels, float("inf"), dtype=np.float64)
    maxs = np.full(n_channels, float("-inf"), dtype=np.float64)
    total_pixels = 0
    
    for img_id in train_ids:
        raw_img = tifffile.imread(str(img_dir_path / f"{img_id}.tif"))
        feat_img = build_engineered_features(raw_img)  # (128, 128, 12)
        h, w, c = feat_img.shape
        n_px = h * w
        total_pixels += n_px
        
        for ch in range(n_channels):
            data = feat_img[:, :, ch].astype(np.float64)
            c_min = float(data.min())
            c_max = float(data.max())
            if c_min < mins[ch]:
                mins[ch] = c_min
            if c_max > maxs[ch]:
                maxs[ch] = c_max
            sums[ch] += float(data.sum())
            sq_sums[ch] += float((data ** 2).sum())
            
    means = sums / total_pixels
    stds = np.sqrt(np.maximum((sq_sums / total_pixels) - (means ** 2), 1e-8))
    
    stats = {
        "channel_names": ENGINEERED_CHANNEL_NAMES,
        "n_channels": n_channels,
        "means": [float(m) for m in means],
        "stds": [float(s) for s in stds],
        "mins": [float(m) for m in mins],
        "maxs": [float(m) for m in maxs],
        "computed_from": "train_split_only",
        "n_train_images": len(train_ids),
        "split_seed": seed,
        "description": "Per-channel dataset-wide statistics computed strictly on the training split."
    }
    return stats

def normalize_engineered_features(
    feat_img: np.ndarray,
    stats: Union[Dict[str, Any], List[float], np.ndarray],
    stds_or_method: Union[str, List[float], np.ndarray] = "zscore",
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalizes engineered features using dataset-wide statistics computed from the train set.
    Supports either stats dict or direct (means, stds) arguments.
    Returns: (C, H, W) float32 array
    """
    img = feat_img.astype(np.float32)
    
    if isinstance(stats, dict):
        m = method if isinstance(stds_or_method, str) else "zscore"
        if m == "minmax":
            mins = np.array(stats["mins"], dtype=np.float32)
            maxs = np.array(stats["maxs"], dtype=np.float32)
            denom = np.maximum(maxs - mins, 1e-6)
            normalized = np.clip((img - mins) / denom, 0.0, 1.0)
        else:  # zscore
            means = np.array(stats["means"], dtype=np.float32)
            stds = np.array(stats["stds"], dtype=np.float32)
            normalized = (img - means) / stds
    else:
        means = np.array(stats, dtype=np.float32)
        stds = np.array(stds_or_method, dtype=np.float32)
        normalized = (img - means) / stds
        
    # Transpose (H, W, C) -> (C, H, W)
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

if __name__ == "__main__":
    train_csv = Path("water_segmentation/data/splits/train.csv")
    df_train = pd.read_csv(train_csv)
    train_ids = df_train["image_id"].tolist()
    
    stats = compute_engineered_train_statistics(
        image_dir="water_segmentation/data/raw/images",
        train_ids=train_ids,
        seed=42
    )
    
    out_file = Path("water_segmentation/data/statistics/engineered_train_stats.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    print("=" * 65)
    print("   ENGINEERED HYDROLOGICAL FEATURE STACK (TRAIN-ONLY STATS)")
    print("=" * 65)
    for i, name in enumerate(stats["channel_names"]):
        m = stats["means"][i]
        s = stats["stds"][i]
        mn = stats["mins"][i]
        mx = stats["maxs"][i]
        print(f"Ch {i:02d} ({name:18s}): Mean={m:9.3f}, Std={s:9.3f}, Min={mn:9.2f}, Max={mx:9.2f}")
    print("=" * 65)
    print(f"Saved stats to: {out_file}")
