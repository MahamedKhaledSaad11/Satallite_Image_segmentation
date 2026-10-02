"""
Comprehensive Unit Tests for MultispectralWaterDataset and Data Pipeline.
Validates shapes, dtypes, normalization ranges, NoData handling,
batch collation, and spatial augmentation consistency.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultispectralWaterDataset
from src.data.preprocessing import load_train_statistics
from src.data.augmentation import SpatialAugmentation

def get_test_dataset(augment=False, band_indices=None):
    base_dir = Path("water_segmentation") if Path("water_segmentation").exists() else Path(".")
    stats_file = base_dir / "data" / "statistics" / "train_stats.json"
    train_csv = base_dir / "data" / "splits" / "train.csv"
    img_dir = base_dir / "data" / "raw" / "images"
    lbl_dir = base_dir / "data" / "raw" / "labels"
    
    stats = load_train_statistics(str(stats_file))
    aug = SpatialAugmentation(hflip_prob=1.0, vflip_prob=1.0, rot90_prob=1.0) if augment else None
    
    ds = MultispectralWaterDataset(
        image_dir=img_dir,
        label_dir=lbl_dir,
        image_ids=train_csv,
        band_means=stats["band_means"],
        band_stds=stats["band_stds"],
        nodata_fill=stats["nodata_fill"],
        augmentation=aug,
        band_indices=band_indices
    )
    return ds

def test_image_and_mask_shapes():
    ds = get_test_dataset()
    img, mask = ds[0]
    assert img.shape == (12, 128, 128), f"Expected (12, 128, 128), got {img.shape}"
    assert mask.shape == (1, 128, 128), f"Expected (1, 128, 128), got {mask.shape}"

def test_image_and_mask_dtypes():
    ds = get_test_dataset()
    img, mask = ds[0]
    assert img.dtype == torch.float32, f"Expected torch.float32 for image, got {img.dtype}"
    assert mask.dtype == torch.float32, f"Expected torch.float32 for mask, got {mask.dtype}"

def test_mask_binary_values():
    ds = get_test_dataset()
    for idx in range(min(10, len(ds))):
        _, mask = ds[idx]
        unique_vals = set(torch.unique(mask).tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Mask {idx} contains non-binary values: {unique_vals}"

def test_no_nan_no_inf():
    ds = get_test_dataset()
    for idx in range(min(10, len(ds))):
        img, mask = ds[idx]
        assert not torch.isnan(img).any(), f"Image {idx} contains NaN"
        assert not torch.isinf(img).any(), f"Image {idx} contains Inf"
        assert not torch.isnan(mask).any(), f"Mask {idx} contains NaN"
        assert not torch.isinf(mask).any(), f"Mask {idx} contains Inf"

def test_normalization_range():
    ds = get_test_dataset()
    # Over first 20 images, mean of normalized pixels should be close to 0
    all_means = []
    for idx in range(min(20, len(ds))):
        img, _ = ds[idx]
        all_means.append(img.mean(dim=(1, 2)).numpy())
    avg_means = np.mean(all_means, axis=0)
    for b in range(12):
        assert abs(avg_means[b]) < 1.5, f"Band {b} normalized mean {avg_means[b]} is far from 0"

def test_nodata_fill_handled():
    ds = get_test_dataset()
    # Image 0 or images with Merit DEM -9999 should not have any -9999 in Band 8
    for idx in range(min(20, len(ds))):
        img, _ = ds[idx]
        # In normalized space, raw -9999 would be (-9999 - 280) / 426 = ~ -24
        # Normalized band 8 should have all values > -15
        assert (img[8] > -15.0).all(), f"Image {idx} contains unhandled NoData values in Band 8!"

def test_batch_collation():
    ds = get_test_dataset()
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch_img, batch_mask = next(iter(loader))
    assert batch_img.shape == (8, 12, 128, 128), f"Batch img shape mismatch: {batch_img.shape}"
    assert batch_mask.shape == (8, 1, 128, 128), f"Batch mask shape mismatch: {batch_mask.shape}"

def test_ablation_band_selection():
    # 7 optical bands: [0, 1, 2, 3, 4, 5, 6]
    ds_7 = get_test_dataset(band_indices=[0, 1, 2, 3, 4, 5, 6])
    img_7, mask = ds_7[0]
    assert img_7.shape == (7, 128, 128), f"Expected 7 channels, got {img_7.shape}"
    
    # 3 RGB bands: [1, 2, 3]
    ds_rgb = get_test_dataset(band_indices=[1, 2, 3])
    img_rgb, _ = ds_rgb[0]
    assert img_rgb.shape == (3, 128, 128), f"Expected 3 channels, got {img_rgb.shape}"

def test_augmentation_consistency():
    aug = SpatialAugmentation(hflip_prob=1.0, vflip_prob=0.0, rot90_prob=0.0)
    dummy_img = np.zeros((12, 128, 128), dtype=np.float32)
    dummy_mask = np.zeros((128, 128), dtype=np.float32)
    # Set a marker at top-left
    dummy_img[:, 0:10, 0:10] = 5.0
    dummy_mask[0:10, 0:10] = 1.0
    
    aug_img, aug_mask = aug(dummy_img, dummy_mask)
    # After horizontal flip, top-left should move to top-right (width axis 118:128)
    assert (aug_img[:, 0:10, 118:128] == 5.0).all()
    assert (aug_mask[0:10, 118:128] == 1.0).all()
    assert (aug_img[:, 0:10, 0:10] == 0.0).all()
    assert (aug_mask[0:10, 0:10] == 0.0).all()

if __name__ == "__main__":
    test_image_and_mask_shapes()
    test_image_and_mask_dtypes()
    test_mask_binary_values()
    test_no_nan_no_inf()
    test_normalization_range()
    test_nodata_fill_handled()
    test_batch_collation()
    test_ablation_band_selection()
    test_augmentation_consistency()
    print("=" * 50)
    print("  ALL 9 DATASET UNIT TESTS PASSED SUCCESSFULLY! ")
    print("=" * 50)
