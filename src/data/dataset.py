"""
PyTorch Dataset for Multispectral Satellite Water Segmentation.
Handles GeoTIFF reading with tifffile, ground truth mask loading,
per-band train-only normalization, NoData fill, spatial augmentations,
and optional band subsetting for ablation studies.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_image, load_train_statistics
from src.data.feature_engineering import build_engineered_features, normalize_engineered_features

class MultispectralWaterDataset(Dataset):
    """
    Multispectral satellite image dataset for water segmentation.
    Supports both raw spectral band loading and domain-specific feature engineering.
    """
    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        image_ids: Union[List[int], str, Path],
        band_means: Union[np.ndarray, List[float]],
        band_stds: Union[np.ndarray, List[float]],
        nodata_fill: Optional[Dict[str, float]] = None,
        augmentation: Optional[Callable] = None,
        band_indices: Optional[List[int]] = None,
        use_feature_engineering: bool = False,
        norm_method: str = "zscore",
        engineered_stats: Optional[Dict[str, Any]] = None
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        
        if isinstance(image_ids, (str, Path)):
            df = pd.read_csv(image_ids)
            self.image_ids = df["image_id"].astype(int).tolist()
        else:
            self.image_ids = [int(i) for i in image_ids]
            
        self.band_means = np.asarray(band_means, dtype=np.float32)
        self.band_stds = np.asarray(band_stds, dtype=np.float32)
        self.nodata_fill = nodata_fill
        self.augmentation = augmentation
        self.band_indices = band_indices
        self.use_feature_engineering = use_feature_engineering
        self.norm_method = norm_method
        self.engineered_stats = engineered_stats

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.tif"
        lbl_path = self.label_dir / f"{img_id}.png"
        
        # 1. Read image with tifffile (H, W, C) int16
        raw_image = tifffile.imread(str(img_path))
        
        # 2. Read mask with PIL (H, W) uint8
        raw_mask = np.array(Image.open(str(lbl_path)), dtype=np.uint8)
        
        # Ensure binary {0, 1}
        raw_mask = (raw_mask > 0).astype(np.float32)
        
        # 3. Preprocess image
        if self.use_feature_engineering and self.engineered_stats is not None:
            feat_img = build_engineered_features(raw_image)
            image_chw = normalize_engineered_features(
                feat_img=feat_img,
                stats=self.engineered_stats,
                method=self.norm_method
            )
        else:
            image_chw = preprocess_image(
                image=raw_image,
                band_means=self.band_means,
                band_stds=self.band_stds,
                nodata_fill=self.nodata_fill
            )
        
        # 4. Optional band selection (ablation studies)
        if self.band_indices is not None:
            image_chw = image_chw[self.band_indices, :, :]
            
        # 5. Apply spatial augmentations synchronously
        if self.augmentation is not None:
            image_chw, raw_mask = self.augmentation(image_chw, raw_mask)
            
        # 6. Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image_chw).float()
        # Add channel dimension to mask: (1, H, W)
        mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor
