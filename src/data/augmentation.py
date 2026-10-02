"""
Spatial-Only Data Augmentation for Multispectral Imagery.
Applies strictly geometric transformations (horizontal flip, vertical flip, random 90-degree rotations)
identically across all spectral channels and ground truth masks.
Spectral distortion (color jitter, hue shifts) is strictly forbidden to preserve physical reflectance values.
"""

from typing import Tuple, Optional
import numpy as np

class SpatialAugmentation:
    """
    Random spatial augmentations applied synchronously to multispectral image and segmentation mask.
    
    Args:
        hflip_prob: Probability of horizontal flip (default: 0.5)
        vflip_prob: Probability of vertical flip (default: 0.5)
        rot90_prob: Probability of 90-degree rotation (default: 0.5)
    """
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: (C, H, W) float32 numpy array
            mask: (H, W) or (1, H, W) uint8 / float32 numpy array
        Returns:
            Tuple of transformed (image, mask) with identical shapes and dtypes.
        """
        # Horizontal flip (axis -1 = width)
        if np.random.rand() < self.hflip_prob:
            image = np.flip(image, axis=-1)
            mask = np.flip(mask, axis=-1)

        # Vertical flip (axis -2 = height)
        if np.random.rand() < self.vflip_prob:
            image = np.flip(image, axis=-2)
            mask = np.flip(mask, axis=-2)

        # Random 90 degree rotation
        if np.random.rand() < self.rot90_prob:
            k = np.random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(-2, -1))
            mask = np.rot90(mask, k=k, axes=(-2, -1))

        # Ensure contiguous memory layout after flips/rotations
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        return image, mask
