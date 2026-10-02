"""
Visualization Utilities for Remote Sensing Flood Water Segmentation.
Generates True Color RGB composites, ground truth overlays, predicted segmentations,
and color-coded error confusion maps:
- Green: True Positive (Correctly detected water)
- Red:   False Positive (False alarm / over-segmentation)
- Blue:  False Negative (Missed flood water)
- Light Gray / Black: True Negative (Background)
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

def create_rgb_composite(
    image: np.ndarray,
    r_idx: Union[int, np.ndarray] = 3,
    g_idx: Union[int, np.ndarray] = 2,
    b_idx: int = 1
) -> np.ndarray:
    """
    Constructs a normalized RGB composite from specified spectral channels.
    Supports:
      - create_rgb_composite(img_hwc_or_chw, r_idx=3, g_idx=2, b_idx=1)
      - create_rgb_composite(r_2d, g_2d, b_2d)
    Applies 2% to 98% percentile contrast stretching.
    """
    if isinstance(r_idx, np.ndarray) and isinstance(g_idx, np.ndarray):
        r = image.astype(np.float32)
        g = r_idx.astype(np.float32)
        b = g_idx.astype(np.float32)
    else:
        r_i = int(r_idx)
        g_i = int(g_idx)
        b_i = int(b_idx)
        if image.ndim == 3:
            if image.shape[0] <= 16:  # CHW format
                r = image[r_i].astype(np.float32)
                g = image[g_i].astype(np.float32)
                b = image[b_i].astype(np.float32)
            else:  # HWC format
                r = image[:, :, r_i].astype(np.float32)
                g = image[:, :, g_i].astype(np.float32)
                b = image[:, :, b_i].astype(np.float32)
        else:
            raise ValueError(f"Expected 3D array, got shape {image.shape}")

    stack = np.stack([r, g, b], axis=-1)
    for c in range(3):
        p2, p98 = np.percentile(stack[:, :, c], 2), np.percentile(stack[:, :, c], 98)
        denom = p98 - p2 if (p98 - p2) > 1e-5 else 1.0
        stack[:, :, c] = np.clip((stack[:, :, c] - p2) / denom, 0.0, 1.0)
    return stack

def create_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Creates an RGB error confusion map:
      - Green  [0, 220, 0]:   TP
      - Red    [220, 0, 0]:   FP
      - Blue   [0, 100, 240]: FN
      - Gray   [235, 235, 235]: TN
    """
    h, w = target.shape
    err_rgb = np.full((h, w, 3), 235, dtype=np.uint8)

    p_bool = (pred == 1)
    t_bool = (target == 1)

    # TP: Green
    err_rgb[p_bool & t_bool] = [46, 204, 113]
    # FP: Red
    err_rgb[p_bool & (~t_bool)] = [231, 76, 60]
    # FN: Blue
    err_rgb[(~p_bool) & t_bool] = [52, 152, 219]

    return err_rgb

def plot_prediction_comparison(
    image_raw_chw: np.ndarray,
    target_hw: np.ndarray,
    pred_hw: np.ndarray,
    title: str = "Segmentation Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a 4-panel comparison:
      1. True Color RGB
      2. Ground Truth Mask
      3. Predicted Mask
      4. Color-coded Error Map
    """
    rgb = create_rgb_composite(image_raw_chw, 3, 2, 1)
    err = create_error_map(pred_hw, target_hw)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("True Color RGB (B3, B2, B1)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(target_hw, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(pred_hw, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("U-Net Prediction", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(err)
    axes[3].set_title("Error Map (Green:TP, Red:FP, Blue:FN)", fontsize=10)
    axes[3].axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
