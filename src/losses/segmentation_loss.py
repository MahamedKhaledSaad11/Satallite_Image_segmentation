"""
Hybrid Loss Functions for Binary Water Segmentation.
Combines numerically stable BCEWithLogitsLoss with soft DiceLoss to optimize
both pixel-level classification and region overlap under class imbalance.
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss operating on raw logits.
    
    Args:
        smooth: Smoothing epsilon to prevent division by zero (default: 1.0).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw unnormalized logits
            targets: (B, 1, H, W) binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCEWithLogitsLoss and DiceLoss.
    
    Loss = bce_weight * BCE(logits, targets) + dice_weight * Dice(sigmoid(logits), targets)
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: float = None
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_val = self.bce(logits, targets)
        dice_val = self.dice(logits, targets)
        return self.bce_weight * bce_val + self.dice_weight * dice_val

def get_loss_function(config_loss: dict) -> nn.Module:
    """Factory function to build loss from configuration dictionary."""
    name = config_loss.get("name", "bce_dice").lower()
    bce_w = config_loss.get("bce_weight", 0.5)
    dice_w = config_loss.get("dice_weight", 0.5)
    smooth = config_loss.get("smooth", 1.0)
    pos_w = config_loss.get("pos_weight", None)

    if name in ["bce_dice", "bcedice"]:
        return BCEDiceLoss(bce_weight=bce_w, dice_weight=dice_w, smooth=smooth, pos_weight=pos_w)
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        return DiceLoss(smooth=smooth)
    else:
        raise ValueError(f"Unsupported loss name: {name}")
