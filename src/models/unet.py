"""
Standard U-Net Architecture from Scratch for Multispectral Water Segmentation.
Supports configurable input channels (e.g. 12 channels for baseline, 7 for optical, 3 for RGB),
flexible base feature channel capacity, and GroupNorm / BatchNorm normalization.
Returns raw logits without Sigmoid activation for numerical stability with BCEWithLogitsLoss.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Optional
import torch
import torch.nn as nn

from src.models.blocks import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015).
    Adapted for multispectral earth observation satellite imagery.
    
    Args:
        in_channels: Number of input spectral bands (default: 12).
        out_channels: Number of output segmentation masks (default: 1 for binary water).
        base_channels: Number of feature maps in the first layer (default: 32).
        bilinear: Whether to use bilinear upsampling instead of transposed convolution (default: False).
        norm_type: 'group' (default) or 'batch'. GroupNorm is strongly recommended for small batches.
        num_groups: Group count for GroupNorm (default: 8).
    """
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
        norm_type: str = "group",
        num_groups: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.norm_type = norm_type

        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, norm_type=norm_type, num_groups=num_groups)
        self.down1 = Down(base_channels, base_channels * 2, norm_type=norm_type, num_groups=num_groups)
        self.down2 = Down(base_channels * 2, base_channels * 4, norm_type=norm_type, num_groups=num_groups)
        self.down3 = Down(base_channels * 4, base_channels * 8, norm_type=norm_type, num_groups=num_groups)
        self.down4 = Down(base_channels * 8, (base_channels * 16) // factor, norm_type=norm_type, num_groups=num_groups)

        # Decoder
        self.up1 = Up(base_channels * 16, (base_channels * 8) // factor, bilinear=bilinear, norm_type=norm_type, num_groups=num_groups)
        self.up2 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear=bilinear, norm_type=norm_type, num_groups=num_groups)
        self.up3 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear=bilinear, norm_type=norm_type, num_groups=num_groups)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear, norm_type=norm_type, num_groups=num_groups)
        
        # Head (produces unnormalized logits)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, in_channels, H, W) tensor
        Returns:
            logits: (B, out_channels, H, W) tensor (raw logits, NO SIGMOID)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SEResUNet(nn.Module):
    """
    Optimized Residual U-Net with Squeeze-and-Excitation Channel Attention.
    Specifically designed for multispectral satellite earth observation to dynamically
    weight the most discriminative spectral bands and engineered hydrological indices.
    """
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        base_channels: int = 32,
        norm_type: str = "group",
        num_groups: int = 8,
        reduction: int = 16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        from src.models.blocks import SEResBlock

        # Encoder with Residual + Channel Attention
        self.inc = SEResBlock(in_channels, base_channels, norm_type=norm_type, num_groups=num_groups, reduction=reduction)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            SEResBlock(base_channels, base_channels * 2, norm_type=norm_type, num_groups=num_groups, reduction=reduction)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            SEResBlock(base_channels * 2, base_channels * 4, norm_type=norm_type, num_groups=num_groups, reduction=reduction)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            SEResBlock(base_channels * 4, base_channels * 8, norm_type=norm_type, num_groups=num_groups, reduction=reduction)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            SEResBlock(base_channels * 8, base_channels * 16, norm_type=norm_type, num_groups=num_groups, reduction=reduction)
        )

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear=False, norm_type=norm_type, num_groups=num_groups)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear=False, norm_type=norm_type, num_groups=num_groups)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear=False, norm_type=norm_type, num_groups=num_groups)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=False, norm_type=norm_type, num_groups=num_groups)

        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(config_model: Optional[dict] = None, **kwargs) -> nn.Module:
    """Builds model architecture from configuration dict or keyword arguments."""
    cfg = dict(config_model) if config_model is not None else {}
    cfg.update(kwargs)
    name = str(cfg.get("name", cfg.get("model_name", "unet"))).lower()
    in_ch = int(cfg.get("in_channels", 12))
    out_ch = int(cfg.get("out_channels", 1))
    base_ch = int(cfg.get("base_channels", 32))
    norm = str(cfg.get("norm_type", "group"))
    groups = int(cfg.get("num_groups", 8))

    if name in ["se_res_unet", "seresunet"]:
        return SEResUNet(in_channels=in_ch, out_channels=out_ch, base_channels=base_ch, norm_type=norm, num_groups=groups)
    else:
        bilinear = bool(cfg.get("bilinear", False))
        return UNet(in_channels=in_ch, out_channels=out_ch, base_channels=base_ch, bilinear=bilinear, norm_type=norm, num_groups=groups)

if __name__ == "__main__":
    model = UNet(in_channels=12, out_channels=1, base_channels=32, norm_type="group")
    dummy = torch.randn(2, 12, 128, 128)
    out = model(dummy)
    print("=" * 50)
    print("           U-NET ARCHITECTURE SANITY CHECK")
    print("=" * 50)
    print(f"Input shape:       {dummy.shape}")
    print(f"Output shape:      {out.shape}")
    print(f"Total Parameters:  {model.count_parameters():,}")
    print("=" * 50)
    assert out.shape == (2, 1, 128, 128), f"Unexpected shape {out.shape}"
    print("Sanity check PASSED successfully!")
