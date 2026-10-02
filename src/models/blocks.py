"""
Building Blocks for Multispectral U-Net Architecture.
Contains DoubleConv, Down, Up (ConvTranspose2d), and OutConv modules
with configurable normalization (GroupNorm / BatchNorm).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two sequential blocks of [Conv2d -> Normalization -> ReLU].
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Optional intermediate channel count (defaults to out_channels).
        norm_type: Normalization type, 'group' (default) or 'batch'.
        num_groups: Number of groups for GroupNorm (default: 8).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm_type: str = "group",
        num_groups: int = 8
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        def get_norm(channels: int) -> nn.Module:
            if norm_type.lower() == "group":
                # Ensure channels is divisible by num_groups
                groups = min(num_groups, channels)
                while channels % groups != 0 and groups > 1:
                    groups -= 1
                return nn.GroupNorm(groups, channels)
            elif norm_type.lower() == "batch":
                return nn.BatchNorm2d(channels)
            elif norm_type.lower() == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            get_norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block (Hu et al., 2018).
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between spectral and feature channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights

class SEResBlock(nn.Module):
    """
    Residual Convolutional Block with Squeeze-and-Excitation Channel Attention.
    Computes: Output = ReLU( Shortcut(X) + SE(Conv2(Conv1(X))) )
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 8,
        reduction: int = 16
    ):
        super().__init__()
        
        def get_norm(channels: int) -> nn.Module:
            if norm_type.lower() == "group":
                groups = min(num_groups, channels)
                while channels % groups != 0 and groups > 1:
                    groups -= 1
                return nn.GroupNorm(groups, channels)
            elif norm_type.lower() == "batch":
                return nn.BatchNorm2d(channels)
            else:
                return nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + res)
        return out

class Down(nn.Module):
    """Downscaling with MaxPool2d(2) followed by DoubleConv."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 8
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling via ConvTranspose2d, concatenating skip connection, then DoubleConv."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        norm_type: str = "group",
        num_groups: int = 8
    ):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_type=norm_type, num_groups=num_groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, num_groups=num_groups)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 Convolution to project feature channels to class logits. NO SIGMOID."""
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
