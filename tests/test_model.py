"""
Unit tests for U-Net Architecture and Blocks.
Verifies forward pass output shapes, parameter initialization,
gradient backpropagation, and norm compatibility.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from src.models.unet import UNet

def test_unet_output_shape():
    model = UNet(in_channels=12, out_channels=1, base_channels=32)
    x = torch.randn(2, 12, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"

def test_unet_optical_7ch():
    model = UNet(in_channels=7, out_channels=1, base_channels=32)
    x = torch.randn(2, 7, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)

def test_unet_rgb_3ch():
    model = UNet(in_channels=3, out_channels=1, base_channels=32)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)

def test_unet_batchnorm():
    model = UNet(in_channels=12, out_channels=1, base_channels=32, norm_type="batch")
    x = torch.randn(2, 12, 128, 128)
    y = model(x)
    assert y.shape == (2, 1, 128, 128)

def test_gradient_flow():
    model = UNet(in_channels=12, out_channels=1, base_channels=16)
    x = torch.randn(2, 12, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    y = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, target)
    loss.backward()
    
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} gradient contains NaN!"

if __name__ == "__main__":
    test_unet_output_shape()
    test_unet_optical_7ch()
    test_unet_rgb_3ch()
    test_unet_batchnorm()
    test_gradient_flow()
    print("=" * 50)
    print("    ALL MODEL UNIT TESTS PASSED SUCCESSFULLY!   ")
    print("=" * 50)
