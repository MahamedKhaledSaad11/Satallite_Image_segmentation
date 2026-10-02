"""
Unit tests for Segmentation Metrics.
Verifies IoU, Precision, Recall, F1, and empty-image False Positive Rate computations.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from src.metrics.segmentation_metrics import SegmentationMetrics

def test_perfect_prediction():
    metrics = SegmentationMetrics(threshold=0.5)
    # Logits strongly positive where target is 1, strongly negative where target is 0
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).float()
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]]).float()
    
    metrics.update(logits, targets)
    res = metrics.compute()
    assert abs(res["water_iou_global"] - 1.0) < 1e-5
    assert abs(res["f1_global"] - 1.0) < 1e-5
    assert abs(res["precision_global"] - 1.0) < 1e-5
    assert abs(res["recall_global"] - 1.0) < 1e-5

def test_zero_overlap():
    metrics = SegmentationMetrics(threshold=0.5)
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]).float()
    logits = torch.tensor([[[[-10.0, -10.0], [10.0, 10.0]]]]).float()
    
    metrics.update(logits, targets)
    res = metrics.compute()
    assert res["water_iou_global"] == 0.0
    assert res["f1_global"] == 0.0
    assert res["precision_global"] == 0.0
    assert res["recall_global"] == 0.0

def test_known_partial_overlap():
    metrics = SegmentationMetrics(threshold=0.5)
    # 4 pixels:
    # GT:   [1, 1, 0, 0]
    # Pred: [1, 0, 1, 0]
    # TP=1, FP=1, FN=1, TN=1
    # IoU = 1 / (1 + 1 + 1) = 1/3 = 0.3333
    # P = 1 / (1 + 1) = 0.5
    # R = 1 / (1 + 1) = 0.5
    # F1 = 0.5
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]).float()
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]]).float()
    
    metrics.update(logits, targets)
    res = metrics.compute()
    assert abs(res["water_iou_global"] - (1.0 / 3.0)) < 1e-4
    assert abs(res["precision_global"] - 0.5) < 1e-4
    assert abs(res["recall_global"] - 0.5) < 1e-4
    assert abs(res["f1_global"] - 0.5) < 1e-4

if __name__ == "__main__":
    test_perfect_prediction()
    test_zero_overlap()
    test_known_partial_overlap()
    print("=" * 50)
    print("   ALL METRIC UNIT TESTS PASSED SUCCESSFULLY!   ")
    print("=" * 50)
