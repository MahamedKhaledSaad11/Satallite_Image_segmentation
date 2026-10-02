"""
Evaluation Metrics for Binary Water Segmentation.
Accumulates True Positives, False Positives, False Negatives, and True Negatives
across entire batches or epochs to compute unbiased global and per-image metrics:
- Water IoU (Jaccard Index) - Primary metric
- Precision & Recall
- F1-Score (Dice coefficient)
- False Positive Rate on empty-water images
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

class SegmentationMetrics:
    """
    Running accumulator for evaluation metrics.
    
    Args:
        threshold: Probability threshold for water classification (default: 0.5).
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.per_image_ious: List[float] = []
        self.empty_gt_fp_pixels = 0
        self.empty_gt_total_pixels = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update running metrics with a batch of predictions and targets.
        
        Args:
            logits: (B, 1, H, W) raw unnormalized logits from model
            targets: (B, 1, H, W) ground truth binary masks {0, 1}
        """
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).bool()
        targs = (targets.to(logits.device) > 0.5).bool()

        # Batch-level global accumulation
        tp = (preds & targs).sum().item()
        fp = (preds & (~targs)).sum().item()
        fn = ((~preds) & targs).sum().item()
        tn = ((~preds) & (~targs)).sum().item()

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn

        # Per-image calculation for distribution analysis
        B = logits.size(0)
        for i in range(B):
            p_img = preds[i, 0]
            t_img = targs[i, 0]
            
            img_tp = (p_img & t_img).sum().item()
            img_fp = (p_img & (~t_img)).sum().item()
            img_fn = ((~p_img) & t_img).sum().item()
            
            denom = img_tp + img_fp + img_fn
            if denom == 0:
                # Both target and prediction are empty: perfect match
                iou = 1.0
            else:
                iou = img_tp / denom
            self.per_image_ious.append(iou)

            # Check if this image had no water in ground truth
            if t_img.sum().item() == 0:
                self.empty_gt_fp_pixels += img_fp
                self.empty_gt_total_pixels += p_img.numel()

    def compute(self) -> Dict[str, float]:
        """
        Compute aggregate metrics from accumulated counts.
        """
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        tn = self.total_tn

        denom_iou = tp + fp + fn
        water_iou_global = tp / denom_iou if denom_iou > 0 else 1.0

        denom_p = tp + fp
        precision_global = tp / denom_p if denom_p > 0 else 0.0

        denom_r = tp + fn
        recall_global = tp / denom_r if denom_r > 0 else 0.0

        denom_f1 = precision_global + recall_global
        f1_global = (2.0 * precision_global * recall_global) / denom_f1 if denom_f1 > 0 else 0.0

        water_iou_mean = float(np.mean(self.per_image_ious)) if self.per_image_ious else 0.0
        empty_image_fpr = (self.empty_gt_fp_pixels / self.empty_gt_total_pixels) if self.empty_gt_total_pixels > 0 else 0.0

        return {
            "water_iou_global": float(water_iou_global),
            "precision_global": float(precision_global),
            "recall_global": float(recall_global),
            "f1_global": float(f1_global),
            "water_iou_mean": float(water_iou_mean),
            "empty_image_fpr": float(empty_image_fpr),
        }

@torch.no_grad()
def find_optimal_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates validation set across candidate thresholds to find optimal threshold for Water IoU.
    
    Returns:
        best_threshold: Optimal threshold float
        best_metrics: Dict of metrics at optimal threshold
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    model.eval()
    all_logits = []
    all_targets = []

    for images, masks in val_loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    best_thresh = 0.5
    best_iou = -1.0
    best_metrics = {}

    for t in thresholds:
        metrics_calc = SegmentationMetrics(threshold=t)
        metrics_calc.update(all_logits, all_targets)
        res = metrics_calc.compute()
        if res["water_iou_global"] > best_iou:
            best_iou = res["water_iou_global"]
            best_thresh = t
            best_metrics = res

    return float(best_thresh), best_metrics
