"""
Modular Training Engine for Multispectral Segmentation.
Orchestrates training loops, validation evaluation, learning rate scheduling,
metric tracking, model checkpointing, and early stopping.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics.segmentation_metrics import SegmentationMetrics
from src.training.checkpoint import save_checkpoint

class Trainer:
    """
    Main training controller.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 12)
        self.monitor_metric = train_cfg.get("early_stopping_metric", "water_iou_global")
        self.monitor_mode = train_cfg.get("early_stopping_mode", "max")

        ckpt_cfg = config.get("checkpoint", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.save_dir / "best_model.pt"

        self.best_metric_val = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "lr": []
        }

    def train_epoch(self) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Run validation evaluation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)
        metrics_calc = SegmentationMetrics(threshold=self.config.get("evaluation", {}).get("threshold", 0.5))

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

            metrics_calc.update(logits, masks)

        val_metrics = metrics_calc.compute()
        val_metrics["val_loss"] = total_loss / max(1, n_batches)
        return val_metrics

    def fit(self) -> Dict[str, Any]:
        """Execute complete training loop."""
        print("=" * 90)
        print(f"Starting training on {self.device} for up to {self.max_epochs} epochs...")
        print(f"Monitoring metric '{self.monitor_metric}' ({self.monitor_mode}) with patience={self.early_stopping_patience}")
        print("=" * 90)

        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()
            elapsed = time.time() - t0

            val_loss = val_metrics["val_loss"]
            current_metric = val_metrics.get(self.monitor_metric, 0.0)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_iou"].append(val_metrics["water_iou_global"])
            self.history["val_f1"].append(val_metrics["f1_global"])
            self.history["val_precision"].append(val_metrics["precision_global"])
            self.history["val_recall"].append(val_metrics["recall_global"])
            self.history["lr"].append(current_lr)

            # Step LR scheduler (ReduceLROnPlateau)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau step depends on metric or val_loss
                    plateau_val = val_loss if self.monitor_mode == "min" else current_metric
                    self.scheduler.step(plateau_val)
                else:
                    self.scheduler.step()

            # Check if this epoch is the new best
            is_best = False
            if self.monitor_mode == "max":
                if current_metric > self.best_metric_val:
                    is_best = True
            else:
                if current_metric < self.best_metric_val:
                    is_best = True

            best_tag = ""
            if is_best:
                self.best_metric_val = current_metric
                self.epochs_without_improvement = 0
                best_epoch = epoch
                best_tag = " [*] Best"
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=self.config,
                    save_path=str(self.best_checkpoint_path)
                )
            else:
                self.epochs_without_improvement += 1

            # Log line per plan Section 13.3
            print(
                f"Epoch [{epoch:02d}/{self.max_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {val_metrics['water_iou_global']:.4f} | "
                f"P: {val_metrics['precision_global']:.4f} | "
                f"R: {val_metrics['recall_global']:.4f} | "
                f"F1: {val_metrics['f1_global']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"({elapsed:.1f}s){best_tag}"
            )

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n[Early Stopping Triggered] No improvement in '{self.monitor_metric}' for {self.early_stopping_patience} consecutive epochs.")
                break

        print("=" * 90)
        print(f"Training completed. Best epoch: {best_epoch} with {self.monitor_metric}: {self.best_metric_val:.4f}")
        print(f"Best checkpoint saved to: {self.best_checkpoint_path}")
        print("=" * 90)

        return {
            "best_epoch": best_epoch,
            "best_metric": self.best_metric_val,
            "history": self.history
        }
