"""Trainer for downstream frozen vs. fine-tuned experiments."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.downstream.classifier import IntentClassifier
from src.utils.config import DownstreamConfig
from src.utils.logging_utils import log_experiment_result
from src.utils.metrics import compute_classification_metrics, get_confusion_matrix


class DownstreamTrainer:
    """Training/evaluation loop for SNIPS intent classification."""

    def __init__(
        self,
        model: IntentClassifier,
        cfg: DownstreamConfig,
        run_name: str,
        mode: str,
        logger,
        class_names: list[str],
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.run_name = run_name
        self.mode = mode
        self.logger = logger
        self.class_names = class_names
        self.device = self._resolve_device(cfg.device)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_optimizer_and_scheduler(
        self, total_steps: int
    ) -> tuple[torch.optim.Optimizer, LambdaLR]:
        """Build AdamW optimizer with differential LR and a linear warmup scheduler.

        Backbone parameters use ``cfg.finetune_lr`` (or ``cfg.frozen_lr`` when
        frozen); the classifier head is given a higher LR via
        ``cfg.head_lr_multiplier``.
        """
        if self.mode == "frozen":
            self.model.lm.eval()
            for p in self.model.lm.parameters():
                p.requires_grad = False
            backbone_lr = 0.0          # frozen — optimizer entry still needed
            head_lr = self.cfg.frozen_lr
        else:
            for p in self.model.lm.parameters():
                p.requires_grad = True
            backbone_lr = self.cfg.finetune_lr
            head_lr = self.cfg.finetune_lr * self.cfg.head_lr_multiplier

        # Split parameter groups: backbone vs. classifier head
        backbone_params = [p for p in self.model.lm.parameters() if p.requires_grad]
        head_params = list(self.model.classifier.parameters())

        param_groups = [
            {"params": head_params, "lr": head_lr},
        ]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        warmup = self.cfg.warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step + 1) / float(max(1, warmup))
            # Linear decay from 1.0 to 0.0 after warmup
            progress = float(current_step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 1.0 - progress)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        """Train with early stopping; return best validation metrics."""
        steps_per_epoch = (
            min(self.cfg.max_steps_per_epoch, len(train_loader))
            if self.cfg.max_steps_per_epoch
            else len(train_loader)
        )
        total_steps = steps_per_epoch * self.cfg.epochs
        optimizer, scheduler = self._build_optimizer_and_scheduler(total_steps)
        criterion = nn.CrossEntropyLoss()

        best: dict = {"accuracy": 0.0, "macro_f1": 0.0}
        patience_counter = 0
        global_step = 0

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.time()
            self.model.train()
            if self.mode == "frozen":
                self.model.lm.eval()
            total_loss = 0.0
            total_steps_epoch = 0

            for step, (input_ids, labels) in enumerate(train_loader, start=1):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids)          # [B, C]
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
                optimizer.step()
                scheduler.step()

                global_step += 1
                total_loss += float(loss.item())
                total_steps_epoch += 1

                if self.cfg.max_steps_per_epoch and step >= self.cfg.max_steps_per_epoch:
                    break

            metrics = self.evaluate(val_loader)
            epoch_time = time.time() - epoch_start
            train_loss = total_loss / max(total_steps_epoch, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            self.logger.info(
                "epoch=%d mode=%s train_loss=%.4f val_acc=%.4f val_macro_f1=%.4f lr=%.2e time=%.2fs",
                epoch,
                self.mode,
                train_loss,
                metrics["accuracy"],
                metrics["macro_f1"],
                current_lr,
                epoch_time,
            )
            log_experiment_result(
                run_name=self.run_name,
                epoch=epoch,
                metrics={
                    "mode": self.mode,
                    "val_acc": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                },
            )

            if metrics["accuracy"] > best["accuracy"]:
                best = metrics
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    self.logger.info(
                        "Early stopping at epoch=%d (no improvement for %d epochs)",
                        epoch,
                        self.cfg.early_stopping_patience,
                    )
                    break

        self._save_confusion_matrix(best["y_true"], best["y_pred"])
        return best

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on the full dataloader and return scalar + array metrics."""
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(input_ids)          # [B, C]
            preds = logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        metrics = compute_classification_metrics(y_true_arr, y_pred_arr)
        metrics["y_true"] = y_true_arr
        metrics["y_pred"] = y_pred_arr
        return metrics

    def _save_checkpoint(self) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir) / self.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "best_model.pt")

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        Path(self.cfg.figures_dir).mkdir(parents=True, exist_ok=True)
        cm = get_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(range(len(self.class_names))),
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix ({self.mode})")
        fig.tight_layout()
        output_path = Path(self.cfg.figures_dir) / f"{self.run_name}_cm.png"
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
