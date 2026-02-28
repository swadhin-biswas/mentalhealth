import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import gc
import os
import sys

from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import logging as transformers_logging

# Handle imports for both direct execution and module import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model_architecture import MentalHealthModel, create_model
from data_preprocessing import MentalHealthDataProcessor, create_data_loaders

transformers_logging.set_verbosity_error()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentalHealthLoss(nn.Module):
    """Combined loss function for mental health model training."""

    def __init__(
        self,
        hidden_size: int,
        crisis_weight: float = 2.0,
        quality_weight: float = 0.5,
        harmony_weight: float = 0.1,
        use_focal: bool = True,
    ):
        super().__init__()
        self.crisis_weight = crisis_weight
        self.quality_weight = quality_weight
        self.harmony_weight = harmony_weight
        self.use_focal = use_focal

        self.crisis_classifier = nn.Linear(hidden_size, 4)
        self.quality_predictor = nn.Linear(hidden_size, 1)

        if use_focal:
            self.focal_gamma = 2.0
            self.focal_alpha = 0.25

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def crisis_loss(
        self,
        hidden_states: torch.Tensor,
        crisis_labels: torch.Tensor,
        harm_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Loss for crisis detection."""
        logits = self.crisis_classifier(hidden_states)

        if self.use_focal:
            loss = self.focal_loss(logits, crisis_labels)
        else:
            loss = F.cross_entropy(logits, crisis_labels)

        if harm_labels is not None:
            harm_logits = hidden_states
            harm_loss = F.cross_entropy(harm_logits, harm_labels)
            loss = loss + 0.5 * harm_loss

        return loss

    def quality_loss(
        self, hidden_states: torch.Tensor, quality_targets: torch.Tensor
    ) -> torch.Tensor:
        """Loss for response quality prediction."""
        predictions = self.quality_predictor(hidden_states)
        return F.mse_loss(predictions.squeeze(-1), quality_targets)

    def harmony_loss(self, encoder_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Regularization loss for multi-encoder harmony."""
        if len(encoder_outputs) < 2:
            return torch.tensor(0.0, device=next(iter(encoder_outputs.values())).device)

        outputs = list(encoder_outputs.values())

        mean_outputs = [o.mean(dim=1) for o in outputs]

        harmony = 0.0
        for i in range(len(mean_outputs)):
            for j in range(i + 1, len(mean_outputs)):
                correlation = F.cosine_similarity(
                    mean_outputs[i], mean_outputs[j], dim=-1
                )
                harmony = harmony + (1 - correlation).mean()

        return harmony / (len(outputs) * (len(outputs) - 1) / 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_outputs: Dict[str, torch.Tensor],
        crisis_labels: torch.Tensor,
        quality_targets: torch.Tensor,
        harm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""

        c_loss = self.crisis_loss(hidden_states, crisis_labels, harm_labels)

        q_loss = self.quality_loss(hidden_states, quality_targets)

        h_loss = self.harmony_loss(encoder_outputs)

        total_loss = (
            self.crisis_weight * c_loss
            + self.quality_weight * q_loss
            + self.harmony_weight * h_loss
        )

        return {
            "total_loss": total_loss,
            "crisis_loss": c_loss,
            "quality_loss": q_loss,
            "harmony_loss": h_loss,
        }


class TrainingMetrics:
    """Track training and validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.crisis_acc = []
        self.quality_corr = []
        self.safety_scores = []
        self.epoch_times = []

    def update(self, metrics: Dict[str, float], phase: str = "train"):
        if phase == "train":
            self.train_losses.append(metrics.get("total_loss", 0))
        else:
            self.val_losses.append(metrics.get("total_loss", 0))

        if "crisis_acc" in metrics:
            self.crisis_acc.append(metrics["crisis_acc"])
        if "quality_corr" in metrics:
            self.quality_corr.append(metrics["quality_corr"])
        if "safety_score" in metrics:
            self.safety_scores.append(metrics["safety_score"])

    def get_summary(self) -> Dict[str, float]:
        return {
            "avg_train_loss": np.mean(self.train_losses[-10:])
            if self.train_losses
            else 0,
            "avg_val_loss": np.mean(self.val_losses[-10:]) if self.val_losses else 0,
            "crisis_accuracy": np.mean(self.crisis_acc[-10:]) if self.crisis_acc else 0,
            "quality_correlation": np.mean(self.quality_corr[-10:])
            if self.quality_corr
            else 0,
            "safety_score": np.mean(self.safety_scores[-10:])
            if self.safety_scores
            else 0,
        }


class MentalHealthTrainer:
    """Complete training pipeline for mental health support model."""

    def __init__(
        self,
        model: MentalHealthModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = MentalHealthLoss(
            model.hidden_size,
            crisis_weight=config.get("crisis_weight", 2.0),
            quality_weight=config.get("quality_weight", 0.5),
            harmony_weight=config.get("harmony_weight", 0.1),
        ).to(self.device)

        self.metrics = TrainingMetrics()

        self.best_val_loss = float("inf")
        self.patience = config.get("patience", 3)
        self.patience_counter = 0

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.gradient_accumulation = config.get("gradient_accumulation", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        self.current_epoch = 0

    def _create_optimizer(self):
        """Create optimizer with layer-wise learning rates."""
        optimizer_config = self.config.get("optimizer", {})

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": optimizer_config.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.get("learning_rate", 2e-5),
            betas=optimizer_config.get("betas", (0.9, 0.999)),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.get("num_epochs", 10)
        warmup_steps = int(total_steps * 0.1)

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_crisis_correct = 0
        total_crisis_samples = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            metadata = batch["metadata"]

            crisis_labels = self._get_crisis_labels(metadata)
            quality_labels = self._get_quality_labels(metadata)

            inputs = {"general": (input_ids, attention_mask)}

            outputs = self.model(inputs, return_safety=True, return_quality=True)

            hidden_states = outputs["hidden_states"]
            encoder_outputs = {"general": hidden_states}

            losses = self.loss_fn(
                hidden_states,
                encoder_outputs,
                crisis_labels.to(self.device),
                quality_labels.to(self.device),
            )

            loss = losses["total_loss"] / self.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += losses["total_loss"].item()

            with torch.no_grad():
                crisis_preds = torch.argmax(outputs["safety"]["crisis_probs"], dim=-1)
                total_crisis_correct += (
                    (crisis_preds == crisis_labels.to(self.device)).sum().item()
                )
                total_crisis_samples += crisis_labels.size(0)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)} - Loss: {losses['total_loss'].item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        crisis_acc = (
            total_crisis_correct / total_crisis_samples
            if total_crisis_samples > 0
            else 0
        )

        return {"total_loss": avg_loss, "crisis_acc": crisis_acc}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_crisis_correct = 0
        total_crisis_samples = 0
        all_safety_scores = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            metadata = batch["metadata"]

            crisis_labels = self._get_crisis_labels(metadata)
            quality_labels = self._get_quality_labels(metadata)

            inputs = {"general": (input_ids, attention_mask)}

            outputs = self.model(inputs, return_safety=True, return_quality=True)

            hidden_states = outputs["hidden_states"]
            encoder_outputs = {"general": hidden_states}

            losses = self.loss_fn(
                hidden_states,
                encoder_outputs,
                crisis_labels.to(self.device),
                quality_labels.to(self.device),
            )

            total_loss += losses["total_loss"].item()

            crisis_preds = torch.argmax(outputs["safety"]["crisis_probs"], dim=-1)
            total_crisis_correct += (
                (crisis_preds == crisis_labels.to(self.device)).sum().item()
            )
            total_crisis_samples += crisis_labels.size(0)

            safety_scores = outputs["safety"]["safety_level"].float().mean()
            all_safety_scores.append(safety_scores.item())

        avg_loss = total_loss / len(self.val_loader)
        crisis_acc = (
            total_crisis_correct / total_crisis_samples
            if total_crisis_samples > 0
            else 0
        )
        avg_safety = np.mean(all_safety_scores)

        return {
            "total_loss": avg_loss,
            "crisis_acc": crisis_acc,
            "safety_score": avg_safety,
        }

    def _get_crisis_labels(self, metadata: List[Dict]) -> torch.Tensor:
        """Extract crisis labels from metadata."""
        labels = []
        for m in metadata:
            if m.get("has_crisis", False):
                labels.append(3)
            else:
                labels.append(0)
        return torch.tensor(labels, dtype=torch.long)

    def _get_quality_labels(self, metadata: List[Dict]) -> torch.Tensor:
        """Extract quality labels from metadata."""
        labels = []
        for m in metadata:
            score = m.get("score", 0)
            normalized = min(score / 100.0, 1.0)
            labels.append(normalized)
        return torch.tensor(labels, dtype=torch.float)

    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        num_epochs = self.config.get("num_epochs", 10)

        logger.info(
            f"Starting training for {num_epochs} epochs on device: {self.device}"
        )

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_metrics = self.train_epoch()
            logger.info(
                f"Train - Loss: {train_metrics['total_loss']:.4f}, Crisis Acc: {train_metrics['crisis_acc']:.4f}"
            )

            val_metrics = self.validate()
            logger.info(
                f"Val - Loss: {val_metrics['total_loss']:.4f}, Crisis Acc: {val_metrics['crisis_acc']:.4f}, Safety: {val_metrics['safety_score']:.4f}"
            )

            self.metrics.update(train_metrics, "train")
            self.metrics.update(val_metrics, "val")

            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.save_checkpoint(f"best_model.pt")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "train_losses": self.metrics.train_losses,
            "val_losses": self.metrics.val_losses,
            "crisis_acc": self.metrics.crisis_acc,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class IncrementalTrainer:
    """For training on new data incrementally."""

    def __init__(self, base_model: MentalHealthModel, config: Dict[str, Any]):
        self.model = base_model
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def continue_training(
        self,
        new_train_loader: DataLoader,
        new_val_loader: DataLoader,
        additional_epochs: int = 3,
    ):
        """Continue training with new data."""

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.get("learning_rate", 1e-5)
        )

        for epoch in range(additional_epochs):
            self.model.train()

            for batch in new_train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                inputs = {"general": (input_ids, attention_mask)}
                outputs = self.model(inputs, return_safety=True, return_quality=False)

                loss = outputs["hidden_states"].mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"Additional epoch {epoch + 1}/{additional_epochs} completed")

        return self.model


def setup_training(
    data_dir: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
) -> Tuple[MentalHealthTrainer, Dict]:
    """Setup and initialize training."""

    if model_config is None:
        model_config = {
            "hidden_size": 768,
            "num_attention_heads": 8,
            "num_attention_layers": 2,
            "dropout": 0.1,
            "use_clinical_bert": True,
            "use_knowledge_graph": True,
        }

    if training_config is None:
        training_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_epochs": 10,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "crisis_weight": 2.0,
            "quality_weight": 0.5,
            "harmony_weight": 0.1,
            "patience": 3,
            "checkpoint_dir": "./checkpoints",
        }

    logger.info("Loading and processing data...")
    processor = MentalHealthDataProcessor(data_dir)
    dataset = processor.load_all_data()

    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./models/all-MiniLM-L6-v2")

    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset, processor, tokenizer, batch_size=training_config["batch_size"]
    )

    logger.info("Creating model...")
    model = create_model(**model_config)

    logger.info("Creating trainer...")
    trainer = MentalHealthTrainer(model, train_loader, val_loader, training_config)

    return trainer, {"dataset": dataset, "processor": processor, "tokenizer": tokenizer}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    training_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "crisis_weight": 2.0,
        "quality_weight": 0.5,
        "harmony_weight": 0.1,
        "patience": 3,
        "checkpoint_dir": "./checkpoints",
    }

    trainer, info = setup_training(args.data_dir, training_config=training_config)

    results = trainer.train()

    logger.info("Training completed!")
    logger.info(f"Final metrics: {trainer.metrics.get_summary()}")
