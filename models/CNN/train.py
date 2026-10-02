"""
HYBRID_V3 - CNN TRAINING PIPELINE
=================================

Trains one CNN architecture at a time using the official
leakage-safe ISIC 2019 splits.

Supported models:
    - efficientnet_v2_s
    - efficientnet_b3
    - mobilenet_v3_large
    - convnext_tiny
    - densenet121
    - resnet50

Usage:
    python models/CNN/train.py --model mobilenet_v3_large --epochs 30

Example:
    python models/CNN/train.py \
        --model mobilenet_v3_large \
        --epochs 30 \
        --lr 1e-4 \
        --batch-size 16

Important:
    The dataloader returns dictionaries:

        {
            "image": Tensor,
            "label": Tensor,
            "image_id": str,
            "path": str
        }

    Therefore training accesses:

        images = batch["image"]
        labels = batch["label"]
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import sys
import argparse
import json
import random
import time

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make Hybrid_v3 importable when running:
# python models/CNN/train.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from scripts.build_dataloader import (
    create_dataloaders,
    CLASS_NAMES,
    NUM_CLASSES,
)

from models.CNN.model_factory import create_model


# =============================================================================
# PATHS
# =============================================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn"


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(labels, predictions):
    """
    Calculate classification metrics.

    Returns:
        dict
    """

    accuracy = accuracy_score(
        labels,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(dataset):
    """
    Calculate inverse-frequency class weights.

    Uses the training dataset only.
    """

    labels = dataset.labels

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    total = len(labels)

    weights = np.zeros(
        NUM_CLASSES,
        dtype=np.float32
    )

    for i, count in enumerate(counts):

        if count > 0:
            weights[i] = (
                total /
                (NUM_CLASSES * count)
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):
    """
    Train model for one epoch.
    """

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):

        # =============================================================
        # DataLoader returns a dictionary
        # =============================================================

        images = batch["image"]
        labels = batch["label"]

        # =============================================================
        # GPU transfer
        # =============================================================

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # =============================================================
        # Clear gradients
        # =============================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        # =============================================================
        # Forward + loss
        # =============================================================

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        # =============================================================
        # Backpropagation
        # =============================================================

        scaler.scale(loss).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        # =============================================================
        # Predictions
        # =============================================================

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

        # =============================================================
        # Store predictions
        # =============================================================

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        # =============================================================
        # Progress
        # =============================================================

        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
            or batch_idx + 1 == len(loader)
        ):

            current_loss = (
                running_loss /
                total_samples
            )

            print(
                f"\r  Batch "
                f"{batch_idx + 1:4d}/"
                f"{len(loader):4d} "
                f"| Loss: {current_loss:.4f}",
                end=""
            )

    print()

    epoch_loss = (
        running_loss /
        total_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = epoch_loss

    metrics["time"] = (
        time.time() -
        start_time
    )

    return metrics


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    """
    Validation pass.
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    start_time = time.time()

    for batch in loader:

        images = batch["image"]
        labels = batch["label"]

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss /
        total_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = epoch_loss

    metrics["time"] = (
        time.time() -
        start_time
    )

    return (
        metrics,
        all_labels,
        all_predictions,
    )


# =============================================================================
# TEST
# =============================================================================

@torch.no_grad()
def evaluate_test(
    model,
    loader,
    criterion,
    device,
):
    """
    Final test evaluation.

    Keeps image IDs so that later explainability/ERI
    analysis can map predictions back to images.
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_ids = []

    for batch in loader:

        images = batch["image"]
        labels = batch["label"]
        image_ids = batch["image_id"]

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        probabilities = torch.softmax(
            outputs.float(),
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_probabilities.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

        all_image_ids.extend(
            list(image_ids)
        )

    loss = (
        running_loss /
        total_samples
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = loss

    return (
        metrics,
        all_labels,
        all_predictions,
        all_probabilities,
        all_image_ids,
    )


# =============================================================================
# SAVE CONFUSION MATRIX
# =============================================================================

def save_confusion_matrix(
    labels,
    predictions,
    output_path,
):
    """
    Save confusion matrix as CSV.
    """

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    df.index.name = "true_class"

    df.to_csv(
        output_path
    )


# =============================================================================
# SAVE TEST PREDICTIONS
# =============================================================================

def save_test_predictions(
    image_ids,
    labels,
    predictions,
    probabilities,
    output_path,
):
    """
    Save per-image predictions.

    This file will be useful later for explainability
    and ERI analysis.
    """

    records = []

    for i, image_id in enumerate(
        image_ids
    ):

        row = {
            "image_id": image_id,
            "true_class": CLASS_NAMES[
                labels[i]
            ],
            "true_index": labels[i],
            "predicted_class": CLASS_NAMES[
                predictions[i]
            ],
            "predicted_index": predictions[i],
            "confidence": float(
                probabilities[i][
                    predictions[i]
                ]
            ),
            "correct": int(
                labels[i] ==
                predictions[i]
            ),
        }

        # Add probability for every class

        for class_idx, class_name in enumerate(
            CLASS_NAMES
        ):

            row[
                f"prob_{class_name}"
            ] = float(
                probabilities[i][
                    class_idx
                ]
            )

        records.append(row)

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        output_path,
        index=False
    )


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    path,
):
    """
    Save complete training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        "scaler_state_dict":
            scaler.state_dict(),

        "metrics":
            metrics,

        "class_names":
            CLASS_NAMES,

        "num_classes":
            NUM_CLASSES,
    }

    torch.save(
        checkpoint,
        path
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # ARGUMENTS
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="Train CNN on ISIC 2019"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "efficientnet_v2_s",
            "efficientnet_b3",
            "mobilenet_v3_large",
            "convnext_tiny",
            "densenet121",
            "resnet50",
        ],
        help="CNN architecture"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=7
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--weighted-sampler",
        action="store_true",
        help="Use weighted random sampler"
    )

    args = parser.parse_args()

    # =========================================================================
    # SETUP
    # =========================================================================

    seed_everything(
        args.seed
    )

    print(
        "=" * 78
    )

    print(
        "HYBRID_V3 CNN TRAINING"
    )

    print(
        "=" * 78
    )

    print(
        f"Model          : {args.model}"
    )

    print(
        f"Epochs         : {args.epochs}"
    )

    print(
        f"Learning rate  : {args.lr}"
    )

    print(
        f"Weight decay   : {args.weight_decay}"
    )

    print(
        f"Batch size     : {args.batch_size}"
    )

    print(
        f"Early stopping : {args.early_stopping}"
    )

    print(
        f"Seed           : {args.seed}"
    )

    print(
        f"CUDA available : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================

    output_dir = (
        OUTPUT_ROOT /
        args.model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Output directory:"
    )
    print(
        output_dir
    )

    # =========================================================================
    # SAVE CONFIGURATION
    # =========================================================================

    config = {
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "early_stopping": args.early_stopping,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "device": str(DEVICE),
    }

    with open(
        output_dir / "config.json",
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "LOADING DATALOADERS"
    )

    print(
        "=" * 78
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        batch_size=args.batch_size,
        use_weighted_sampler=args.weighted_sampler,
    )

    print(
        f"Train samples : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples  : "
        f"{len(test_loader.dataset)}"
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "CREATING MODEL"
    )

    print(
        "=" * 78
    )

    model = create_model(
        args.model,
        num_classes=NUM_CLASSES,
    )

    model = model.to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # =========================================================================
    # CLASS WEIGHTS
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "CLASS WEIGHTS"
    )

    print(
        "=" * 78
    )

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:>5} "
            f"{class_weights[i].item():10.6f}"
        )

    # =========================================================================
    # LOSS
    # =========================================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # =========================================================================
    # AMP
    # =========================================================================

    scaler = GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        )
    )

    # =========================================================================
    # TRAINING STATE
    # =========================================================================

    best_score = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # =========================================================================
    # TRAINING
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 78
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print()
        print(
            "-" * 78
        )

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            "-" * 78
        )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=DEVICE,
        )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        val_metrics, _, _ = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        # ---------------------------------------------------------------------
        # Scheduler
        # ---------------------------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        # ---------------------------------------------------------------------
        # Print metrics
        # ---------------------------------------------------------------------

        print()
        print(
            "TRAIN"
        )

        print(
            f"  Loss             : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Accuracy         : "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"  Balanced Acc     : "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Macro F1         : "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print()
        print(
            "VALIDATION"
        )

        print(
            f"  Loss             : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Accuracy         : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Balanced Acc     : "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Macro F1         : "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Macro Precision  : "
            f"{val_metrics['macro_precision']:.4f}"
        )

        print(
            f"  Macro Recall     : "
            f"{val_metrics['macro_recall']:.4f}"
        )

        print(
            f"  Learning Rate    : "
            f"{current_lr:.2e}"
        )

        # ---------------------------------------------------------------------
        # History
        # ---------------------------------------------------------------------

        history_row = {
            "epoch": epoch,

            "learning_rate": current_lr,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

            "train_macro_f1":
                train_metrics["macro_f1"],

            "train_weighted_f1":
                train_metrics["weighted_f1"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "val_macro_f1":
                val_metrics["macro_f1"],

            "val_weighted_f1":
                val_metrics["weighted_f1"],

            "val_macro_precision":
                val_metrics[
                    "macro_precision"
                ],

            "val_macro_recall":
                val_metrics[
                    "macro_recall"
                ],
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_dir /
            "training_history.csv",
            index=False
        )

        # ---------------------------------------------------------------------
        # Best model selection
        #
        # Macro-F1 is the primary criterion because ISIC 2019 is
        # heavily imbalanced.
        # ---------------------------------------------------------------------

        score = val_metrics[
            "macro_f1"
        ]

        if score > best_score:

            best_score = score

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint_path = (
                output_dir /
                "best_model.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=val_metrics,
                path=checkpoint_path,
            )

            print()
            print(
                "✓ NEW BEST MODEL"
            )

            print(
                f"  Validation Macro-F1: "
                f"{best_score:.4f}"
            )

            print(
                f"  Saved: "
                f"{checkpoint_path}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{args.early_stopping})"
            )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= args.early_stopping
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            print(
                f"No improvement for "
                f"{args.early_stopping} epochs."
            )

            break

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 78
    )

    best_checkpoint_path = (
        output_dir /
        "best_model.pt"
    )

    if best_checkpoint_path.exists():

        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            f"Best epoch : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Best Macro-F1 : "
            f"{checkpoint['metrics']['macro_f1']:.4f}"
        )

    else:

        print(
            "WARNING: Best checkpoint not found."
        )

    # =========================================================================
    # FINAL TEST
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 78
    )

    (
        test_metrics,
        test_labels,
        test_predictions,
        test_probabilities,
        test_image_ids,
    ) = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=DEVICE,
    )

    # =========================================================================
    # TEST RESULTS
    # =========================================================================

    print()
    print(
        "TEST RESULTS"
    )

    print(
        "-" * 78
    )

    print(
        f"Loss             : "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Accuracy         : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1         : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1      : "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    # =========================================================================
    # SAVE TEST METRICS
    # =========================================================================

    with open(
        output_dir /
        "test_metrics.json",
        "w"
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=4
        )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    save_confusion_matrix(
        labels=test_labels,
        predictions=test_predictions,
        output_path=(
            output_dir /
            "test_confusion_matrix.csv"
        ),
    )

    # =========================================================================
    # TEST PREDICTIONS
    # =========================================================================

    save_test_predictions(
        image_ids=test_image_ids,
        labels=test_labels,
        predictions=test_predictions,
        probabilities=test_probabilities,
        output_path=(
            output_dir /
            "test_predictions.csv"
        ),
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print(
        "=" * 78
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Model              : "
        f"{args.model}"
    )

    print(
        f"Best epoch         : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1 : "
        f"{best_score:.4f}"
    )

    print()
    print(
        "FINAL TEST"
    )

    print(
        f"  Accuracy          : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"  Balanced Accuracy : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"  Macro F1          : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "OUTPUTS"
    )

    print(
        f"  {output_dir}"
    )

    print(
        f"  ├── best_model.pt"
    )

    print(
        f"  ├── config.json"
    )

    print(
        f"  ├── training_history.csv"
    )

    print(
        f"  ├── test_metrics.json"
    )

    print(
        f"  ├── test_predictions.csv"
    )

    print(
        f"  └── test_confusion_matrix.csv"
    )

    print()
    print(
        "✓ CNN training pipeline completed."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()