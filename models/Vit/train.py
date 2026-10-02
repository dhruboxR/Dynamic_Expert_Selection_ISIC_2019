import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import sys

PROJECT_ROOT = Path("/home/dhrubo_roy/Hybrid_v3")

sys.path.append(str(PROJECT_ROOT / "models" / "ViT"))

from model_factory import create_model
import training_config as cfg


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class ISICDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.classes = cfg.CLASSES
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        print(f"Loading: {csv_file}")

        # --------------------------------------------
        # Detect image path column
        # --------------------------------------------

        possible_image_columns = [
            "image_path",
            "path",
            "filepath",
            "file_path",
            "image",
            "image_id",
            "filename",
        ]

        self.image_column = None

        for column in possible_image_columns:
            if column in self.data.columns:
                self.image_column = column
                break

        if self.image_column is None:
            raise ValueError(
                f"Could not find image column in {csv_file}. "
                f"Available columns: {list(self.data.columns)}"
            )

        # --------------------------------------------
        # Detect label column
        # --------------------------------------------

        possible_label_columns = [
            "label",
            "class",
            "dx",
            "diagnosis",
            "target",
        ]

        self.label_column = None

        for column in possible_label_columns:
            if column in self.data.columns:
                self.label_column = column
                break

        if self.label_column is None:
            raise ValueError(
                f"Could not find label column in {csv_file}. "
                f"Available columns: {list(self.data.columns)}"
            )

        print(f"  Image column : {self.image_column}")
        print(f"  Label column : {self.label_column}")
        print(f"  Samples      : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(str(row[self.image_column]))

        # --------------------------------------------
        # If CSV contains relative paths
        # --------------------------------------------

        if not image_path.is_absolute():

            candidates = [
                PROJECT_ROOT / image_path,
                PROJECT_ROOT / "Dataset ISIC 2019" / image_path,
            ]

            for candidate in candidates:
                if candidate.exists():
                    image_path = candidate
                    break

        image = Image.open(image_path).convert("RGB")

        label_name = str(row[self.label_column])

        if label_name not in self.class_to_idx:

            # Handle numeric labels if present
            try:
                label = int(label_name)
            except ValueError:
                raise ValueError(
                    f"Unknown label '{label_name}' at index {index}"
                )

        else:
            label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([

    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    labels = []

    for label_name in dataset.data[dataset.label_column]:

        if label_name in dataset.class_to_idx:
            labels.append(
                dataset.class_to_idx[label_name]
            )
        else:
            labels.append(int(label_name))

    counts = np.bincount(
        labels,
        minlength=cfg.NUM_CLASSES
    )

    total = counts.sum()

    weights = total / (
        cfg.NUM_CLASSES * np.maximum(counts, 1)
    )

    print("\nCLASS WEIGHTS")
    print("-" * 60)

    for class_name, count, weight in zip(
        cfg.CLASSES,
        counts,
        weights
    ):
        print(
            f"{class_name:>5} "
            f"{count:6d} "
            f"{weight:10.6f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": np.array(all_probabilities),
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(model_name):

    set_seed(cfg.SEED)

    device = torch.device(
        cfg.DEVICE if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("HYBRID_V3 ViT TRAINING")
    print("=" * 78)

    print(f"Model          : {model_name}")
    print(f"Epochs         : {cfg.EPOCHS}")
    print(f"Learning rate  : {cfg.LEARNING_RATE}")
    print(f"Weight decay   : {cfg.WEIGHT_DECAY}")
    print(f"Batch size     : {cfg.BATCH_SIZE}")
    print(f"Early stopping : {cfg.EARLY_STOPPING}")
    print(f"Seed           : {cfg.SEED}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    output_dir = (
        cfg.OUTPUT_ROOT /
        model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nOutput directory:")
    print(output_dir)

    # ========================================================
    # DATASETS
    # ========================================================

    print("\n" + "=" * 78)
    print("LOADING DATALOADERS")
    print("=" * 78)

    train_dataset = ISICDataset(
        cfg.TRAIN_CSV,
        transform=train_transform
    )

    val_dataset = ISICDataset(
        cfg.VAL_CSV,
        transform=val_transform
    )

    test_dataset = ISICDataset(
        cfg.TEST_CSV,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = calculate_class_weights(
        train_dataset
    ).to(device)

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 78)
    print("CREATING MODEL")
    print("=" * 78)

    model = create_model(
        model_name=model_name,
        num_classes=cfg.NUM_CLASSES,
        pretrained=True
    )

    model = model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 78)
    print("STARTING TRAINING")
    print("=" * 78)

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, cfg.EPOCHS + 1):

        print("\n" + "-" * 78)
        print(f"EPOCH {epoch}/{cfg.EPOCHS}")
        print("-" * 78)

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):

                print(
                    f"\r  Batch "
                    f"{batch_idx}/{len(train_loader)} "
                    f"| Loss: "
                    f"{running_loss / total:.4f}",
                    end=""
                )

        train_loss = running_loss / total
        train_accuracy = correct / total

        train_metrics = evaluate(
            model,
            train_loader,
            criterion,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print("\n")

        print("TRAIN")
        print(
            f"  Loss             : "
            f"{train_loss:.4f}"
        )
        print(
            f"  Accuracy         : "
            f"{train_accuracy:.4f}"
        )
        print(
            f"  Balanced Acc     : "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"  Macro F1         : "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print("\nVALIDATION")

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

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_balanced_accuracy":
                train_metrics["balanced_accuracy"],
            "train_macro_f1":
                train_metrics["macro_f1"],
            "val_loss":
                val_metrics["loss"],
            "val_accuracy":
                val_metrics["accuracy"],
            "val_balanced_accuracy":
                val_metrics["balanced_accuracy"],
            "val_macro_f1":
                val_metrics["macro_f1"],
            "val_macro_precision":
                val_metrics["macro_precision"],
            "val_macro_recall":
                val_metrics["macro_recall"],
            "learning_rate": current_lr
        })

        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt"
            )

            print(
                "\n✓ NEW BEST MODEL"
            )

            print(
                f"  Validation Macro-F1: "
                f"{best_macro_f1:.4f}"
            )

            print(
                f"  Saved: "
                f"{output_dir / 'best_model.pt'}"
            )

        else:

            patience_counter += 1

            print(
                f"\nNo improvement "
                f"({patience_counter}/"
                f"{cfg.EARLY_STOPPING})"
            )

        if patience_counter >= cfg.EARLY_STOPPING:

            print("\nEARLY STOPPING")
            print(
                f"No improvement for "
                f"{cfg.EARLY_STOPPING} epochs."
            )

            break

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir / "training_history.csv",
        index=False
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n" + "=" * 78)
    print("LOADING BEST MODEL")
    print("=" * 78)

    print(
        f"Best epoch : {best_epoch}"
    )

    print(
        f"Best Macro-F1 : "
        f"{best_macro_f1:.4f}"
    )

    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt",
            map_location=device
        )
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 78)
    print("FINAL TEST EVALUATION")
    print("=" * 78)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\nTEST RESULTS")
    print("-" * 78)

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

    # ========================================================
    # SAVE TEST METRICS
    # ========================================================

    metrics_to_save = {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_validation_macro_f1":
            best_macro_f1,
        "test_loss":
            test_metrics["loss"],
        "test_accuracy":
            test_metrics["accuracy"],
        "test_balanced_accuracy":
            test_metrics["balanced_accuracy"],
        "test_macro_f1":
            test_metrics["macro_f1"],
        "test_weighted_f1":
            test_metrics["weighted_f1"],
        "test_macro_precision":
            test_metrics["macro_precision"],
        "test_macro_recall":
            test_metrics["macro_recall"],
    }

    with open(
        output_dir / "test_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics_to_save,
            f,
            indent=4
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    print("\nSaving detailed test predictions...")

    # --------------------------------------------------------
    # Get test image IDs
    # --------------------------------------------------------

    test_image_ids = test_dataset.data[
        test_dataset.image_column
    ].astype(str).tolist()

    # --------------------------------------------------------
    # Build prediction table
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({

        "image_id":
            test_image_ids,

        "true_index":
            test_metrics["labels"],

        "predicted_index":
            test_metrics["predictions"],
    })

    # --------------------------------------------------------
    # Convert indices to class names
    # --------------------------------------------------------

    predictions_df["true_class"] = (
        predictions_df["true_index"]
        .apply(lambda x: cfg.CLASSES[int(x)])
    )

    predictions_df["predicted_class"] = (
        predictions_df["predicted_index"]
        .apply(lambda x: cfg.CLASSES[int(x)])
    )

    # --------------------------------------------------------
    # Correct / wrong
    # --------------------------------------------------------

    predictions_df["correct"] = (
        predictions_df["true_index"]
        ==
        predictions_df["predicted_index"]
    )

    # --------------------------------------------------------
    # Probability columns
    # --------------------------------------------------------

    probabilities = test_metrics["probabilities"]

    for class_index, class_name in enumerate(cfg.CLASSES):

        predictions_df[
            f"prob_{class_name}"
        ] = probabilities[:, class_index]

    # --------------------------------------------------------
    # Confidence = probability of predicted class
    # --------------------------------------------------------

    predicted_indices = np.asarray(
        test_metrics["predictions"]
    )

    predicted_confidences = probabilities[
        np.arange(len(predicted_indices)),
        predicted_indices
    ]

    predictions_df["confidence"] = predicted_confidences

    # --------------------------------------------------------
    # Arrange columns
    # --------------------------------------------------------

    prediction_columns = [

        "image_id",

        "true_class",
        "true_index",

        "predicted_class",
        "predicted_index",

        "confidence",
        "correct",
    ]

    prediction_columns += [
        f"prob_{class_name}"
        for class_name in cfg.CLASSES
    ]

    predictions_df = predictions_df[
        prediction_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prediction_path = (
        output_dir /
        "test_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Saved: {prediction_path}"
    )

    print(
        f"Rows : {len(predictions_df)}"
    )

    print(
        f"Columns : {predictions_df.columns.tolist()}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=list(range(cfg.NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=cfg.CLASSES,
        columns=cfg.CLASSES
    )

    cm_df.to_csv(
        output_dir / "test_confusion_matrix.csv"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {
        "model": model_name,
        "epochs": cfg.EPOCHS,
        "learning_rate":
            cfg.LEARNING_RATE,
        "weight_decay":
            cfg.WEIGHT_DECAY,
        "batch_size":
            cfg.BATCH_SIZE,
        "early_stopping":
            cfg.EARLY_STOPPING,
        "seed":
            cfg.SEED,
        "image_size":
            cfg.IMAGE_SIZE,
        "num_classes":
            cfg.NUM_CLASSES,
        "classes":
            cfg.CLASSES,
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

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)

    print(
        f"Model              : "
        f"{model_name}"
    )

    print(
        f"Best epoch         : "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1 : "
        f"{best_macro_f1:.4f}"
    )

    print("\nFINAL TEST")

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

    print("\nOUTPUTS")

    print(
        f"  {output_dir}"
    )

    print("  ├── best_model.pt")
    print("  ├── config.json")
    print("  ├── training_history.csv")
    print("  ├── test_metrics.json")
    print("  ├── test_predictions.csv")
    print("  └── test_confusion_matrix.csv")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "deit_tiny",
            "swin_tiny",
        ],
        help="Model architecture to train"
    )

    args = parser.parse_args()

    train_model(args.model)