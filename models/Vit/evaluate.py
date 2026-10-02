import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/home/dhrubo_roy/Hybrid_v3")

sys.path.append(
    str(PROJECT_ROOT / "models" / "Vit")
)

from model_factory import create_model
import training_config as cfg


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

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

        print(f"Loading dataset:")
        print(f"  {csv_file}")

        # ----------------------------------------------------
        # IMAGE COLUMN
        # ----------------------------------------------------

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
                "Could not find image column. "
                f"Available columns: {list(self.data.columns)}"
            )

        # ----------------------------------------------------
        # LABEL COLUMN
        # ----------------------------------------------------

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
                "Could not find label column. "
                f"Available columns: {list(self.data.columns)}"
            )

        print(f"  Image column : {self.image_column}")
        print(f"  Label column : {self.label_column}")
        print(f"  Samples      : {len(self.data)}")

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # ----------------------------------------------------
        # IMAGE PATH
        # ----------------------------------------------------

        image_path = Path(
            str(row[self.image_column])
        )

        if not image_path.is_absolute():

            candidates = [

                PROJECT_ROOT / image_path,

                PROJECT_ROOT /
                "Dataset ISIC 2019" /
                image_path,
            ]

            for candidate in candidates:

                if candidate.exists():

                    image_path = candidate
                    break

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label_value = row[
            self.label_column
        ]

        label_name = str(label_value)

        if label_name in self.class_to_idx:

            label = self.class_to_idx[
                label_name
            ]

        else:

            try:

                label = int(label_value)

            except (ValueError, TypeError):

                raise ValueError(
                    f"Unknown label '{label_value}' "
                    f"at index {index}"
                )

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        if self.transform:

            image = self.transform(image)

        return image, label, index


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (
            cfg.IMAGE_SIZE,
            cfg.IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
    dataset
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_probabilities = []
    all_indices = []

    with torch.no_grad():

        for images, labels, indices in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_confidences.extend(
                confidences.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            all_indices.extend(
                indices.cpu().numpy().tolist()
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
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

    return {
        "labels": all_labels,
        "predictions": all_predictions,
        "confidences": all_confidences,
        "probabilities": all_probabilities,
        "indices": all_indices,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_model_checkpoint(
    model_name
):

    set_seed(cfg.SEED)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        cfg.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n")
    print("=" * 78)
    print("HYBRID_V3 ViT CHECKPOINT EVALUATION")
    print("=" * 78)

    print(
        f"Model          : {model_name}"
    )

    print(
        f"Device         : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    test_csv = (
        PROJECT_ROOT /
        "outputs" /
        "dataset_split" /
        "test.csv"
    )

    output_dir = (
        PROJECT_ROOT /
        "outputs" /
        "vit" /
        model_name
    )

    checkpoint_path = (
        output_dir /
        "best_model.pt"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not test_csv.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{test_csv}"
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print("\nPATHS")
    print("-" * 78)

    print(
        f"Test CSV     : {test_csv}"
    )

    print(
        f"Checkpoint   : {checkpoint_path}"
    )

    print(
        f"Output dir   : {output_dir}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print("\n")
    print("=" * 78)
    print("LOADING TEST DATASET")
    print("=" * 78)

    dataset = ISICDataset(
        test_csv,
        transform=test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\n")
    print("=" * 78)
    print("CREATING MODEL")
    print("=" * 78)

    model = create_model(
        model_name=model_name,
        num_classes=cfg.NUM_CLASSES,
        pretrained=False
    )

    model = model.to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    print("✓ Checkpoint loaded successfully")

    # ========================================================
    # PARAMETER INFORMATION
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters : "
        f"{total_params:,}"
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    print("\n")
    print("=" * 78)
    print("RUNNING INFERENCE")
    print("=" * 78)

    results = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        dataset=dataset
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 78)
    print("TEST RESULTS")
    print("=" * 78)

    print(
        f"Samples            : "
        f"{len(results['labels'])}"
    )

    correct_count = sum(
    p == y
    for p, y in zip(
        results["predictions"],
        results["labels"]
    )
    )

    wrong_count = sum(
        p != y
        for p, y in zip(
            results["predictions"],
            results["labels"]
        )
    )

    print(
        f"Correct            : {correct_count}"
    )

    print(
        f"Wrong              : {wrong_count}"
    )

    print(
        f"Accuracy           : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{results['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{results['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{results['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{results['macro_recall']:.4f}"
    )

    # ========================================================
    # CREATE PREDICTION TABLE
    # ========================================================

    print("\n")
    print("=" * 78)
    print("CREATING EXPANDED PREDICTION TABLE")
    print("=" * 78)

    rows = []

    class_names = cfg.CLASSES

    for i in range(
        len(results["labels"])
    ):

        dataset_index = (
            results["indices"][i]
        )

        original_row = (
            dataset.data.iloc[
                dataset_index
            ]
        )

        # ----------------------------------------------------
        # IMAGE ID
        # ----------------------------------------------------

        if "image" in dataset.data.columns:

            image_id = str(
                original_row["image"]
            )

        elif "image_id" in dataset.data.columns:

            image_id = str(
                original_row["image_id"]
            )

        elif "path" in dataset.data.columns:

            image_id = Path(
                str(original_row["path"])
            ).stem

        else:

            image_id = str(
                dataset_index
            )

        true_index = (
            results["labels"][i]
        )

        predicted_index = (
            results["predictions"][i]
        )

        row = {

            "image_id":
                image_id,

            "true_class":
                class_names[true_index],

            "true_index":
                true_index,

            "predicted_class":
                class_names[predicted_index],

            "predicted_index":
                predicted_index,

            "confidence":
                results["confidences"][i],

            "correct":
                int(
                    true_index ==
                    predicted_index
                ),
        }

        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        probabilities = (
            results["probabilities"][i]
        )

        for class_index, class_name in enumerate(
            class_names
        ):

            row[
                f"prob_{class_name}"
            ] = probabilities[
                class_index
            ]

        rows.append(row)

    predictions_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_path = (
        output_dir /
        "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"\nSaved predictions:"
    )

    print(
        f"  {predictions_path}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        results["labels"],
        results["predictions"],
        labels=list(
            range(cfg.NUM_CLASSES)
        )
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_path = (
        output_dir /
        "test_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )

    print(
        f"Saved confusion matrix:"
    )

    print(
        f"  {cm_path}"
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "model":
            model_name,

        "checkpoint":
            str(checkpoint_path),

        "samples":
            len(results["labels"]),

        "correct":
            int(
                sum(
                    p == y
                    for p, y in zip(
                        results["predictions"],
                        results["labels"]
                    )
                )
            ),

        "wrong":
            int(
                sum(
                    p != y
                    for p, y in zip(
                        results["predictions"],
                        results["labels"]
                    )
                )
            ),

        "test_accuracy":
            results["accuracy"],

        "test_balanced_accuracy":
            results["balanced_accuracy"],

        "test_macro_f1":
            results["macro_f1"],

        "test_weighted_f1":
            results["weighted_f1"],

        "test_macro_precision":
            results["macro_precision"],

        "test_macro_recall":
            results["macro_recall"],
    }

    metrics_path = (
        output_dir /
        "test_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Saved metrics:"
    )

    print(
        f"  {metrics_path}"
    )

    # ========================================================
    # DISPLAY SAMPLE PREDICTIONS
    # ========================================================

    print("\n")
    print("=" * 78)
    print("SAMPLE PREDICTIONS")
    print("=" * 78)

    display_columns = [

        "image_id",

        "true_class",

        "predicted_class",

        "confidence",

        "correct",
    ]

    print(
        predictions_df[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 78)
    print("EVALUATION COMPLETE")
    print("=" * 78)

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Accuracy   : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Macro F1   : "
        f"{results['macro_f1']:.4f}"
    )

    print("\nOutputs:")
    print(
        f"  {predictions_path}"
    )
    print(
        f"  {cm_path}"
    )
    print(
        f"  {metrics_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an existing Hybrid_v3 "
            "ViT checkpoint"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "deit_tiny",
            "swin_tiny",
        ],
        help="ViT model to evaluate"
    )

    args = parser.parse_args()

    evaluate_model_checkpoint(
        args.model
    )

