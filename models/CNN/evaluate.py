#!/usr/bin/env python3

"""
HYBRID_V3 CNN ARCHITECTURE EVALUATION
======================================

Evaluates already-trained CNN checkpoints through their saved
test_predictions.csv files.

NO TRAINING IS PERFORMED.

Supported models:
    - resnet50
    - efficientnet_v2_s
    - efficientnet_b3
    - mobilenet_v3_large
    - all

Expected prediction format:

    image_id
    true_class
    true_index
    predicted_class
    predicted_index
    confidence
    correct
    prob_AK
    prob_BCC
    prob_BKL
    prob_DF
    prob_MEL
    prob_NV
    prob_SCC
    prob_VASC

Outputs per model:

    test_predictions.csv
    test_confusion_matrix.csv
    test_metrics.json
    per_class_metrics.csv
    reliability_analysis.csv

Outputs globally:

    cnn_architecture_comparison.csv
"""


# ============================================================
# IMPORTS
# ============================================================

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "outputs" /
    "cnn"
)

TEST_CSV = (
    PROJECT_ROOT /
    "outputs" /
    "dataset_split" /
    "test.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {

    "resnet50": {
        "display_name": "ResNet50",
        "prediction_file":
            OUTPUT_ROOT /
            "resnet50" /
            "test_predictions.csv",
    },

    "efficientnet_v2_s": {
        "display_name": "EfficientNetV2-S",
        "prediction_file":
            OUTPUT_ROOT /
            "efficientnet_v2_s" /
            "test_predictions.csv",
    },

    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "prediction_file":
            OUTPUT_ROOT /
            "efficientnet_b3" /
            "test_predictions.csv",
    },

    "mobilenet_v3_large": {
        "display_name": "MobileNetV3-Large",
        "prediction_file":
            OUTPUT_ROOT /
            "mobilenet_v3_large" /
            "test_predictions.csv",
    },
}


# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASSES = [
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
]

NUM_CLASSES = len(CLASSES)

PROBABILITY_COLUMNS = [
    f"prob_{class_name}"
    for class_name in CLASSES
]


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title):

    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title):

    print("\n")
    print("-" * 78)
    print(title)
    print("-" * 78)


# ============================================================
# VALIDATE PREDICTION DATA
# ============================================================

def validate_prediction_dataframe(df, model_name):

    required_columns = [
        "image_id",
        "true_class",
        "true_index",
        "predicted_class",
        "predicted_index",
        "confidence",
        "correct",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{model_name}: Missing required columns: "
            f"{missing}"
        )

    missing_probabilities = [
        column
        for column in PROBABILITY_COLUMNS
        if column not in df.columns
    ]

    if missing_probabilities:

        raise ValueError(
            f"{model_name}: Missing probability columns: "
            f"{missing_probabilities}"
        )

    if len(df) == 0:

        raise ValueError(
            f"{model_name}: Prediction file is empty."
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "true_index",
        "predicted_index",
        "confidence",
    ] + PROBABILITY_COLUMNS

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_counts = df[
        numeric_columns
    ].isna().sum()

    problematic = (
        missing_counts[
            missing_counts > 0
        ]
    )

    if len(problematic) > 0:

        raise ValueError(
            f"{model_name}: Missing/non-numeric values found:\n"
            f"{problematic}"
        )

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    probability_matrix = (
        df[PROBABILITY_COLUMNS]
        .to_numpy(dtype=np.float64)
    )

    probability_sums = (
        probability_matrix.sum(axis=1)
    )

    max_sum_error = np.max(
        np.abs(probability_sums - 1.0)
    )

    print(
        f"Max probability sum error : "
        f"{max_sum_error:.8f}"
    )

    if max_sum_error > 1e-3:

        print(
            "WARNING: Probability sums deviate "
            "from 1.0 by more than 1e-3."
        )

    # --------------------------------------------------------
    # Index validation
    # --------------------------------------------------------

    invalid_true = (
        (df["true_index"] < 0)
        |
        (df["true_index"] >= NUM_CLASSES)
    )

    invalid_pred = (
        (df["predicted_index"] < 0)
        |
        (df["predicted_index"] >= NUM_CLASSES)
    )

    if invalid_true.any():

        raise ValueError(
            f"{model_name}: Invalid true class indices."
        )

    if invalid_pred.any():

        raise ValueError(
            f"{model_name}: Invalid predicted class indices."
        )

    return df


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(model_key):

    config = MODELS[model_key]

    model_name = config["display_name"]

    prediction_file = (
        config["prediction_file"]
    )

    print_section(
        f"LOADING {model_name}"
    )

    print(
        f"Path    : {prediction_file}"
    )

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"\nPrediction file not found:\n"
            f"{prediction_file}\n\n"
            f"Make sure the model has already been "
            f"evaluated/trained."
        )

    df = pd.read_csv(
        prediction_file
    )

    print(
        f"Samples : {len(df)}"
    )

    print(
        f"Columns : {df.columns.tolist()}"
    )

    print(
        "Format  : CNN expanded prediction format ✓"
    )

    df = validate_prediction_dataframe(
        df,
        model_name
    )

    return df


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_basic_metrics(df):

    y_true = (
        df["true_index"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["predicted_index"]
        .astype(int)
        .to_numpy()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    correct = int(
        (y_true == y_pred).sum()
    )

    wrong = int(
        (y_true != y_pred).sum()
    )

    return {

        "samples": int(len(df)),

        "correct": correct,

        "wrong": wrong,

        "accuracy": float(
            accuracy
        ),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),
    }


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probabilities *
        np.log(probabilities),
        axis=1
    )

    # Normalize by maximum possible entropy
    normalized_entropy = (
        entropy /
        math.log(NUM_CLASSES)
    )

    return normalized_entropy


# ============================================================
# TOP-2 MARGIN
# ============================================================

def calculate_top2_margin(probabilities):

    sorted_probabilities = np.sort(
        probabilities,
        axis=1
    )

    top1 = sorted_probabilities[:, -1]

    top2 = sorted_probabilities[:, -2]

    return top1 - top2


# ============================================================
# RELIABILITY ANALYSIS
# ============================================================

def create_reliability_analysis(
    df,
    model_name,
    output_dir
):

    print_section(
        "RELIABILITY ANALYSIS"
    )

    probabilities = (
        df[PROBABILITY_COLUMNS]
        .to_numpy(dtype=np.float64)
    )

    confidence = (
        probabilities.max(axis=1)
    )

    entropy = calculate_entropy(
        probabilities
    )

    top2_margin = calculate_top2_margin(
        probabilities
    )

    y_true = (
        df["true_index"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["predicted_index"]
        .astype(int)
        .to_numpy()
    )

    correct = (
        y_true == y_pred
    )

    reliability_df = pd.DataFrame({

        "image_id":
            df["image_id"].values,

        "true_class":
            df["true_class"].values,

        "predicted_class":
            df["predicted_class"].values,

        "true_index":
            y_true,

        "predicted_index":
            y_pred,

        "confidence":
            confidence,

        "entropy":
            entropy,

        "top2_margin":
            top2_margin,

        "correct":
            correct.astype(int),
    })

    output_file = (
        output_dir /
        "reliability_analysis.csv"
    )

    reliability_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    correct_mask = correct

    wrong_mask = ~correct

    mean_conf_correct = (
        confidence[correct_mask].mean()
        if correct_mask.any()
        else np.nan
    )

    mean_conf_wrong = (
        confidence[wrong_mask].mean()
        if wrong_mask.any()
        else np.nan
    )

    mean_entropy_correct = (
        entropy[correct_mask].mean()
        if correct_mask.any()
        else np.nan
    )

    mean_entropy_wrong = (
        entropy[wrong_mask].mean()
        if wrong_mask.any()
        else np.nan
    )

    mean_margin_correct = (
        top2_margin[correct_mask].mean()
        if correct_mask.any()
        else np.nan
    )

    mean_margin_wrong = (
        top2_margin[wrong_mask].mean()
        if wrong_mask.any()
        else np.nan
    )

    print(
        f"Mean confidence (correct)         : "
        f"{mean_conf_correct:.4f}"
    )

    print(
        f"Mean confidence (wrong)           : "
        f"{mean_conf_wrong:.4f}"
    )

    print(
        f"Mean entropy (correct)             : "
        f"{mean_entropy_correct:.4f}"
    )

    print(
        f"Mean entropy (wrong)               : "
        f"{mean_entropy_wrong:.4f}"
    )

    print(
        f"Mean top-2 margin (correct)        : "
        f"{mean_margin_correct:.4f}"
    )

    print(
        f"Mean top-2 margin (wrong)           : "
        f"{mean_margin_wrong:.4f}"
    )

    print(
        f"\nSaved reliability analysis:"
    )

    print(
        f"  {output_file}"
    )

    return {

        "mean_confidence_correct":
            float(mean_conf_correct),

        "mean_confidence_wrong":
            float(mean_conf_wrong),

        "mean_entropy_correct":
            float(mean_entropy_correct),

        "mean_entropy_wrong":
            float(mean_entropy_wrong),

        "mean_margin_correct":
            float(mean_margin_correct),

        "mean_margin_wrong":
            float(mean_margin_wrong),
    }


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

def create_per_class_analysis(
    df,
    model_name,
    output_dir
):

    print_section(
        "PER-CLASS EVALUATION"
    )

    y_true = (
        df["true_index"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["predicted_index"]
        .astype(int)
        .to_numpy()
    )

    labels = list(
        range(NUM_CLASSES)
    )

    # --------------------------------------------------------
    # Calculate each metric separately
    #
    # IMPORTANT:
    # We explicitly create the "f1" field.
    # This fixes the previous KeyError.
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Support
    # --------------------------------------------------------

    support = np.bincount(
        y_true,
        minlength=NUM_CLASSES
    )

    rows = []

    for index, class_name in enumerate(CLASSES):

        class_mask = (
            y_true == index
        )

        class_correct = (
            (
                y_true[class_mask]
                ==
                y_pred[class_mask]
            ).sum()
        )

        class_total = (
            class_mask.sum()
        )

        class_accuracy = (
            class_correct /
            class_total
            if class_total > 0
            else 0.0
        )

        rows.append({

            "class_index":
                index,

            "class":
                class_name,

            "support":
                int(support[index]),

            "correct":
                int(class_correct),

            "wrong":
                int(
                    class_total -
                    class_correct
                ),

            "class_accuracy":
                float(class_accuracy),

            "precision":
                float(precision[index]),

            "recall":
                float(recall[index]),

            "f1":
                float(f1[index]),
        })

    per_class_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    display_columns = [
        "class",
        "support",
        "correct",
        "wrong",
        "class_accuracy",
        "precision",
        "recall",
        "f1",
    ]

    print(
        per_class_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        output_dir /
        "per_class_metrics.csv"
    )

    per_class_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved per-class metrics:"
    )

    print(
        f"  {output_file}"
    )

    return per_class_df


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    df,
    model_name,
    output_dir
):

    print_section(
        "CONFUSION MATRIX"
    )

    y_true = (
        df["true_index"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df["predicted_index"]
        .astype(int)
        .to_numpy()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    )

    cm_df.index.name = "true_class"

    output_file = (
        output_dir /
        "test_confusion_matrix.csv"
    )

    cm_df.to_csv(
        output_file
    )

    print(
        cm_df.to_string()
    )

    print(
        f"\nSaved confusion matrix:"
    )

    print(
        f"  {output_file}"
    )

    # --------------------------------------------------------
    # Row-normalized confusion matrix
    # --------------------------------------------------------

    row_sums = (
        cm.sum(axis=1)
    )

    normalized_cm = np.divide(
        cm,
        row_sums[:, None],
        out=np.zeros_like(
            cm,
            dtype=np.float64
        ),
        where=row_sums[:, None] != 0
    )

    normalized_df = pd.DataFrame(
        normalized_cm,
        index=CLASSES,
        columns=CLASSES
    )

    normalized_df.index.name = (
        "true_class"
    )

    normalized_file = (
        output_dir /
        "test_confusion_matrix_normalized.csv"
    )

    normalized_df.to_csv(
        normalized_file
    )

    print(
        f"Saved normalized confusion matrix:"
    )

    print(
        f"  {normalized_file}"
    )

    return cm


# ============================================================
# CONFIDENCE BINS
# ============================================================

def create_confidence_bins(
    df,
    output_dir
):

    probabilities = (
        df[PROBABILITY_COLUMNS]
        .to_numpy(dtype=np.float64)
    )

    confidence = (
        probabilities.max(axis=1)
    )

    correct = (
        df["true_index"].astype(int).to_numpy()
        ==
        df["predicted_index"].astype(int).to_numpy()
    )

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    rows = []

    for lower, upper in zip(
        bins[:-1],
        bins[1:]
    ):

        if upper == 1.0:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            accuracy = np.nan
            mean_confidence = np.nan

        else:

            accuracy = (
                correct[mask].mean()
            )

            mean_confidence = (
                confidence[mask].mean()
            )

        rows.append({

            "confidence_lower":
                lower,

            "confidence_upper":
                upper,

            "samples":
                count,

            "accuracy":
                accuracy,

            "mean_confidence":
                mean_confidence,
        })

    bins_df = pd.DataFrame(
        rows
    )

    output_file = (
        output_dir /
        "confidence_bins.csv"
    )

    bins_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved confidence-bin analysis:"
    )

    print(
        f"  {output_file}"
    )

    return bins_df


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model_key):

    config = MODELS[model_key]

    model_name = (
        config["display_name"]
    )

    output_dir = (
        OUTPUT_ROOT /
        model_key
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_predictions(
        model_key
    )

    # ========================================================
    # BASIC METRICS
    # ========================================================

    print_section(
        "OVERALL TEST RESULTS"
    )

    metrics = calculate_basic_metrics(
        df
    )

    print(
        f"Model              : "
        f"{model_name}"
    )

    print(
        f"Samples            : "
        f"{metrics['samples']}"
    )

    print(
        f"Correct            : "
        f"{metrics['correct']}"
    )

    print(
        f"Wrong              : "
        f"{metrics['wrong']}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{metrics['macro_recall']:.4f}"
    )

    # ========================================================
    # RELIABILITY
    # ========================================================

    reliability_metrics = (
        create_reliability_analysis(
            df,
            model_name,
            output_dir
        )
    )

    metrics.update(
        reliability_metrics
    )

    # ========================================================
    # PER-CLASS
    # ========================================================

    per_class_df = (
        create_per_class_analysis(
            df,
            model_name,
            output_dir
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    create_confusion_matrix(
        df,
        model_name,
        output_dir
    )

    # ========================================================
    # CONFIDENCE BINS
    # ========================================================

    create_confidence_bins(
        df,
        output_dir
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics["model"] = model_name

    metrics["model_key"] = model_key

    metrics["num_classes"] = NUM_CLASSES

    metrics["classes"] = CLASSES

    metrics_file = (
        output_dir /
        "test_metrics.json"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"\nSaved metrics:"
    )

    print(
        f"  {metrics_file}"
    )

    # ========================================================
    # PRINT TOP / WORST CLASSES
    # ========================================================

    print_section(
        "CLASS PERFORMANCE SUMMARY"
    )

    sorted_classes = (
        per_class_df
        .sort_values(
            "f1",
            ascending=False
        )
    )

    print(
        "\nBest classes by F1:"
    )

    print(
        sorted_classes[
            [
                "class",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .head(3)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print(
        "\nWeakest classes by F1:"
    )

    print(
        sorted_classes[
            [
                "class",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .tail(3)
        .sort_values(
            "f1"
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    return metrics


# ============================================================
# COMPARE ALL CNN MODELS
# ============================================================

def create_model_comparison(
    all_metrics
):

    print_header(
        "CNN ARCHITECTURE COMPARISON"
    )

    rows = []

    for model_key, metrics in (
        all_metrics.items()
    ):

        rows.append({

            "model":
                metrics["model"],

            "samples":
                metrics["samples"],

            "correct":
                metrics["correct"],

            "wrong":
                metrics["wrong"],

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "weighted_f1":
                metrics["weighted_f1"],

            "macro_precision":
                metrics["macro_precision"],

            "macro_recall":
                metrics["macro_recall"],

            "mean_confidence_correct":
                metrics[
                    "mean_confidence_correct"
                ],

            "mean_confidence_wrong":
                metrics[
                    "mean_confidence_wrong"
                ],

            "mean_entropy_correct":
                metrics[
                    "mean_entropy_correct"
                ],

            "mean_entropy_wrong":
                metrics[
                    "mean_entropy_wrong"
                ],

            "mean_margin_correct":
                metrics[
                    "mean_margin_correct"
                ],

            "mean_margin_wrong":
                metrics[
                    "mean_margin_wrong"
                ],
        })

    comparison_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Sort by Macro F1 first
    # --------------------------------------------------------

    comparison_df = (
        comparison_df
        .sort_values(
            [
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_ROOT /
        "cnn_architecture_comparison.csv"
    )

    comparison_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    display_columns = [
        "model",
        "samples",
        "correct",
        "wrong",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    print(
        comparison_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print(
        f"\nSaved comparison:"
    )

    print(
        f"  {output_file}"
    )

    return comparison_df


# ============================================================
# BEST MODEL SUMMARY
# ============================================================

def print_best_model_summary(
    comparison_df
):

    print_header(
        "BEST CNN ARCHITECTURE"
    )

    # --------------------------------------------------------
    # Best accuracy
    # --------------------------------------------------------

    best_accuracy = (
        comparison_df
        .sort_values(
            "accuracy",
            ascending=False
        )
        .iloc[0]
    )

    # --------------------------------------------------------
    # Best balanced accuracy
    # --------------------------------------------------------

    best_balanced = (
        comparison_df
        .sort_values(
            "balanced_accuracy",
            ascending=False
        )
        .iloc[0]
    )

    # --------------------------------------------------------
    # Best Macro F1
    # --------------------------------------------------------

    best_f1 = (
        comparison_df
        .sort_values(
            "macro_f1",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "BEST ACCURACY"
    )

    print(
        f"  Model    : "
        f"{best_accuracy['model']}"
    )

    print(
        f"  Accuracy : "
        f"{best_accuracy['accuracy']:.4f}"
    )

    print(
        "\nBEST BALANCED ACCURACY"
    )

    print(
        f"  Model            : "
        f"{best_balanced['model']}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{best_balanced['balanced_accuracy']:.4f}"
    )

    print(
        "\nBEST MACRO F1"
    )

    print(
        f"  Model    : "
        f"{best_f1['model']}"
    )

    print(
        f"  Macro F1 : "
        f"{best_f1['macro_f1']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Hybrid_V3 CNN "
            "architectures from saved predictions."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "resnet50",
            "efficientnet_v2_s",
            "efficientnet_b3",
            "mobilenet_v3_large",
            "all",
        ],
        help=(
            "CNN model to evaluate, "
            "or 'all'."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print_header(
        "HYBRID_V3 CNN ARCHITECTURE EVALUATION"
    )

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Test CSV     : "
        f"{TEST_CSV}"
    )

    print(
        f"Output root  : "
        f"{OUTPUT_ROOT}"
    )

    if args.model == "all":

        selected_models = list(
            MODELS.keys()
        )

    else:

        selected_models = [
            args.model
        ]

    print(
        f"Models       : "
        f"{len(selected_models)}"
    )

    # ========================================================
    # CHECK TEST CSV
    # ========================================================

    if not TEST_CSV.exists():

        print(
            "\nWARNING:"
        )

        print(
            f"Test dataset CSV was not found:"
        )

        print(
            f"  {TEST_CSV}"
        )

        print(
            "\nThis is not required for evaluating "
            "the saved prediction files, so evaluation "
            "will continue."
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    all_metrics = {}

    for model_key in selected_models:

        try:

            metrics = evaluate_model(
                model_key
            )

            all_metrics[
                model_key
            ] = metrics

        except Exception as error:

            print(
                "\n"
                + "!" * 78
            )

            print(
                f"ERROR EVALUATING "
                f"{MODELS[model_key]['display_name']}"
            )

            print(
                "!" * 78
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "\nThe remaining models will "
                "continue to be evaluated."
            )

    # ========================================================
    # GLOBAL COMPARISON
    # ========================================================

    if len(all_metrics) > 0:

        comparison_df = (
            create_model_comparison(
                all_metrics
            )
        )

        print_best_model_summary(
            comparison_df
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print_header(
        "EVALUATION COMPLETE"
    )

    print(
        f"Successfully evaluated : "
        f"{len(all_metrics)} / "
        f"{len(selected_models)}"
    )

    if len(all_metrics) > 0:

        print(
            "\nModels:"
        )

        for model_key in all_metrics:

            metrics = (
                all_metrics[
                    model_key
                ]
            )

            print(
                f"  {metrics['model']:<24}"
                f"Accuracy="
                f"{metrics['accuracy']:.4f} "
                f"Macro-F1="
                f"{metrics['macro_f1']:.4f}"
            )

    print(
        "\nOutput root:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print(
        "\nPer-model outputs:"
    )

    print(
        "  ├── test_predictions.csv"
    )

    print(
        "  ├── test_confusion_matrix.csv"
    )

    print(
        "  ├── test_confusion_matrix_normalized.csv"
    )

    print(
        "  ├── test_metrics.json"
    )

    print(
        "  ├── per_class_metrics.csv"
    )

    print(
        "  ├── reliability_analysis.csv"
    )

    print(
        "  └── confidence_bins.csv"
    )

    if args.model == "all":

        print(
            "\nGlobal comparison:"
        )

        print(
            "  └── cnn_architecture_comparison.csv"
        )

    print(
        "\n✓ CNN evaluation completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()