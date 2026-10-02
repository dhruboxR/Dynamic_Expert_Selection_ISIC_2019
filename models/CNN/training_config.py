"""
Hybrid_v3 CNN Training Configuration
====================================

Central configuration for all CNN experiments.

The goal is to keep the training protocol identical across
all CNN architectures so their results can be compared fairly.
"""

from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "Dataset ISIC 2019"

SPLIT_ROOT = PROJECT_ROOT / "outputs" / "dataset_split"

TRAIN_CSV = SPLIT_ROOT / "train.csv"
VAL_CSV = SPLIT_ROOT / "val.csv"
TEST_CSV = SPLIT_ROOT / "test.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn"

CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
LOG_ROOT = OUTPUT_ROOT / "logs"
RESULT_ROOT = OUTPUT_ROOT / "results"


# =============================================================================
# DATASET
# =============================================================================

NUM_CLASSES = 8

CLASS_NAMES = [
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
]

IMAGE_SIZE = 224


# =============================================================================
# DATA LOADING
# =============================================================================

BATCH_SIZE = 16

NUM_WORKERS = 4

PIN_MEMORY = True

PERSISTENT_WORKERS = True

PREFETCH_FACTOR = 2


# =============================================================================
# TRAINING
# =============================================================================

EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

OPTIMIZER = "adamw"

BETAS = (0.9, 0.999)

EPS = 1e-8


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

SCHEDULER = "cosine"

MIN_LR = 1e-6

WARMUP_EPOCHS = 2


# =============================================================================
# CLASS IMBALANCE
# =============================================================================

# We will compare class-aware loss strategies experimentally.
#
# "effective_number" is the default because the ISIC 2019 training set
# has a severe imbalance between NV and DF.
#
# Other supported modes in the future:
#     "none"
#     "balanced"
#     "effective_number"

CLASS_WEIGHT_MODE = "effective_number"

EFFECTIVE_NUMBER_BETA = 0.9999


# =============================================================================
# SAMPLING
# =============================================================================

# Do not blindly oversample minority classes.
#
# The default training experiment uses the original training distribution
# together with class-aware loss weighting.

USE_WEIGHTED_SAMPLER = False


# =============================================================================
# AUGMENTATION
# =============================================================================

USE_TRAIN_AUGMENTATION = True

USE_HORIZONTAL_FLIP = True

USE_VERTICAL_FLIP = True

USE_ROTATION = True

USE_COLOR_AUGMENTATION = True

USE_RANDOM_ERASING = True


# =============================================================================
# REGULARIZATION
# =============================================================================

LABEL_SMOOTHING = 0.05

DROPOUT = 0.0


# =============================================================================
# MIXED PRECISION
# =============================================================================

USE_AMP = True


# =============================================================================
# GRADIENT CONTROL
# =============================================================================

GRADIENT_CLIP_NORM = 1.0


# =============================================================================
# EARLY STOPPING
# =============================================================================

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 7

EARLY_STOPPING_MIN_DELTA = 1e-4


# =============================================================================
# CHECKPOINTING
# =============================================================================

SAVE_BEST_ONLY = True

MONITOR_METRIC = "balanced_accuracy"

MONITOR_MODE = "max"


# =============================================================================
# EVALUATION
# =============================================================================

EVALUATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "cohen_kappa",
]


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = "cuda"


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# CNN architectures available through model_factory.py

CNN_MODELS = [
    "efficientnet_v2_s",
    "efficientnet_b3",
    "convnext_tiny",
    "densenet121",
    "resnet50",
]


# =============================================================================
# PRETRAINING
# =============================================================================

PRETRAINED = True


# =============================================================================
# FREEZE / FINE-TUNING
# =============================================================================

# Full fine-tuning is preferred for the final experiments.
#
# We can later perform a controlled comparison with:
#   1. frozen backbone
#   2. partially frozen backbone
#   3. full fine-tuning

FREEZE_BACKBONE = False

UNFREEZE_AFTER_EPOCH = 0


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_config():
    """Validate the training configuration before starting an experiment."""

    assert NUM_CLASSES == len(CLASS_NAMES)

    assert IMAGE_SIZE > 0

    assert BATCH_SIZE > 0

    assert EPOCHS > 0

    assert LEARNING_RATE > 0

    assert WEIGHT_DECAY >= 0

    assert NUM_WORKERS >= 0

    assert CLASS_WEIGHT_MODE in {
        "none",
        "balanced",
        "effective_number",
    }

    assert MONITOR_MODE in {
        "min",
        "max",
    }

    assert MONITOR_METRIC in {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "cohen_kappa",
    }

    for model_name in CNN_MODELS:
        assert isinstance(model_name, str)

    return True


def print_config():
    """Print the complete training configuration."""

    print("=" * 72)
    print("HYBRID_V3 CNN TRAINING CONFIGURATION")
    print("=" * 72)

    print()
    print("PROJECT")
    print("-" * 72)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Dataset root       : {DATASET_ROOT}")
    print(f"Train CSV          : {TRAIN_CSV}")
    print(f"Validation CSV     : {VAL_CSV}")
    print(f"Test CSV           : {TEST_CSV}")

    print()
    print("DATA")
    print("-" * 72)
    print(f"Classes            : {NUM_CLASSES}")
    print(f"Class names        : {CLASS_NAMES}")
    print(f"Image size         : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Workers            : {NUM_WORKERS}")

    print()
    print("OPTIMIZATION")
    print("-" * 72)
    print(f"Epochs             : {EPOCHS}")
    print(f"Optimizer          : {OPTIMIZER}")
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Weight decay       : {WEIGHT_DECAY}")
    print(f"Scheduler          : {SCHEDULER}")
    print(f"Minimum LR         : {MIN_LR}")
    print(f"Warmup epochs      : {WARMUP_EPOCHS}")

    print()
    print("CLASS IMBALANCE")
    print("-" * 72)
    print(f"Weight mode        : {CLASS_WEIGHT_MODE}")
    print(f"Effective beta     : {EFFECTIVE_NUMBER_BETA}")
    print(f"Weighted sampler   : {USE_WEIGHTED_SAMPLER}")

    print()
    print("REGULARIZATION")
    print("-" * 72)
    print(f"Label smoothing    : {LABEL_SMOOTHING}")
    print(f"Gradient clipping  : {GRADIENT_CLIP_NORM}")
    print(f"AMP                : {USE_AMP}")

    print()
    print("EARLY STOPPING")
    print("-" * 72)
    print(f"Enabled            : {EARLY_STOPPING}")
    print(f"Patience           : {EARLY_STOPPING_PATIENCE}")
    print(f"Monitor            : {MONITOR_METRIC}")
    print(f"Mode               : {MONITOR_MODE}")

    print()
    print("CNN MODELS")
    print("-" * 72)

    for i, model_name in enumerate(CNN_MODELS, start=1):
        print(f"{i}. {model_name}")

    print()
    print("REPRODUCIBILITY")
    print("-" * 72)
    print(f"Seed               : {SEED}")
    print(f"Device             : {DEVICE}")
    print(f"Pretrained         : {PRETRAINED}")
    print(f"Full fine-tuning   : {not FREEZE_BACKBONE}")

    print()
    print("=" * 72)
    print("CONFIGURATION VALID")
    print("=" * 72)


if __name__ == "__main__":

    validate_config()

    print_config()