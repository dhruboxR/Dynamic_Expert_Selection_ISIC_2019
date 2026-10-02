"""
Hybrid_v3 CNN Model Factory
===========================

Creates pretrained CNN classifiers for the ISIC 2019 dataset.

Supported models:
    - efficientnet_v2_s
    - efficientnet_b3
    - convnext_tiny
    - densenet121
    - resnet50
    - mobilenet_v3_large

All models:
    - Use ImageNet pretrained weights
    - Output 8 ISIC 2019 classes
    - Replace the original classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


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


# =============================================================================
# SUPPORTED MODELS
# =============================================================================

SUPPORTED_MODELS = [
    "efficientnet_v2_s",
    "efficientnet_b3",
    "convnext_tiny",
    "densenet121",
    "resnet50",
    "mobilenet_v3_large",
]


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a pretrained CNN classifier.

    Parameters
    ----------
    model_name : str
        Name of the CNN architecture.

    num_classes : int
        Number of output classes.

    pretrained : bool
        Whether to load ImageNet pretrained weights.

    Returns
    -------
    torch.nn.Module
        Configured CNN classifier.
    """

    model_name = model_name.lower().strip()

    # -------------------------------------------------------------------------
    # Validate model name
    # -------------------------------------------------------------------------

    if model_name not in SUPPORTED_MODELS:

        raise ValueError(
            f"Unknown model: '{model_name}'\n"
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # =========================================================================
    # EfficientNetV2-S
    # =========================================================================

    if model_name == "efficientnet_v2_s":

        weights = (
            models.EfficientNet_V2_S_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_v2_s(
            weights=weights
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # EfficientNet-B3
    # =========================================================================

    elif model_name == "efficientnet_b3":

        weights = (
            models.EfficientNet_B3_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_b3(
            weights=weights
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # ConvNeXt-Tiny
    # =========================================================================

    elif model_name == "convnext_tiny":

        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.convnext_tiny(
            weights=weights
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # DenseNet-121
    # =========================================================================

    elif model_name == "densenet121":

        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.densenet121(
            weights=weights
        )

        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # ResNet-50
    # =========================================================================

    elif model_name == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # MobileNetV3-Large
    # =========================================================================

    elif model_name == "mobilenet_v3_large":

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.mobilenet_v3_large(
            weights=weights
        )

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    # =========================================================================
    # Safety check
    # =========================================================================

    else:

        raise RuntimeError(
            f"Model '{model_name}' was not handled."
        )

    return model


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(
    model: nn.Module
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# =============================================================================
# MODEL SUMMARY
# =============================================================================

def model_summary(
    model_name: str
) -> None:
    """
    Print a compact model summary.
    """

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    total, trainable = count_parameters(
        model
    )

    print("=" * 70)
    print("HYBRID_V3 CNN MODEL")
    print("=" * 70)

    print(
        f"Model           : {model_name}"
    )

    print(
        f"Classes         : {NUM_CLASSES}"
    )

    print(
        f"Pretrained      : ImageNet"
    )

    print(
        f"Total params    : {total:,}"
    )

    print(
        f"Trainable params: {trainable:,}"
    )

    print("=" * 70)


# =============================================================================
# TEST ALL MODELS
# =============================================================================

def test_all_models() -> None:
    """
    Instantiate every supported CNN and verify
    its output shape.
    """

    print("=" * 70)
    print("HYBRID_V3 CNN MODEL FACTORY TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # -------------------------------------------------------------------------
    # Fake 224x224 RGB image
    # -------------------------------------------------------------------------

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    print()

    # -------------------------------------------------------------------------
    # Test every model
    # -------------------------------------------------------------------------

    for model_name in SUPPORTED_MODELS:

        print("-" * 70)

        print(
            f"Testing: {model_name}"
        )

        model = create_model(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=True,
        )

        model = model.to(
            device
        )

        model.eval()

        total, trainable = count_parameters(
            model
        )

        with torch.no_grad():

            output = model(x)

        print(
            f"Output shape    : "
            f"{tuple(output.shape)}"
        )

        print(
            f"Total params    : "
            f"{total:,}"
        )

        print(
            f"Trainable params: "
            f"{trainable:,}"
        )

        assert output.shape == (
            1,
            NUM_CLASSES,
        ), (
            f"Unexpected output shape "
            f"for {model_name}: "
            f"{output.shape}"
        )

        print(
            "Status          : PASS"
        )

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    print()

    print("=" * 70)

    print(
        "ALL CNN MODEL FACTORY TESTS PASSED"
    )

    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    test_all_models()