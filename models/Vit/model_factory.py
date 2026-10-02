import timm
import torch.nn as nn


NUM_CLASSES = 8


def create_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Create a ViT-family model using timm.

    Supported:
        - deit_tiny
        - swin_tiny
    """

    model_map = {
        "deit_tiny": "deit_tiny_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_map.keys())}"
        )

    model = timm.create_model(
        model_map[model_name],
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model