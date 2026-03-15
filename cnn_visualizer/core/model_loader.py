"""
Responsible for instantiating the CNN model and finding its Conv2d layers.
"""
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def load_model(model_name: str) -> tuple[nn.Module, list[str]]:
    """
    Loads a pre-trained model and parses it to find all nn.Conv2d layers.
    
    Args:
        model_name (str): The name of the model to load (e.g., "ResNet18").
        
    Returns:
        tuple[nn.Module, list[str]]: The loaded model and a list of Conv2d layer names.
    """
    if model_name == "ResNet18":
        model = models.resnet18(weights="DEFAULT")
    elif model_name == "VGG16":
        model = models.vgg16(weights="DEFAULT")
    elif model_name == "AlexNet":
        model = models.alexnet(weights="DEFAULT")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    layer_names = []

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer_names.append(name)

    return model, layer_names
