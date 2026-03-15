"""
Handles rendering feature map tensors into lists of displayable PIL images.
"""
from __future__ import annotations
import torch
import cv2
import numpy as np
from PIL import Image

from core.diagnostics import check_dead_relus, check_redundancy, compute_saturation, compute_snr, compute_diversity

def process_tensor_to_images(tensor: torch.Tensor, max_channels: int = 36, is_live: bool = False, use_heatmap: bool = True, ema_sums: torch.Tensor | None = None, alpha: float = 0.15, dead_threshold: float = 1e-5, cached_redundancy: torch.Tensor | None = None, frame_count: int = 0) -> tuple[list, torch.Tensor | None, dict, torch.Tensor | None, torch.Tensor | None]:
    """
    Takes a raw PyTorch tensor of shape (1, Channels, Height, Width), normalizes
    the activations globally, sorts them by descending activation magnitude (using EMA for stability),
    and converts them into PIL images.
    
    Returns:
       images: list of tuples -> (PIL_Image, label_string, diagnostic_status_class)
       ema_sums: tensor
       health_metrics: dict containing percentage of dead neurons, saturation, snr
       redundant_mask: tensor
    """
    health_metrics = {"dead_percent": 0.0, "total": 0, "dead": 0}

    if tensor is None or len(tensor.shape) != 4:
        return [], ema_sums, health_metrics, cached_redundancy, None

    # Remove batch dimension: (1, C, H, W) -> (C, H, W)
    tensor = tensor.squeeze(0)

    # Externalized diagnostics logic
    dead_relu_metrics = check_dead_relus(tensor, dead_threshold)
    dead_mask = dead_relu_metrics["dead_mask"]
    num_total = dead_relu_metrics["total"]

    health_metrics["total"] = num_total
    health_metrics["dead"] = dead_relu_metrics["dead"]
    health_metrics["dead_percent"] = dead_relu_metrics["dead_percent"]

    # SNR and Saturation
    health_metrics["snr"] = compute_snr(tensor)

    # Calculate global redundancy every 10 frames to save CPU
    sim_matrix = None
    if cached_redundancy is None or frame_count % 10 == 0:
        redundant_mask, sim_matrix = check_redundancy(tensor, 0.95)
    else:
        redundant_mask = cached_redundancy

    health_metrics["diversity"] = compute_diversity(sim_matrix, num_total)

    images = []

    # Special Case: If it's a 3-channel tensor (like the input layer) and we want color
    if is_live and num_total == 3:
        feature_map = tensor.clone()

        fm_min = feature_map.min()
        fm_max = feature_map.max()
        if fm_max - fm_min > 1e-6:
            feature_map = (feature_map - fm_min) / (fm_max - fm_min)
        else:
            feature_map = torch.zeros_like(feature_map)

        # Shape (3, H, W) -> (H, W, 3) for PIL
        feature_map = feature_map.permute(1, 2, 0)
        feature_map = (feature_map * 255).byte().numpy()

        img = Image.fromarray(feature_map, mode='RGB')
        images.append((img, "RGB Input", "normal"))
        return images, ema_sums, health_metrics, cached_redundancy, None

    # Standard Case: Grayscale / Heatmap feature maps
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    # Store saturation info calculated on relative range
    health_metrics["saturation"] = compute_saturation(tensor - tensor_min)
    
    if tensor_max - tensor_min > 1e-6:
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    else:
        tensor = torch.zeros_like(tensor)

    # Smart Feature Sorting via EMA
    current_channel_sums = tensor.sum(dim=(1, 2))

    if ema_sums is None or ema_sums.shape[0] != current_channel_sums.shape[0]:
        ema_sums = current_channel_sums
    else:
        ema_sums = (1 - alpha) * ema_sums + alpha * current_channel_sums

    sorted_indices = torch.argsort(ema_sums, descending=True)
    num_channels = min(tensor.shape[0], max_channels)

    for i in range(num_channels):
        original_idx = sorted_indices[i].item()
        feature_map = tensor[original_idx]  # Shape: (H, W)

        feature_map = (feature_map * 255).byte().numpy()

        if use_heatmap:
            colored_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_VIRIDIS)
            colored_map_rgb = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(colored_map_rgb, mode='RGB')
        else:
            img = Image.fromarray(feature_map, mode='L')

        # Apply diagnostic classes
        diag_class = "normal"
        if dead_mask[original_idx]:
            diag_class = "dead"
        elif redundant_mask[original_idx]:
            diag_class = "redundant"

        images.append((img, f"Channel {original_idx}", diag_class))

    return images, ema_sums, health_metrics, redundant_mask, sim_matrix
