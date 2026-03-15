"""
Handles rendering feature map tensors into lists of displayable PIL images.
"""
from __future__ import annotations
import torch
import cv2
import numpy as np
from PIL import Image

from core.diagnostics import (
    check_dead_relus,
    check_redundancy,
    compute_saturation,
    compute_snr,
    compute_diversity,
)


def process_tensor_to_images(
    tensor: torch.Tensor,
    max_channels: int = 10000,  # effectively unlimited — show all channels
    is_live: bool = False,
    use_heatmap: bool = True,
    ema_sums: torch.Tensor | None = None,
    alpha: float = 0.15,
    dead_threshold: float = 1e-5,
    cached_redundancy: torch.Tensor | None = None,
    frame_count: int = 0,
) -> tuple[list, torch.Tensor | None, dict, torch.Tensor | None, torch.Tensor | None]:
    """
    Converts a (1, C, H, W) activation tensor into a list of displayable
    PIL images with associated diagnostic metrics.

    Returns:
        images:           list of (PIL_Image, label_str, diag_class, channel_idx)
        ema_sums:         updated EMA channel sums tensor
        health_metrics:   dict with dead %, saturation, snr, diversity
        redundant_mask:   bool tensor shape (C,)
        sim_matrix:       float tensor shape (C, C) or None if throttled
    """
    health_metrics: dict = {"dead_percent": 0.0, "total": 0, "dead": 0}

    if tensor is None or len(tensor.shape) != 4:
        return [], ema_sums, health_metrics, cached_redundancy, None

    # Remove batch dim: (1, C, H, W) → (C, H, W)
    tensor = tensor.squeeze(0)

    # ── Dead ReLU detection ───────────────────────────────────────────────
    dead_relu_metrics = check_dead_relus(tensor, dead_threshold)
    dead_mask = dead_relu_metrics["dead_mask"]
    num_total = dead_relu_metrics["total"]

    health_metrics["total"]        = num_total
    health_metrics["dead"]         = dead_relu_metrics["dead"]
    health_metrics["dead_percent"] = dead_relu_metrics["dead_percent"]

    # ── SNR (on raw tensor before normalisation) ──────────────────────────
    health_metrics["snr"] = compute_snr(tensor)

    # ── Redundancy / diversity (throttled every 10 frames) ───────────────
    sim_matrix = None
    if cached_redundancy is None or frame_count % 10 == 0:
        redundant_mask, sim_matrix = check_redundancy(tensor, 0.95)
    else:
        redundant_mask = cached_redundancy

    health_metrics["diversity"] = compute_diversity(sim_matrix, num_total)

    images: list = []

    # ── Special case: 3-channel RGB input tensor in live mode ────────────
    if is_live and num_total == 3:
        feature_map = tensor.clone()
        fm_min, fm_max = feature_map.min(), feature_map.max()
        if fm_max - fm_min > 1e-6:
            feature_map = (feature_map - fm_min) / (fm_max - fm_min)
        else:
            feature_map = torch.zeros_like(feature_map)

        feature_map = feature_map.permute(1, 2, 0)
        feature_map = (feature_map * 255).byte().numpy()
        img = Image.fromarray(feature_map, mode="RGB")
        # 4-tuple: explicit channel_idx = -1 (not individually selectable)
        images.append((img, "RGB Input", "normal", -1))
        return images, ema_sums, health_metrics, cached_redundancy, None

    # ── Global normalisation ─────────────────────────────────────────────
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # FIX #15: compute_saturation must receive the NORMALISED [0, 1] tensor.
    # Previously it was called on (tensor - tensor_min) before dividing by
    # the range, so values were in [0, range] not [0, 1].
    if tensor_max - tensor_min > 1e-6:
        tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min)
    else:
        tensor_norm = torch.zeros_like(tensor)

    # Saturation now computed on the correctly normalised tensor
    health_metrics["saturation"] = compute_saturation(tensor_norm)

    # Use the normalised tensor from here on
    tensor = tensor_norm

    # ── EMA-based channel sorting ─────────────────────────────────────────
    current_channel_sums = tensor.sum(dim=(1, 2))

    if ema_sums is None or ema_sums.shape[0] != current_channel_sums.shape[0]:
        ema_sums = current_channel_sums
    else:
        ema_sums = (1 - alpha) * ema_sums + alpha * current_channel_sums

    sorted_indices = torch.argsort(ema_sums, descending=True)
    num_channels = min(tensor.shape[0], max_channels)

    for i in range(num_channels):
        original_idx = sorted_indices[i].item()
        feature_map = tensor[original_idx]  # (H, W)

        feature_map_np = (feature_map * 255).byte().numpy()

        if use_heatmap:
            colored = cv2.applyColorMap(feature_map_np, cv2.COLORMAP_VIRIDIS)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(colored_rgb, mode="RGB")
        else:
            img = Image.fromarray(feature_map_np, mode="L")

        diag_class = "normal"
        if dead_mask[original_idx]:
            diag_class = "dead"
        elif redundant_mask[original_idx]:
            diag_class = "redundant"

        # FIX #13: include explicit channel index as 4th tuple element so
        # GridView doesn't have to parse it out of the label string.
        images.append((img, f"Channel {original_idx}", diag_class, original_idx))

    return images, ema_sums, health_metrics, redundant_mask, sim_matrix