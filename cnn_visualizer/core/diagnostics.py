"""
Diagnostic functions for analyzing CNN feature maps and input distributions.
All functions are pure and perform dead ReLU detection, redundancy checks, etc.
"""
from __future__ import annotations
import torch
from torch import Tensor

def check_dead_relus(tensor: Tensor, threshold: float) -> dict:
    """
    Checks for dead ReLUs by evaluating variance across the spatial dimensions.
    Returns a dict with 'dead_percent', 'total', 'dead', and the 'dead_mask'.
    
    Args:
        tensor (Tensor): Output activation tensor of shape (C, H, W)
        threshold (float): Variance threshold below which a neuron is flagged dead.
    """
    variances = tensor.var(dim=(1, 2))
    dead_mask = variances < threshold
    num_dead = dead_mask.sum().item()
    num_total = tensor.shape[0]

    return {
        "dead_percent": (num_dead / max(1, num_total)) * 100.0,
        "total": num_total,
        "dead": num_dead,
        "dead_mask": dead_mask
    }

def compute_saturation(tensor: Tensor, saturation_threshold: float = 0.95) -> dict:
    """
    Computes the percentage of positive activations that are saturated (close to the maximum value).
    """
    total = tensor.numel()
    if total == 0:
        return {"layer_score": 0.0, "level": "healthy", "per_channel": torch.zeros(tensor.shape[0])}
        
    v_max = tensor.max()
    if v_max <= 1e-6:
        return {"layer_score": 0.0, "level": "healthy", "per_channel": torch.zeros(tensor.shape[0])}
        
    saturated_mask = tensor >= (v_max * saturation_threshold)
    layer_score = saturated_mask.sum().item() / total
    
    per_channel = saturated_mask.sum(dim=(1, 2)).float() / (tensor.shape[1] * tensor.shape[2])
    
    if layer_score < 0.10:
        level = "healthy"
    elif layer_score < 0.30:
        level = "mild"
    else:
        level = "saturated"
        
    return {"layer_score": layer_score, "level": level, "per_channel": per_channel}

def compute_snr(tensor: Tensor) -> dict:
    """
    Computes a basic Signal-to-Noise Ratio (Mean / Std Dev) per layer to gauge feature robustness.
    """
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()
    
    if std_val < 1e-6:
        snr = 0.0
    else:
        snr = abs(mean_val) / std_val
        
    if snr < 0.5:
        level = "weak"
    elif snr < 1.5:
        level = "moderate"
    else:
        level = "strong"
        
    return {"snr": snr, "level": level}

def compute_diversity(sim_matrix: torch.Tensor | None, num_channels: int) -> dict:
    """
    Computes a layer-wide filter diversity score from the cosine similarity matrix.

    Args:
        sim_matrix: (C, C) absolute cosine similarity matrix, diagonal zeroed.
                    May be None if computation was throttled this frame.
        num_channels: total number of filters C (used for edge cases)

    Returns:
        {
            "score": float,         # 0.0–1.0. Higher = more diverse filters.
            "score_pct": int,       # 0–100 integer for display
            "level": str,           # "low" | "moderate" | "high"
            "per_channel_max": Tensor | None
                # shape (C,), max similarity each filter has with any other filter.
                # High value = this filter is very similar to at least one other.
                # None if sim_matrix is None.
        }
    """
    if sim_matrix is None or num_channels <= 1:
        return {
            "score": 1.0, "score_pct": 100,
            "level": "n/a", "per_channel_max": None
        }

    # Mean of all off-diagonal elements (diagonal was already zeroed)
    num_off_diag = num_channels * (num_channels - 1)
    mean_similarity = sim_matrix.sum().item() / max(num_off_diag, 1)

    # Diversity is the inverse of mean similarity
    score = 1.0 - mean_similarity
    score = max(0.0, min(1.0, score))

    level = "low" if score < 0.4 else "moderate" if score < 0.7 else "high"
    per_channel_max = sim_matrix.max(dim=1).values  # (C,)

    return {
        "score": score,
        "score_pct": int(score * 100),
        "level": level,
        "per_channel_max": per_channel_max
    }


def check_redundancy(tensor: Tensor, similarity_threshold: float = 0.95) -> tuple[Tensor, Tensor]:
    """
    Checks for redundant filters via pairwise cosine similarity of flattened channels.
    
    Args:
        tensor (Tensor): Output activation tensor of shape (C, H, W)
        similarity_threshold (float): Similarity above which a filter is considered redundant.
    
    Returns:
        tuple[Tensor, Tensor]: A boolean redundant mask (length C), and the (C, C) cosine similarity matrix.
    """
    num_total = tensor.shape[0]
    flat_features = tensor.view(num_total, -1)

    norms = torch.norm(flat_features, p=2, dim=1, keepdim=True)
    norms[norms < 1e-8] = 1.0
    normalized_features = flat_features / norms

    sim_matrix = torch.abs(torch.mm(normalized_features, normalized_features.t()))
    sim_matrix.fill_diagonal_(0.0)

    lower_tri = torch.tril(sim_matrix, diagonal=-1)
    redundant_mask = (lower_tri > similarity_threshold).any(dim=1)
    return redundant_mask, sim_matrix

def check_input_sanity(input_stats: dict) -> str:
    """
    Evaluates basic input tensor mean/std to ensure it aligns with ImageNet norms.
    
    Args:
        input_stats (dict): Dictionary containing 'mean' and 'std' keys.
        
    Returns:
        str: Diagnostic string describing the sanity check status.
    """
    if input_stats and input_stats.get('std') is not None and input_stats['std'] != 0.0:
        if abs(input_stats['mean']) > 0.8 or input_stats['std'] < 0.2:
            return "⚠ Input Norm Error!\n"
        else:
            return "Input Dist: OK\n"
    return "Input Dist: N/A\n"
