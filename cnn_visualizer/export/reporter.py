"""
Responsible for generating and saving diagnostic reports.
"""
from __future__ import annotations

def generate_report(layer_name: str, health_metrics: dict | None, input_stats: dict | None, threshold: float) -> str:
    """
    Generates a diagnostic report as a string.
    
    Args:
        layer_name (str): The name of the layer being analyzed.
        health_metrics (dict): Health metrics for the layer (dead neurons, etc.).
        input_stats (dict): Input normalisation statistics.
        threshold (float): Dead ReLU threshold.
        
    Returns:
        str: The generated report string.
    """
    lines = []
    lines.append("=== CNN VISUALIZER PRO: DIAGNOSTIC REPORT ===")
    lines.append(f"Target Layer: {layer_name}\n")

    lines.append("--- LAYER HEALTH ---")
    if health_metrics:
        lines.append(f"Total Filters: {health_metrics['total']}")
        lines.append(f"Dead ReLUs Detected: {health_metrics['dead']} "
                     f"({health_metrics['dead_percent']:.2f}%)")
        lines.append(f"Threshold applied: {threshold:.6f} variance\n")

        if "saturation" in health_metrics:
            lines.append(f"Saturation Score: {health_metrics['saturation']['layer_score']:.2%} "
                         f"[{health_metrics['saturation']['level']}]")
        if "snr" in health_metrics:
            lines.append(f"Signal-to-Noise Ratio (SNR): {health_metrics['snr']['snr']:.2f} "
                         f"[{health_metrics['snr']['level']}]")
        if "diversity" in health_metrics:
            lines.append(f"Diversity Score: {health_metrics['diversity']['score_pct']}% "
                         f"[{health_metrics['diversity']['level']}]\n")

    lines.append("--- INPUT SANITY ---")
    if input_stats:
        lines.append(f"Tensor Mean: {input_stats['mean']:.4f}")
        lines.append(f"Tensor Std:  {input_stats['std']:.4f}")
        if (abs(input_stats['mean']) > 0.8 or input_stats['std'] < 0.2):
            lines.append("WARNING: Input Distribution strongly deviates "
                         "from ImageNet normalization norms.")
        else:
            lines.append("Input Distribution matches expected ranges.")
            
    return "\n".join(lines) + "\n"

def save_report(content: str, path: str = "diagnostic_report.txt"):
    """
    Saves the diagnostic report content to a file.
    
    Args:
        content (str): Text content to write.
        path (str): File path to save to. Defaults to "diagnostic_report.txt".
    """
    with open(path, "w") as f:
        f.write(content)
