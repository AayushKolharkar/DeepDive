"""
Core package holding shared constants and business logic.
"""

DIAG_COLORS = {
    "dead":      "#f87171",  # Red   – dead / zero-variance filter
    "redundant": "#facc15",  # Yellow – near-duplicate of another filter
    "normal":    "#e4e4e7",  # Neutral white-grey
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
