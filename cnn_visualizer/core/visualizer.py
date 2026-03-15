"""
Feature visualization via activation maximization.

Generates a synthetic image that maximally activates a specific channel
in a CNN layer. Uses gradient ascent on a random noise input with several
regularization tricks to produce interpretable, artifact-free patterns:

  - L2 regularization: prevents pixel values from diverging
  - Gradient normalization: stable step sizes regardless of layer depth
  - Random jitter: each step rolls the image ±8px, discouraging high-freq noise
  - Periodic Gaussian blur: every 20 steps, smooths the image slightly
  - Adam optimizer: adaptive learning rate, far better than vanilla SGD here

No UI dependencies — pure torch + PIL.
"""
from __future__ import annotations
import time
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

# ImageNet stats used by the pre-trained models
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Input image is in normalised space; these are the corresponding clamping limits
# (roughly ±3 std devs around each channel mean)
_INPUT_MIN = (torch.zeros(1, 3, 1, 1) - _MEAN) / _STD       # black in normalised space
_INPUT_MAX = (torch.ones(1, 3, 1, 1) - _MEAN) / _STD        # white in normalised space


def _to_device(*tensors, device: torch.device):
    return [t.to(device) for t in tensors]


def synthesize_channel_pattern(
    model:        nn.Module,
    layer_name:   str,
    channel_idx:  int,
    device:       torch.device | None = None,
    iterations:   int = 256,
    lr:           float = 0.05,
    l2_weight:    float = 1e-4,
    jitter:       int = 8,
    blur_every:   int = 20,
    on_progress:  Callable[[int, int, float], None] | None = None,
) -> Image.Image:
    """
    Generate a synthetic image that maximally activates channel_idx in the
    specified layer using gradient ascent.

    Args:
        model:       PyTorch model (eval mode, any device).
        layer_name:  Dotted module path, e.g. "features.14" or "layer3.1.conv2".
        channel_idx: Index of the channel to maximize.
        device:      torch.device to run on. Auto-detects CUDA if None.
        iterations:  Number of gradient ascent steps. 256 is a good default.
        lr:          Adam learning rate. 0.05 works well across architectures.
        l2_weight:   Regularization strength. Higher = blurrier but more stable.
        jitter:      Max pixel roll per axis per step. Reduces high-freq noise.
        blur_every:  Apply Gaussian blur every N steps.
        on_progress: Optional callback(step, total, eta_seconds) called every 5 steps.

    Returns:
        PIL.Image (RGB, 224×224) of the synthesized pattern.
    """
    # ── Device selection ──────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    # ── Hook to capture target activation ────────────────────────────────
    activation: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        activation["val"] = out

    try:
        target_module = model.get_submodule(layer_name)
    except AttributeError:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    handle = target_module.register_forward_hook(_hook)

    # ── Initialise input image ────────────────────────────────────────────
    # Small random noise in normalised input space
    img = torch.empty(1, 3, 224, 224, device=device).uniform_(-0.1, 0.1)
    img.requires_grad_(True)

    optimizer = torch.optim.Adam([img], lr=lr)

    # Clamp limits on device
    inp_min, inp_max = _INPUT_MIN.to(device), _INPUT_MAX.to(device)

    start_time = time.time()

    for step in range(1, iterations + 1):
        optimizer.zero_grad()

        # Random jitter: roll ±jitter pixels on H and W
        jx = int(torch.randint(-jitter, jitter + 1, (1,)).item())
        jy = int(torch.randint(-jitter, jitter + 1, (1,)).item())
        inp_jittered = torch.roll(img, shifts=(jy, jx), dims=(2, 3))

        # Forward pass
        model(inp_jittered)

        act = activation.get("val")
        if act is None:
            continue

        # Loss: maximise mean activation of target channel + L2 penalty
        if channel_idx >= act.shape[1]:
            break  # safety — layer may have fewer channels than expected
        channel_act = act[0, channel_idx]
        loss = -channel_act.mean() + l2_weight * (img ** 2).mean()

        loss.backward()

        # Normalise gradient (unit-norm per step for stable updates)
        with torch.no_grad():
            grad_norm = img.grad.norm() + 1e-8
            img.grad.div_(grad_norm)

        optimizer.step()

        # Clamp to valid normalised range
        with torch.no_grad():
            img.clamp_(inp_min, inp_max)

            # Periodic Gaussian blur (applied in pixel space, then back)
            if blur_every > 0 and step % blur_every == 0:
                img_pil = _tensor_to_pil(img.detach().cpu())
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
                img.copy_(_pil_to_tensor(img_pil).to(device))

        # Progress callback
        if on_progress and step % 5 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / step) * (iterations - step) if step > 0 else 0.0
            on_progress(step, iterations, eta)

    handle.remove()

    # Final progress
    if on_progress:
        on_progress(iterations, iterations, 0.0)

    return _tensor_to_pil(img.detach().cpu())


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, 3, H, W) normalised tensor to a PIL RGB image."""
    t = tensor.squeeze(0).clone()
    # Denormalise: pixel = (normalised * std) + mean
    mean = _MEAN.squeeze(0)
    std  = _STD.squeeze(0)
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image back to a (1, 3, H, W) normalised tensor."""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    mean = _MEAN.squeeze(0)
    std  = _STD.squeeze(0)
    for c in range(3):
        t[c] = (t[c] - mean[c]) / std[c]
    return t.unsqueeze(0)