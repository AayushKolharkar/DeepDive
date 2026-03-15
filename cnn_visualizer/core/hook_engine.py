"""
Handles registration and execution of forward hooks for extracting features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms

# FIX #14: was 'from core import ...' (absolute) which breaks when the
# process is launched from the repo root rather than cnn_visualizer/.
# Relative import is correct since hook_engine lives inside the core package.
from . import IMAGENET_MEAN, IMAGENET_STD


class HookEngine:
    """
    Manages hooks to extract activations from Conv2d layers.
    """

    def __init__(self, model: nn.Module, conv_layers: dict[str, nn.Module]):
        """
        Args:
            model:       The loaded PyTorch model.
            conv_layers: Dict mapping layer names to nn.Module layers.
        """
        self.model = model
        self.conv_layers = conv_layers
        self.activation: dict[str, torch.Tensor] = {}
        self.hook_handles: list = []
        self._summary_hook_handles: list = []
        self.layer_flow_data: dict[str, float] = {}
        self.input_stats: dict = {"mean": 0.0, "std": 0.0, "hist": None}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------ #
    #  Hook factories                                                      #
    # ------------------------------------------------------------------ #

    def _get_activation(self, name: str):
        """Forward hook that stores the full output tensor."""
        def hook(module, input, output):
            self.activation[name] = output.detach().cpu()
        return hook

    def _get_summary_hook(self, name: str):
        """
        Lightweight forward hook — stores only the mean absolute activation
        scalar. Safe to register on all layers simultaneously.
        """
        def hook(module, input, output):
            self.layer_flow_data[name] = output.detach().abs().mean().item()
        return hook

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _register_hooks(self, targets: list[str]):
        """Register full-tensor hooks for the given target layers."""
        for t in targets:
            if t in self.conv_layers:
                handle = self.conv_layers[t].register_forward_hook(
                    self._get_activation(t)
                )
                self.hook_handles.append(handle)

    def _register_summary_hooks(self):
        """Register lightweight summary hooks on every conv layer."""
        for name, layer in self.conv_layers.items():
            handle = layer.register_forward_hook(self._get_summary_hook(name))
            self._summary_hook_handles.append(handle)

    def _clear_hooks(self):
        """Remove all registered hooks and reset handle lists."""
        for h in self.hook_handles:
            h.remove()
        for h in self._summary_hook_handles:
            h.remove()
        self.hook_handles.clear()
        self._summary_hook_handles.clear()

    # ------------------------------------------------------------------ #
    #  Public extraction methods                                           #
    # ------------------------------------------------------------------ #

    def extract_features(
        self,
        image_path: str,
        target_names: str | list[str],
    ) -> tuple[torch.Tensor | dict | None, dict, dict]:
        """
        Runs a forward pass on an image file and returns feature maps.

        Args:
            image_path:   Path to image file.
            target_names: Layer name or list of layer names to capture.

        Returns:
            (tensor_or_dict, input_stats, layer_flow_data)
        """
        if self.model is None:
            return None, self.input_stats, {}

        self._clear_hooks()
        self.activation.clear()
        self.layer_flow_data.clear()

        targets = [target_names] if isinstance(target_names, str) else target_names
        self._register_hooks(targets)
        self._register_summary_hooks()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

        try:
            input_tensor = self.transform(image).unsqueeze(0)
            self.input_stats["mean"] = input_tensor.mean().item()
            self.input_stats["std"] = input_tensor.std().item()
        except BaseException as e:
            print(f"Error transforming image: {e}")
            self._clear_hooks()
            return None, self.input_stats, {}

        with torch.no_grad():
            self.model(input_tensor)

        self._clear_hooks()

        if isinstance(target_names, str):
            return (
                self.activation.get(target_names),
                self.input_stats,
                dict(self.layer_flow_data),
            )
        return self.activation, self.input_stats, dict(self.layer_flow_data)

    def extract_features_from_frame(
        self,
        frame_bgr,
        target_names: str | list[str],
    ) -> tuple[torch.Tensor | dict | None, dict, dict]:
        """
        Runs a forward pass on an OpenCV BGR frame.

        Args:
            frame_bgr:    OpenCV BGR ndarray.
            target_names: Layer name or list of layer names to capture.

        Returns:
            (tensor_or_dict, input_stats, layer_flow_data)
        """
        if not self.model or not self.conv_layers:
            return None, self.input_stats, {}

        self._clear_hooks()
        self.activation.clear()
        self.layer_flow_data.clear()

        targets = [target_names] if isinstance(target_names, str) else target_names
        self._register_hooks(targets)
        self._register_summary_hooks()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        try:
            input_tensor = self.transform(img_pil).unsqueeze(0)
            self.input_stats["mean"] = input_tensor.mean().item()
            self.input_stats["std"] = input_tensor.std().item()
        except BaseException as e:
            print(f"Error transforming frame: {e}")
            self._clear_hooks()
            return None, self.input_stats, {}

        with torch.no_grad():
            self.model(input_tensor)

        self._clear_hooks()

        if isinstance(target_names, str):
            return (
                self.activation.get(target_names),
                self.input_stats,
                dict(self.layer_flow_data),
            )
        return self.activation, self.input_stats, dict(self.layer_flow_data)