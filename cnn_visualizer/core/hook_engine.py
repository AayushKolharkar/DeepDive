"""
Handles registration and execution of forward hooks for extracting features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms
from core import IMAGENET_MEAN, IMAGENET_STD

class HookEngine:
    """
    Manages hooks to extract activations from Conv2d layers.
    """
    def __init__(self, model: nn.Module, conv_layers: dict[str, nn.Module]):
        """
        Args:
            model (nn.Module): The loaded PyTorch model.
            conv_layers (dict): Dictionary mapping layer names to nn.Module layers.
        """
        self.model = model
        self.conv_layers = conv_layers
        self.activation = {}
        self.hook_handles = []
        self._summary_hook_handles = []
        self.layer_flow_data: dict[str, float] = {}
        self.input_stats = {"mean": 0.0, "std": 0.0, "hist": None}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _get_activation(self, name: str):
        """
        A robust register_forward_hook callback factory.
        
        Args:
            name (str): Name of the layer being hooked.
        """
        def hook(model, input, output):
            self.activation[name] = output.detach().cpu()
        return hook

    def _get_summary_hook(self, name: str):
        """
        Lightweight hook that records only the mean absolute activation for a layer.
        Does NOT store the full tensor — scalar only. Safe to register on all layers
        simultaneously without memory overhead.
        """
        def hook(module, input, output):
            self.layer_flow_data[name] = output.detach().abs().mean().item()
        return hook

    def extract_features(self, image_path: str, target_names: str | list[str]) -> tuple[torch.Tensor | dict[str, torch.Tensor] | None, dict, dict]:
        """
        Registers a hook, runs an image forward, and fetches the feature maps.
        
        Args:
            image_path (str): Path to the image file.
            target_names (str | list[str]): A single layer name or a list of layer names.
            
        Returns:
            tuple: Activation tensor(s), input statistics dict, and layer flow data dict.
        """
        if self.model is None:
            return None, self.input_stats, dict(self.layer_flow_data)

        # Remove previous hooks
        for handle in self.hook_handles:
            handle.remove()
        for handle in self._summary_hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._summary_hook_handles.clear()
        self.activation.clear()
        self.layer_flow_data.clear()

        # Register new forward hook(s)
        targets = [target_names] if isinstance(target_names, str) else target_names
        for t in targets:
            if t in self.conv_layers:
                selected_layer = self.conv_layers[t]
                handle = selected_layer.register_forward_hook(self._get_activation(t))
                self.hook_handles.append(handle)

        for name, layer in self.conv_layers.items():
            handle = layer.register_forward_hook(self._get_summary_hook(name))
            self._summary_hook_handles.append(handle)

        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
            
        try:
            input_tensor = self.transform(image).unsqueeze(0)

            # Diagnostic: Input Sanity Check
            self.input_stats["mean"] = input_tensor.mean().item()
            self.input_stats["std"] = input_tensor.std().item()
        except BaseException as e:
            print(f"Error transforming frame for extraction: {e}")
            return None, None, dict(self.layer_flow_data)

        # Forward pass
        with torch.no_grad():
            self.model(input_tensor)

        # Cleanup: remove hooks
        for handle in self.hook_handles:
            handle.remove()
        for handle in self._summary_hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._summary_hook_handles.clear()

        if isinstance(target_names, str):
            res = self.activation.get(target_names, None)
            return res, self.input_stats, dict(self.layer_flow_data)
        
        return self.activation, self.input_stats, dict(self.layer_flow_data)

    def extract_features_from_frame(self, frame_bgr, target_names: str | list[str]) -> tuple[torch.Tensor | dict[str, torch.Tensor] | None, dict, dict]:
        """
        Standard forward pass from an OpenCV Live Camera frame.
        
        Args:
            frame_bgr (np.ndarray): OpenCV BGR frame.
            target_names (str | list[str]): A single layer name or a list of layer names.
            
        Returns:
            tuple: Activation tensor(s), input statistics dict, and layer flow data dict.
        """
        if not self.model or not self.conv_layers:
            return None, None, dict(self.layer_flow_data)

        for handle in self.hook_handles:
            handle.remove()
        for handle in self._summary_hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._summary_hook_handles.clear()
        self.activation.clear()
        self.layer_flow_data.clear()

        # Hook registration
        target_list = [target_names] if isinstance(target_names, str) else target_names
        for name in target_list:
            if name in self.conv_layers:
                layer = self.conv_layers[name]
                handle = layer.register_forward_hook(self._get_activation(name))
                self.hook_handles.append(handle)

        for name, layer in self.conv_layers.items():
            handle = layer.register_forward_hook(self._get_summary_hook(name))
            self._summary_hook_handles.append(handle)

        # Convert the OpenCV BGR frame -> RGB -> PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        try:
            input_tensor = self.transform(img_pil).unsqueeze(0)

            self.input_stats["mean"] = input_tensor.mean().item()
            self.input_stats["std"] = input_tensor.std().item()
        except BaseException as e:
            print(f"Error transforming frame for extraction: {e}")
            return None, None, dict(self.layer_flow_data)

        with torch.no_grad():
            self.model(input_tensor)

        for handle in self.hook_handles:
            handle.remove()
        for handle in self._summary_hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._summary_hook_handles.clear()

        if isinstance(target_names, str):
            return self.activation.get(target_names), self.input_stats, dict(self.layer_flow_data)
            
        return self.activation, self.input_stats, dict(self.layer_flow_data)
