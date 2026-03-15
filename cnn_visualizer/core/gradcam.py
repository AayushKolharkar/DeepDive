"""
Isolated Grad-CAM computation logic.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np

def compute_gradcam(model: nn.Module, conv_layers: dict[str, nn.Module], transform, frame_bgr: np.ndarray, target_layer_name: str) -> tuple[Image.Image | None, np.ndarray | None, dict]:
    """
    Calculates the Gradient-weighted Class Activation Mapping (Grad-CAM).
    Returns the original PIL image, the computed heatmap tensor overlay, and layer flow data.
    
    Args:
        model (nn.Module): The loaded PyTorch model.
        conv_layers (dict): Dictionary mapping string layer names to nn.Module layers.
        transform (Callable): Tortchvision transforms for the input image.
        frame_bgr (np.ndarray): OpenCV BGR frame array.
        target_layer_name (str): Name of the layer to compute Grad-CAM for.
        
    Returns:
        tuple[PIL.Image.Image, np.ndarray, dict]: The base Image, the Grad-CAM heatmap, and flow data.
    """
    layer_flow_data = {}
    if not model or target_layer_name not in conv_layers:
        return None, None, layer_flow_data

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    try:
        input_tensor = transform(img_pil).unsqueeze(0)
        input_tensor.requires_grad = True
    except BaseException as e:
        print(f"Error transforming frame for Grad-CAM: {e}")
        return None, None, layer_flow_data

    layer = conv_layers[target_layer_name]
    
    activation = {}
    gradients = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach().cpu()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach().cpu()
        return hook

    def _get_summary_hook(name):
        def hook(module, input, output):
            layer_flow_data[name] = output.detach().abs().mean().item()
        return hook

    # Register BOTH Forward and Backward hooks
    fw_handle = layer.register_forward_hook(get_activation(target_layer_name))
    bw_handle = layer.register_full_backward_hook(save_gradient(target_layer_name))

    summary_handles = []
    for name, m in conv_layers.items():
        summary_handles.append(m.register_forward_hook(_get_summary_hook(name)))

    # 1. Forward Pass with gradients enabled
    model.zero_grad()
    with torch.set_grad_enabled(True):
        output = model(input_tensor)

        # Find the predicted class (argmax)
        pred_class_idx = output.argmax(dim=1).item()

        # 2. Backward Pass from the winning class score
        score = output[0, pred_class_idx]
        score.backward()

    # Clean up hooks
    fw_handle.remove()
    bw_handle.remove()
    for sh in summary_handles:
        sh.remove()

    # Fetch captured data
    activations = activation[target_layer_name].squeeze(0)  # (C, H, W)
    grads = gradients[target_layer_name].squeeze(0)         # (C, H, W)

    # 3. Global Average Pooling on the gradients to get importance weights
    weights = torch.mean(grads, dim=(1, 2))  # (C,)

    # 4. Multiply activations by weights
    grad_cam_map = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
    for i, w in enumerate(weights):
        grad_cam_map += w * activations[i]

    # 5. Apply ReLU
    grad_cam_map = torch.nn.functional.relu(grad_cam_map)

    # 6. Normalize
    cam_min, cam_max = grad_cam_map.min(), grad_cam_map.max()
    if cam_max - cam_min > 1e-8:
        grad_cam_map = (grad_cam_map - cam_min) / (cam_max - cam_min)
    else:
        grad_cam_map = torch.zeros_like(grad_cam_map)

    return img_pil, grad_cam_map.numpy(), layer_flow_data
