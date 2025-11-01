"""
Interpretability tools using Grad-CAM
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM implementation for model interpretability
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            input_image: Input image tensor
            target_class: Target class for CAM (None for predicted class)

        Returns:
            CAM heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def visualize_cam(
        self,
        input_image: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Visualize CAM overlaid on image

        Args:
            input_image: Input image tensor
            cam: CAM heatmap
            alpha: Overlay transparency

        Returns:
            Visualization image
        """
        # Convert input image to numpy
        img = input_image.squeeze().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Apply colormap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        # Overlay
        visualization = alpha * heatmap + (1 - alpha) * img
        visualization = np.clip(visualization, 0, 1)

        return visualization


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """
    Get target layer for Grad-CAM based on model architecture

    Args:
        model: PyTorch model
        model_name: Name of model architecture

    Returns:
        Target layer
    """
    if 'efficientnet' in model_name.lower():
        # For EfficientNet models
        return model.backbone.blocks[-1]
    elif 'resnet' in model_name.lower():
        # For ResNet models
        return model.backbone.layer4[-1]
    elif 'densenet' in model_name.lower():
        # For DenseNet models
        return model.backbone.features.denseblock4
    else:
        # Default: last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError(f"Could not find suitable layer for {model_name}")


def visualize_multiple_samples(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    model_name: str,
    num_samples: int = 4,
    save_path: str = None
):
    """
    Visualize Grad-CAM for multiple samples

    Args:
        model: PyTorch model
        images: Input images
        labels: True labels
        predictions: Predicted labels
        model_name: Name of model architecture
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    # Get target layer
    target_layer = get_target_layer(model, model_name)

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    class_names = ['Normal', 'Pneumonia']

    for i in range(min(num_samples, len(images))):
        img = images[i:i+1]
        true_label = labels[i].item()
        pred_label = predictions[i].item()

        # Generate CAM
        cam = grad_cam.generate_cam(img, target_class=pred_label)

        # Visualize
        viz = grad_cam.visualize_cam(img, cam)

        # Original image
        original = img.squeeze().cpu().numpy()
        if original.shape[0] == 3:
            original = np.transpose(original, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original = std * original + mean
        original = np.clip(original, 0, 1)

        # Plot
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f'Original\nTrue: {class_names[true_label]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(viz)
        axes[i, 2].set_title(f'Overlay\nPred: {class_names[pred_label]}')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
