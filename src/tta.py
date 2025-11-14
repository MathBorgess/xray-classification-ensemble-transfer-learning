"""
Test-Time Augmentation (TTA) for improved prediction robustness

Applies multiple augmentations at inference time and averages predictions
to reduce prediction variance and improve accuracy.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import cv2

from src.advanced_augmentation import get_test_time_augmentation


class TTAWrapper:
    """
    Test-Time Augmentation wrapper for models

    Applies multiple augmentations to each image and averages the predictions
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        n_augmentations: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Trained model
            config: Configuration dictionary
            n_augmentations: Number of augmentations to apply (default: 5)
            device: Device to use for inference
        """
        self.model = model
        self.config = config
        self.n_augmentations = n_augmentations
        self.device = device if device is not None else torch.device('cpu')

        # Get augmentation pipelines
        self.augmentations = get_test_time_augmentation(
            config, n_augmentations)

        # Set model to eval mode
        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        image: np.ndarray,
        return_all: bool = False
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Predict with test-time augmentation

        Args:
            image: Input image (numpy array, shape [H, W, C])
            return_all: If True, return all augmented predictions

        Returns:
            Tuple of (averaged_prediction, all_predictions)
            - averaged_prediction: shape [num_classes]
            - all_predictions: list of predictions (if return_all=True)
        """
        all_predictions = []

        with torch.no_grad():
            for aug in self.augmentations:
                # Apply augmentation
                augmented = aug(image=image)
                aug_image = augmented['image'].unsqueeze(0).to(self.device)

                # Get prediction
                outputs = self.model(aug_image)
                probs = torch.softmax(outputs, dim=1)

                all_predictions.append(probs.cpu().numpy()[0])

        # Average predictions
        all_predictions = np.array(all_predictions)
        avg_prediction = np.mean(all_predictions, axis=0)

        if return_all:
            return avg_prediction, all_predictions
        else:
            return avg_prediction, None

    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict a batch of images with TTA

        Args:
            images: List of images (each shape [H, W, C])
            batch_size: Batch size for processing

        Returns:
            Predictions array, shape [num_images, num_classes]
        """
        predictions = []

        for i in tqdm(range(0, len(images), batch_size), desc="TTA Prediction"):
            batch_images = images[i:i+batch_size]

            for img in batch_images:
                pred, _ = self.predict(img, return_all=False)
                predictions.append(pred)

        return np.array(predictions)

    def evaluate_with_tta(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model with TTA on a dataset

        Args:
            dataloader: DataLoader with evaluation data

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            confusion_matrix, precision_score, recall_score
        )

        all_predictions = []
        all_labels = []

        print(
            f"Evaluating with TTA ({self.n_augmentations} augmentations per image)...")

        for images, labels in tqdm(dataloader, desc="TTA Evaluation"):
            # Process each image in batch
            batch_predictions = []

            for img in images:
                # Convert tensor to numpy
                img_np = img.permute(1, 2, 0).cpu().numpy()

                # Denormalize if needed
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = (img_np * 255).astype(np.uint8)

                # Get TTA prediction
                pred, _ = self.predict(img_np, return_all=False)
                batch_predictions.append(pred)

            all_predictions.extend(batch_predictions)
            all_labels.extend(labels.cpu().numpy())

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Get predicted classes
        y_pred = np.argmax(all_predictions, axis=1)
        y_probs = all_predictions[:, 1]  # Probability of class 1

        # Calculate metrics
        accuracy = accuracy_score(all_labels, y_pred)
        auc = roc_auc_score(all_labels, y_probs)
        f1 = f1_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred)
        recall = recall_score(all_labels, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        return metrics


def compare_with_without_tta(
    model: nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    n_augmentations: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compare model performance with and without TTA

    Args:
        model: Trained model
        dataloader: DataLoader with test data
        config: Configuration dictionary
        device: Device to use
        n_augmentations: Number of augmentations for TTA

    Returns:
        Dictionary with 'without_tta' and 'with_tta' metrics
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score,
        confusion_matrix, precision_score, recall_score
    )

    print("\n" + "="*60)
    print("COMPARING: Standard Inference vs TTA")
    print("="*60)

    # 1. Standard inference (without TTA)
    print("\n1. Standard Inference (no augmentation)...")
    model.eval()

    all_predictions_std = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Standard"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_predictions_std.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions_std = np.array(all_predictions_std)
    all_labels = np.array(all_labels)

    y_pred_std = np.argmax(all_predictions_std, axis=1)
    y_probs_std = all_predictions_std[:, 1]

    # Calculate standard metrics
    metrics_std = {
        'accuracy': accuracy_score(all_labels, y_pred_std),
        'auc': roc_auc_score(all_labels, y_probs_std),
        'f1_score': f1_score(all_labels, y_pred_std),
        'precision': precision_score(all_labels, y_pred_std),
        'recall': recall_score(all_labels, y_pred_std)
    }

    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred_std).ravel()
    metrics_std['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics_std['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\nStandard Inference Results:")
    for key, value in metrics_std.items():
        print(f"  {key}: {value:.4f}")

    # 2. TTA inference
    print(f"\n2. Test-Time Augmentation ({n_augmentations} augmentations)...")
    tta_wrapper = TTAWrapper(model, config, n_augmentations, device)
    metrics_tta = tta_wrapper.evaluate_with_tta(dataloader)

    print("\nTTA Results:")
    for key, value in metrics_tta.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}" if isinstance(
                value, float) else f"  {key}: {value}")

    # 3. Comparison
    print("\n" + "="*60)
    print("IMPROVEMENT with TTA:")
    print("="*60)

    for key in ['accuracy', 'auc', 'f1_score', 'sensitivity', 'specificity']:
        if key in metrics_std and key in metrics_tta:
            std_val = metrics_std[key]
            tta_val = metrics_tta[key]
            improvement = tta_val - std_val
            improvement_pct = (improvement / std_val *
                               100) if std_val > 0 else 0

            symbol = "✅" if improvement > 0 else "➖"
            print(f"{symbol} {key:15s}: {std_val:.4f} → {tta_val:.4f} "
                  f"({improvement:+.4f}, {improvement_pct:+.2f}%)")

    return {
        'without_tta': metrics_std,
        'with_tta': metrics_tta
    }


if __name__ == '__main__':
    print("Test-Time Augmentation (TTA) Module")
    print("="*60)
    print("\nThis module provides:")
    print("  1. TTAWrapper - Apply TTA to any model")
    print("  2. compare_with_without_tta - Benchmark TTA improvement")
    print("\nUsage:")
    print("  from src.tta import TTAWrapper")
    print("  tta = TTAWrapper(model, config, n_augmentations=5)")
    print("  prediction = tta.predict(image)")
