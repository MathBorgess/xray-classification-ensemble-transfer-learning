"""
Evaluation metrics and robustness testing
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes

    Returns:
        Confusion matrix
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = None
):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = None
):
    """
    Plot Precision-Recall curve

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        save_path: Path to save figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'auc',
    save_path: str = None
):
    """
    Compare multiple models on a metric

    Args:
        results: Dictionary of model results
        metric: Metric to compare
        save_path: Path to save figure
    """
    models = list(results.keys())
    scores = [results[model][metric] for model in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform paired t-test

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        alpha: Significance level

    Returns:
        Tuple of (t_statistic, p_value, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = p_value < alpha

    return t_stat, p_value, is_significant


def apply_gaussian_noise(
    images: torch.Tensor,
    sigma: float = 10.0
) -> torch.Tensor:
    """
    Apply Gaussian noise to images

    Args:
        images: Input images
        sigma: Standard deviation of noise

    Returns:
        Noisy images
    """
    noise = torch.randn_like(images) * (sigma / 255.0)
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


def apply_contrast_reduction(
    images: torch.Tensor,
    level: float = 0.5
) -> torch.Tensor:
    """
    Reduce contrast of images

    Args:
        images: Input images
        level: Contrast reduction level (0.5 = 50% reduction)

    Returns:
        Images with reduced contrast
    """
    mean = images.mean(dim=(2, 3), keepdim=True)
    contrast_reduced = mean + level * (images - mean)
    return torch.clamp(contrast_reduced, 0.0, 1.0)


def apply_rotation(
    images: torch.Tensor,
    angle: float
) -> torch.Tensor:
    """
    Apply rotation to images

    Args:
        images: Input images
        angle: Rotation angle in degrees

    Returns:
        Rotated images
    """
    from torchvision.transforms.functional import rotate
    return rotate(images, angle)


def test_robustness(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness under perturbations

    Args:
        model: PyTorch model
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to test on

    Returns:
        Dictionary of robustness metrics
    """
    from src.trainer import evaluate

    eval_config = config.get('evaluation', {})
    perturbations_config = eval_config.get('perturbations', {})

    results = {}
    criterion = torch.nn.CrossEntropyLoss()

    # Baseline (no perturbation)
    print("\nTesting baseline (no perturbation)...")
    baseline_metrics = evaluate(model, test_loader, criterion, device)
    results['baseline'] = baseline_metrics

    # Gaussian noise
    if 'gaussian_noise' in perturbations_config:
        sigmas = perturbations_config['gaussian_noise'].get('sigma', [10, 20])
        for sigma in sigmas:
            print(f"\nTesting Gaussian noise (sigma={sigma})...")
            # Create perturbed dataset
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_gaussian_noise(x, sigma)
            )
            results[f'gaussian_noise_sigma_{sigma}'] = perturbed_metrics

    # Contrast reduction
    if 'contrast_reduction' in perturbations_config:
        levels = perturbations_config['contrast_reduction'].get('levels', [
                                                                0.5, 0.7])
        for level in levels:
            print(f"\nTesting contrast reduction (level={level})...")
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_contrast_reduction(x, level)
            )
            results[f'contrast_reduction_{int(level*100)}'] = perturbed_metrics

    # Rotation
    if 'rotation' in perturbations_config:
        angles = perturbations_config['rotation'].get('angles', [5, 10])
        for angle in angles:
            print(f"\nTesting rotation (angle=±{angle}°)...")
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_rotation(x, angle)
            )
            results[f'rotation_{angle}deg'] = perturbed_metrics

    return results


def test_with_perturbation(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    perturbation_fn
) -> Dict[str, float]:
    """
    Test model with specific perturbation

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to test on
        criterion: Loss function
        perturbation_fn: Function to apply perturbation

    Returns:
        Dictionary of metrics
    """
    from src.trainer import evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply perturbation
            inputs = perturbation_fn(inputs)

            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.0
    f1 = f1_score(all_labels, all_preds, average='binary')

    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'f1_score': f1
    }
