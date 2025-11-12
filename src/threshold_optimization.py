"""
Threshold Optimization for better Specificity/Sensitivity balance

This module implements multiple methods to find optimal classification thresholds.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    method: str = 'youden',
    target_specificity: float = 0.60,
    target_sensitivity: float = 0.95
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels (0 or 1)
        y_probs: Predicted probabilities for class 1
        method: Optimization method
            - 'youden': Maximize Youden's J statistic (Sens + Spec - 1)
            - 'f1': Maximize F1 score
            - 'balanced': Balanced accuracy (Sens + Spec) / 2
            - 'target_specificity': Achieve target specificity
        target_specificity: Target specificity (for 'target_specificity' method)
        target_sensitivity: Minimum sensitivity threshold
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Calculate specificity
    specificity = 1 - fpr
    sensitivity = tpr
    
    best_threshold = 0.5
    best_metrics = {}
    
    if method == 'youden':
        # Maximize Youden's J = Sensitivity + Specificity - 1
        j_scores = sensitivity + specificity - 1
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        best_metrics = {
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'youden_j': j_scores[best_idx]
        }
        
    elif method == 'f1':
        # Maximize F1 score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        best_metrics = {
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'f1_score': f1_scores[best_idx]
        }
        
    elif method == 'balanced':
        # Maximize balanced accuracy
        balanced_acc = (sensitivity + specificity) / 2
        best_idx = np.argmax(balanced_acc)
        best_threshold = thresholds[best_idx]
        
        best_metrics = {
            'sensitivity': sensitivity[best_idx],
            'specificity': specificity[best_idx],
            'balanced_accuracy': balanced_acc[best_idx]
        }
        
    elif method == 'target_specificity':
        # Find threshold that achieves target specificity while maximizing sensitivity
        # Find indices where specificity >= target
        valid_indices = np.where(specificity >= target_specificity)[0]
        
        if len(valid_indices) > 0:
            # Among valid thresholds, choose one with highest sensitivity
            best_idx = valid_indices[np.argmax(sensitivity[valid_indices])]
            best_threshold = thresholds[best_idx]
            
            best_metrics = {
                'sensitivity': sensitivity[best_idx],
                'specificity': specificity[best_idx],
                'target_met': True
            }
        else:
            # If target can't be met, get closest
            best_idx = np.argmin(np.abs(specificity - target_specificity))
            best_threshold = thresholds[best_idx]
            
            best_metrics = {
                'sensitivity': sensitivity[best_idx],
                'specificity': specificity[best_idx],
                'target_met': False
            }
    
    # Calculate additional metrics
    y_pred = (y_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    best_metrics.update({
        'threshold': best_threshold,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
    })
    
    return best_threshold, best_metrics


def evaluate_with_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    threshold: float,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model with custom threshold
    
    Args:
        model: Trained model
        dataloader: Data loader
        threshold: Classification threshold
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Apply threshold
    y_pred = (all_probs >= threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    # AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'auc': auc_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def optimize_threshold_for_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    methods: list = None,
    save_dir: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Find optimal thresholds using multiple methods
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to use
        methods: List of optimization methods
        save_dir: Directory to save plots
        
    Returns:
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['youden', 'f1', 'balanced', 'target_specificity']
    
    # Get predictions
    model.eval()
    all_probs = []
    all_labels = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Optimize for each method
    results = {}
    
    for method in methods:
        print(f"\nOptimizing threshold using: {method}")
        threshold, metrics = find_optimal_threshold(
            all_labels, all_probs, method=method,
            target_specificity=0.60
        )
        
        results[method] = {
            'threshold': threshold,
            'metrics': metrics
        }
        
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    # Plot comparison
    if save_dir:
        plot_threshold_analysis(all_labels, all_probs, results, save_dir)
    
    return results


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    optimization_results: Dict[str, Dict],
    save_dir: str
):
    """
    Plot threshold analysis
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        optimization_results: Results from optimize_threshold_for_model
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. ROC Curve with optimal points
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    ax = axes[0, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Mark optimal points
    colors = {'youden': 'red', 'f1': 'blue', 'balanced': 'green', 'target_specificity': 'purple'}
    for method, result in optimization_results.items():
        sens = result['metrics']['sensitivity']
        spec = result['metrics']['specificity']
        fpr_point = 1 - spec
        ax.plot(fpr_point, sens, 'o', color=colors.get(method, 'black'),
                markersize=10, label=f'{method}: {result["threshold"]:.3f}')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve with Optimal Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Sensitivity vs Specificity across thresholds
    ax = axes[0, 1]
    specificity = 1 - fpr
    ax.plot(thresholds, tpr, label='Sensitivity', linewidth=2)
    ax.plot(thresholds, specificity, label='Specificity', linewidth=2)
    
    # Mark optimal points
    for method, result in optimization_results.items():
        thresh = result['threshold']
        ax.axvline(thresh, color=colors.get(method, 'black'),
                   linestyle='--', alpha=0.5, label=f'{method}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity & Specificity vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Comparison bar chart
    ax = axes[1, 0]
    methods = list(optimization_results.keys())
    sens_values = [optimization_results[m]['metrics']['sensitivity'] for m in methods]
    spec_values = [optimization_results[m]['metrics']['specificity'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, sens_values, width, label='Sensitivity', alpha=0.8)
    ax.bar(x + width/2, spec_values, width, label='Specificity', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity & Specificity by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for i, (s, sp) in enumerate(zip(sens_values, spec_values)):
        ax.text(i - width/2, s + 0.02, f'{s:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, sp + 0.02, f'{sp:.3f}', ha='center', fontsize=8)
    
    # 4. Confusion matrices
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create mini confusion matrix plots
    n_methods = len(methods)
    for idx, method in enumerate(methods):
        metrics = optimization_results[method]['metrics']
        
        # Create mini subplot
        mini_ax = fig.add_subplot(2, 2, 4, frameon=False)
        
        # Extract confusion matrix values
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        tp = metrics['true_positives']
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Print summary
        text = f"{method.upper()}:\n"
        text += f"Threshold: {metrics['threshold']:.4f}\n"
        text += f"Sens: {metrics['sensitivity']:.3f}, Spec: {metrics['specificity']:.3f}\n"
        text += f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}"
        
        y_pos = 0.9 - (idx * 0.22)
        ax.text(0.05, y_pos, text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plot saved to: {save_dir}/threshold_optimization.png")


if __name__ == '__main__':
    print("Threshold Optimization Module")
    print("Use this module by importing and calling optimize_threshold_for_model()")
