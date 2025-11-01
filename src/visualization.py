"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
):
    """
    Plot training history

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-',
                 label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-',
                 label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-',
                 label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-',
                 label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # AUC
    axes[2].plot(epochs, history['val_auc'], 'g-',
                 label='Val AUC', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Validation AUC')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray = None,
    class_names: List[str] = None,
    num_samples: int = 8,
    save_path: str = None
):
    """
    Plot sample images with labels

    Args:
        images: Array of images
        labels: True labels
        predictions: Predicted labels (optional)
        class_names: Names of classes
        num_samples: Number of samples to plot
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    num_cols = 4
    num_rows = (num_samples + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    axes = axes.flatten()

    for i in range(num_samples):
        if i >= len(images):
            break

        img = images[i]

        # Denormalize if needed
        if img.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)

        title = f"True: {class_names[labels[i]]}"
        if predictions is not None:
            color = 'green' if predictions[i] == labels[i] else 'red'
            title += f"\nPred: {class_names[predictions[i]]}"
            axes[i].set_title(title, color=color, fontweight='bold')
        else:
            axes[i].set_title(title)

        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str] = None,
    title: str = 'Class Distribution',
    save_path: str = None
):
    """
    Plot class distribution

    Args:
        labels: Array of labels
        class_names: Names of classes
        title: Plot title
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    unique, counts = np.unique(labels, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    bars = ax1.bar([class_names[i] for i in unique],
                   counts, color=['steelblue', 'coral'])
    ax1.set_ylabel('Count')
    ax1.set_title(title)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontweight='bold')

    # Pie chart
    colors = ['steelblue', 'coral']
    ax2.pie(counts, labels=[f"{class_names[i]}\n({counts[i]})" for i in range(len(unique))],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: str = None
):
    """
    Plot comparison of multiple metrics across models

    Args:
        results: Dictionary of model results
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    if metrics is None:
        metrics = ['accuracy', 'auc', 'f1_score', 'sensitivity', 'specificity']

    models = list(results.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        scores = [results[model].get(metric, 0) for model in models]

        bars = axes[i].bar(models, scores, color='steelblue', alpha=0.7)
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'Comparison - {metric.upper()}')
        axes[i].set_ylim([0, 1.1])
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
