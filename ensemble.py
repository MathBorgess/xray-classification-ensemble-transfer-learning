"""
Script to create and evaluate ensemble models

Authors:
    Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)
    Matheus Borges Figueirôa (mbf3@cin.ufpe.br)
    CIn - UFPE
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from src.utils import set_seed, load_config, get_device
from src.data_loader import get_dataloaders
from src.models import create_model, create_ensemble
from src.trainer import evaluate
from src.evaluation import compute_confusion_matrix, plot_confusion_matrix, plot_roc_curve, compare_models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create and Evaluate Ensemble Models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Path to output directory')
    return parser.parse_args()


def load_trained_model(model_name: str, model_path: str, config: dict, device: torch.device):
    """Load a trained model"""
    model = create_model(model_name, config, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    """Main ensemble evaluation function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device(config)

    print("=" * 80)
    print("Ensemble Model Evaluation")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, args.data_dir)

    # Load individual models
    model_dir = Path(args.model_dir)
    model_architectures = ['efficientnet_b0', 'resnet50', 'densenet121']

    models = []
    val_metrics_list = []
    individual_results = {}

    criterion = torch.nn.CrossEntropyLoss()

    print("\nLoading and evaluating individual models...")
    for arch in model_architectures:
        model_path = model_dir / f"{arch}_final.pth"

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        print(f"\n{arch}:")
        print("-" * 50)

        # Load model
        model = load_trained_model(arch, model_path, config, device)
        models.append(model)

        # Evaluate on validation set (for ensemble weights)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_metrics_list.append(val_metrics)

        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, criterion, device)
        individual_results[arch] = test_metrics

        print(f"Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if len(models) < 2:
        print("\nError: Need at least 2 models for ensemble. Exiting.")
        return

    # Create ensemble models
    print("\n" + "=" * 80)
    print("Creating Ensemble Models")
    print("=" * 80)

    ensemble_results = {}

    # Simple voting ensemble
    print("\nSimple Voting Ensemble:")
    print("-" * 50)
    simple_ensemble = create_ensemble(
        models,
        val_metrics=None,
        config={'ensemble': {'methods': ['simple_voting']}}
    )
    simple_ensemble = simple_ensemble.to(device)

    test_metrics = evaluate(simple_ensemble, test_loader, criterion, device)
    ensemble_results['simple_voting'] = test_metrics

    print(f"Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Weighted voting ensemble
    print("\nWeighted Voting Ensemble (based on validation AUC):")
    print("-" * 50)
    weighted_ensemble = create_ensemble(
        models,
        val_metrics=val_metrics_list,
        config=config
    )
    weighted_ensemble = weighted_ensemble.to(device)

    # Print weights
    print(f"Weights: {weighted_ensemble.weights.cpu().numpy()}")

    test_metrics = evaluate(weighted_ensemble, test_loader, criterion, device)
    ensemble_results['weighted_voting'] = test_metrics

    print(f"Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Compare all models
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    all_results = {**individual_results, **ensemble_results}

    print("\nComparison Table:")
    print("-" * 100)
    print(f"{'Model':<30} {'Accuracy':<12} {'AUC':<12} {'F1-Score':<12} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 100)

    for model_name, metrics in all_results.items():
        print(f"{model_name:<30} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['auc']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['sensitivity']:<12.4f} "
              f"{metrics['specificity']:<12.4f}")

    print("-" * 100)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    with open(output_dir / 'ensemble_comparison.txt', 'w') as f:
        f.write("Ensemble Model Comparison\n")
        f.write("=" * 100 + "\n\n")
        f.write(
            f"{'Model':<30} {'Accuracy':<12} {'AUC':<12} {'F1-Score':<12} {'Sensitivity':<12} {'Specificity':<12}\n")
        f.write("-" * 100 + "\n")

        for model_name, metrics in all_results.items():
            f.write(f"{model_name:<30} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['auc']:<12.4f} "
                    f"{metrics['f1_score']:<12.4f} "
                    f"{metrics['sensitivity']:<12.4f} "
                    f"{metrics['specificity']:<12.4f}\n")

    print(f"\nResults saved to {output_dir / 'ensemble_comparison.txt'}")

    # Plot comparisons
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    for metric in ['accuracy', 'auc', 'f1_score']:
        compare_models(
            all_results,
            metric=metric,
            save_path=figures_dir / f'comparison_{metric}.png'
        )
        print(
            f"Saved comparison plot: {figures_dir / f'comparison_{metric}.png'}")

    print("\nEnsemble evaluation completed successfully!")


if __name__ == '__main__':
    main()
