"""
Script to test model robustness

Authors:
    Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)
    Matheus Borges Figueirôa (mbf3@cin.ufpe.br)
    CIn - UFPE
"""

import argparse
import torch
from pathlib import Path

from src.utils import set_seed, load_config, get_device
from src.data_loader import get_dataloaders
from src.models import create_model
from src.evaluation import test_robustness


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Model Robustness')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'densenet121'],
                        help='Model architecture')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Path to output directory')
    return parser.parse_args()


def main():
    """Main robustness testing function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(config.get('seed', 42))

    # Get device
    device = get_device(config)

    print("=" * 80)
    print(f"Robustness Testing for {args.model}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, args.data_dir)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = create_model(args.model, config, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Test robustness
    print("\nTesting robustness under perturbations...")
    robustness_results = test_robustness(model, test_loader, config, device)

    # Display results
    print("\n" + "=" * 80)
    print("Robustness Test Results")
    print("=" * 80)

    print(f"\n{'Perturbation':<30} {'Accuracy':<12} {'AUC':<12} {'F1-Score':<12}")
    print("-" * 66)

    for perturbation, metrics in robustness_results.items():
        print(f"{perturbation:<30} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['auc']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")

    print("-" * 66)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f'{args.model}_robustness.txt', 'w') as f:
        f.write(f"Robustness Test Results for {args.model}\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"{'Perturbation':<30} {'Accuracy':<12} {'AUC':<12} {'F1-Score':<12}\n")
        f.write("-" * 66 + "\n")

        for perturbation, metrics in robustness_results.items():
            f.write(f"{perturbation:<30} "
                    f"{metrics['accuracy']:<12.4f} "
                    f"{metrics['auc']:<12.4f} "
                    f"{metrics['f1_score']:<12.4f}\n")

    print(f"\nResults saved to {output_dir / f'{args.model}_robustness.txt'}")
    print("\nRobustness testing completed successfully!")


if __name__ == '__main__':
    main()
