"""
Main training script for Chest X-Ray Classification

Authors:
    Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)
    Matheus Borges Figueirôa (mbf3@cin.ufpe.br)
    CIn - UFPE
"""

import os
import argparse
import torch
from pathlib import Path

from src.utils import set_seed, load_config, create_directories, get_device, save_checkpoint
from src.data_loader import get_dataloaders
from src.models import create_model
from src.trainer import train_model
from src.trainer import evaluate
from src.evaluation import compute_confusion_matrix, plot_confusion_matrix, plot_roc_curve


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Chest X-Ray Classification Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'densenet121'],
                        help='Model architecture')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Path to output directory')
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Create directories
    create_directories(config)

    # Get device
    device = get_device(config)

    print("=" * 80)
    print(f"Training {args.model} for Chest X-Ray Classification")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, args.data_dir)

    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(args.model, config, pretrained=True)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training stages
    training_config = config.get('training', {})

    # Stage 1: Baseline (frozen backbone)
    print("\n" + "=" * 80)
    print("STAGE 1: Baseline Training (Frozen Backbone)")
    print("=" * 80)

    model.freeze_backbone()
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, history_baseline = train_model(
        model, train_loader, val_loader, config, device, training_stage='baseline'
    )

    # Save baseline model
    save_path = Path(args.output_dir) / f"{args.model}_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nBaseline model saved to {save_path}")

    # Stage 2: Progressive Unfreezing - Stage 1
    if 'progressive_unfreezing' in training_config:
        print("\n" + "=" * 80)
        print("STAGE 2: Progressive Unfreezing - Stage 1")
        print("=" * 80)

        stage_1_config = training_config['progressive_unfreezing'].get(
            'stage_1', {})
        num_layers = stage_1_config.get('unfreeze_layers', 20)

        model.unfreeze_backbone(num_layers)
        print(f"Unfrozen last {num_layers} layers")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        model, history_stage1 = train_model(
            model, train_loader, val_loader, config, device, training_stage='stage_1'
        )

        # Save stage 1 model
        save_path = Path(args.output_dir) / f"{args.model}_stage1.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\nStage 1 model saved to {save_path}")

        # Stage 3: Progressive Unfreezing - Stage 2
        print("\n" + "=" * 80)
        print("STAGE 3: Progressive Unfreezing - Stage 2")
        print("=" * 80)

        stage_2_config = training_config['progressive_unfreezing'].get(
            'stage_2', {})
        num_layers = stage_2_config.get('unfreeze_layers', 50)

        model.unfreeze_backbone(num_layers)
        print(f"Unfrozen last {num_layers} layers")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        model, history_stage2 = train_model(
            model, train_loader, val_loader, config, device, training_stage='stage_2'
        )

        # Save final model
        save_path = Path(args.output_dir) / f"{args.model}_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\nFinal model saved to {save_path}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results_dir = Path(config.get('paths', {}).get('results', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{args.model}_test_results.txt", 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write("=" * 50 + "\n")
        f.write("\nTest Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")

    print(
        f"\nResults saved to {results_dir / f'{args.model}_test_results.txt'}")
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
