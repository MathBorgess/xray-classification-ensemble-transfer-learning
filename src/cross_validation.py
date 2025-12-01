"""
K-Fold Cross-Validation for robust model evaluation

This module implements stratified K-Fold cross-validation to obtain
robust performance metrics with confidence intervals.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm
from pathlib import Path

from src.models import create_model
from src.trainer import train_model, evaluate
from src.utils import get_device, set_seed, load_config
from src.data_loader import ChestXRayDataset, get_transforms, get_augmentation, calculate_class_weights
from torch.utils.data import DataLoader


class CrossValidator:
    """
    Implements Stratified K-Fold Cross-Validation
    """

    def __init__(self, config: Dict[str, Any], n_splits: int = 5):
        """
        Args:
            config: Configuration dictionary
            n_splits: Number of folds (default: 5)
        """
        self.config = config
        self.n_splits = n_splits
        self.device = get_device(config)

    def split_data(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified K-fold splits

        Args:
            image_paths: List of image paths
            labels: List of labels

        Returns:
            List of (train_indices, val_indices) tuples
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.config.get('seed', 42)
        )

        splits = []
        indices = np.arange(len(image_paths))
        for train_idx, val_idx in skf.split(indices, labels):
            splits.append((train_idx.tolist(), val_idx.tolist()))

        return splits

    def train_fold(
        self,
        fold_num: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str
    ) -> Tuple[nn.Module, Dict[str, float], Dict[str, list]]:
        """
        Train a single fold

        Args:
            fold_num: Fold number
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of model architecture

        Returns:
            Tuple of (trained_model, val_metrics, history)
        """
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num + 1}/{self.n_splits}: {model_name}")
        print(f"{'='*60}")

        # Create fresh model
        set_seed(self.config.get('seed', 42) + fold_num)
        model = create_model(model_name, self.config)
        model = model.to(self.device)

        # Training stages
        stages = ['baseline', 'stage_1', 'stage_2']
        history_combined = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }

        for stage in stages:
            print(f"\n--- Training Stage: {stage} ---")

            # Unfreeze layers for progressive stages
            if stage == 'stage_1':
                self._unfreeze_layers(model, num_layers=20)
            elif stage == 'stage_2':
                self._unfreeze_layers(model, num_layers=50)

            model, history = train_model(
                model, train_loader, val_loader,
                self.config, self.device, stage
            )

            # Combine history
            for key in history_combined.keys():
                if key in history:
                    history_combined[key].extend(history[key])

        # Final evaluation
        criterion = nn.CrossEntropyLoss()
        val_metrics = evaluate(model, val_loader, criterion, self.device)

        return model, val_metrics, history_combined

    def _unfreeze_layers(self, model: nn.Module, num_layers: int):
        """Unfreeze last N layers"""
        params = list(model.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True

    def cross_validate(
        self,
        model_name: str,
        image_paths: List[str],
        labels: List[int],
        save_dir: str = 'models/cv_models'
    ) -> Dict[str, Any]:
        """
        Perform complete cross-validation

        Args:
            model_name: Name of model architecture
            image_paths: List of image paths
            labels: List of labels
            save_dir: Directory to save fold models

        Returns:
            Dictionary with cross-validation results
        """
        os.makedirs(save_dir, exist_ok=True)

        splits = self.split_data(image_paths, labels)

        fold_results = []
        all_metrics = {
            'accuracy': [],
            'auc': [],
            'f1_score': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'recall': []
        }

        for fold_num, (train_idx, val_idx) in enumerate(splits):
            
            fold_model_path = os.path.join(
                save_dir, f'{model_name}_fold{fold_num + 1}.pth'
            )

            #codigo adicional para identificar checkpoints
            if os.path.exists(fold_model_path):
                print(
                    f"\n✅ FOLD {fold_num + 1}/{self.n_splits} ({model_name}) já concluído. Pulando treinamento.")
                continue
            #fim do codigo adicional
            
            # Create data loaders for this fold
            train_paths = [image_paths[i] for i in train_idx]
            train_labels_fold = [labels[i] for i in train_idx]
            val_paths = [image_paths[i] for i in val_idx]
            val_labels_fold = [labels[i] for i in val_idx]

            # Get transforms and augmentation
            transform = get_transforms(self.config, train=True)
            augmentation = get_augmentation(self.config)

            train_dataset = ChestXRayDataset(
                train_paths, train_labels_fold,
                transform=transform, augmentation=augmentation
            )

            val_transform = get_transforms(self.config, train=False)
            val_dataset = ChestXRayDataset(
                val_paths, val_labels_fold,
                transform=val_transform, augmentation=None
            )

            batch_size = self.config.get('data', {}).get('batch_size', 32)
            num_workers = self.config.get('data', {}).get('num_workers', 4)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True
            )

            print(
                f"\nFold {fold_num + 1} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

            # Train fold
            model, val_metrics, history = self.train_fold(
                fold_num, train_loader, val_loader, model_name
            )

            # Save fold model
            torch.save(model.state_dict(), fold_model_path)

            # Collect metrics
            fold_results.append({
                'fold': fold_num + 1,
                'metrics': val_metrics,
                'model_path': fold_model_path
            })

            for key in all_metrics.keys():
                if key in val_metrics:
                    all_metrics[key].append(val_metrics[key])

            print(f"\nFold {fold_num + 1} Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Calculate statistics
        summary = {
            'model_name': model_name,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'mean_metrics': {},
            'std_metrics': {},
            'ci_95_metrics': {}
        }

        for key, values in all_metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                # 95% confidence interval
                ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))
                ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))

                summary['mean_metrics'][key] = mean_val
                summary['std_metrics'][key] = std_val
                summary['ci_95_metrics'][key] = (ci_lower, ci_upper)

        # Save summary
        summary_path = os.path.join(save_dir, f'{model_name}_cv_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types
            summary_serializable = {
                'model_name': summary['model_name'],
                'n_splits': summary['n_splits'],
                'mean_metrics': {k: float(v) for k, v in summary['mean_metrics'].items()},
                'std_metrics': {k: float(v) for k, v in summary['std_metrics'].items()},
                'ci_95_metrics': {k: [float(v[0]), float(v[1])] for k, v in summary['ci_95_metrics'].items()}
            }
            json.dump(summary_serializable, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY: {model_name}")
        print(f"{'='*60}")

        for key in all_metrics.keys():
            if key in summary['mean_metrics']:
                mean = summary['mean_metrics'][key]
                std = summary['std_metrics'][key]
                ci = summary['ci_95_metrics'][key]
                print(
                    f"{key:15s}: {mean:.4f} ± {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

        return summary


def run_cross_validation_experiment(
    config_path: str = 'configs/config.yaml',
    models: List[str] = None
):
    """
    Run complete cross-validation experiment for specified models

    Args:
        config_path: Path to configuration file
        models: List of model names to validate (default: all 3 models)
    """
    config = load_config(config_path)

    # Load ALL training data (we'll split it ourselves)
    from src.data_loader import load_data_from_directory

    data_dir = config.get('data', {}).get('data_dir', 'data/raw/chest_xray')
    train_dir = os.path.join(data_dir, 'train')

    print("Loading training data...")
    image_paths, labels = load_data_from_directory(train_dir, config)
    print(f"Total samples: {len(image_paths)}")
    print(f"Normal: {sum(1 for l in labels if l == 0)}")
    print(f"Pneumonia: {sum(1 for l in labels if l == 1)}")

    # Create cross-validator
    cv = CrossValidator(config, n_splits=5)

    # Default to all models
    if models is None:
        models = ['efficientnet_b0', 'resnet50', 'densenet121']

    results = {}

    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# STARTING CROSS-VALIDATION: {model_name.upper()}")
        print(f"{'#'*60}")

        summary = cv.cross_validate(model_name, image_paths, labels)
        results[model_name] = summary

    # Save combined results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    combined_path = results_dir / 'cross_validation_results.json'

    with open(combined_path, 'w') as f:
        results_serializable = {}
        for model_name, summary in results.items():
            results_serializable[model_name] = {
                'mean_metrics': {k: float(v) for k, v in summary['mean_metrics'].items()},
                'std_metrics': {k: float(v) for k, v in summary['std_metrics'].items()},
                'ci_95_metrics': {k: [float(v[0]), float(v[1])] for k, v in summary['ci_95_metrics'].items()}
            }
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'#'*60}")
    print("# CROSS-VALIDATION COMPLETE")
    print(f"# Results saved to: {combined_path}")
    print(f"{'#'*60}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON ACROSS MODELS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Accuracy':<20} {'AUC':<20} {'Specificity':<20}")
    print(f"{'-'*80}")

    for model_name, summary in results.items():
        acc = summary['mean_metrics'].get('accuracy', 0)
        acc_ci = summary['ci_95_metrics'].get('accuracy', (0, 0))
        auc = summary['mean_metrics'].get('auc', 0)
        auc_ci = summary['ci_95_metrics'].get('auc', (0, 0))
        spec = summary['mean_metrics'].get('specificity', 0)
        spec_ci = summary['ci_95_metrics'].get('specificity', (0, 0))

        print(f"{model_name:<20} {acc:.4f} [{acc_ci[0]:.4f},{acc_ci[1]:.4f}]  "
              f"{auc:.4f} [{auc_ci[0]:.4f},{auc_ci[1]:.4f}]  "
              f"{spec:.4f} [{spec_ci[0]:.4f},{spec_ci[1]:.4f}]")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run cross-validation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to validate (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with only EfficientNetB0')

    args = parser.parse_args()

    models = args.models
    if args.quick and models is None:
        models = ['efficientnet_b0']
        print("🚀 Quick mode: Running only EfficientNetB0")

    results = run_cross_validation_experiment(args.config, models)
