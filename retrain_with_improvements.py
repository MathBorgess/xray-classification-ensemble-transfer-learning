"""
Re-training Script with All Improvements

This script re-trains models using:
1. Cross-Validation (K=5)
2. Advanced Augmentation
3. Focal Loss
4. Threshold Optimization
5. Test-Time Augmentation

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from src.utils import load_config, set_seed, get_device
from src.cross_validation import run_cross_validation_experiment
from src.threshold_optimization import optimize_threshold_for_model
from src.tta import compare_with_without_tta
from src.data_loader import load_data_from_directory, get_dataloaders
from src.models import create_model


def main():
    parser = argparse.ArgumentParser(description='Re-train with improvements')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to train (default: all)')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation (use existing models)')
    parser.add_argument('--skip-threshold', action='store_true',
                        help='Skip threshold optimization')
    parser.add_argument('--skip-tta', action='store_true',
                        help='Skip TTA evaluation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: only EfficientNetB0')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    device = get_device(config)

    # Determine models to process
    models = args.models
    if args.quick and models is None:
        models = ['efficientnet_b0']
        print("🚀 Quick mode: Processing only EfficientNetB0\n")
    elif models is None:
        models = ['efficientnet_b0', 'resnet50', 'densenet121']

    print("="*80)
    print("RE-TRAINING WITH IMPROVEMENTS")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Create results directory
    results_dir = Path('results/improved_training')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'models': models,
        'improvements_applied': [],
        'results': {}
    }

    # =========================================================================
    # PHASE 1: CROSS-VALIDATION
    # =========================================================================

    if not args.skip_cv:
        print("\n" + "#"*80)
        print("# PHASE 1: CROSS-VALIDATION (K=5)")
        print("#"*80)

        cv_results = run_cross_validation_experiment(args.config, models)

        results_summary['improvements_applied'].append('cross_validation')
        results_summary['cross_validation'] = {}

        for model_name, summary in cv_results.items():
            results_summary['cross_validation'][model_name] = {
                'mean_metrics': summary['mean_metrics'],
                'std_metrics': summary['std_metrics'],
                'ci_95_metrics': summary['ci_95_metrics']
            }

        print("\n✅ Cross-Validation Complete!")
    else:
        print("\n⏭️  Skipping Cross-Validation")

    # =========================================================================
    # PHASE 2: THRESHOLD OPTIMIZATION
    # =========================================================================

    if not args.skip_threshold:
        print("\n" + "#"*80)
        print("# PHASE 2: THRESHOLD OPTIMIZATION")
        print("#"*80)

        # Load validation data
        data_dir = config.get('data', {}).get(
            'data_dir', 'data/raw/chest_xray')
        _, val_loader, _ = get_dataloaders(config, use_existing_splits=True)

        results_summary['improvements_applied'].append(
            'threshold_optimization')
        results_summary['threshold_optimization'] = {}

        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Optimizing thresholds: {model_name}")
            print(f"{'='*60}")

            # Load best model from CV (fold 1 for consistency)
            model_path = f'models/cv_models/{model_name}_fold1.pth'

            if not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                print("   Using original trained model...")
                model_path = f'models/{model_name}_final.pth'

            if not os.path.exists(model_path):
                print(f"❌ No model found for {model_name}, skipping...")
                continue

            # Load model
            model = create_model(model_name, config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Optimize thresholds
            threshold_results = optimize_threshold_for_model(
                model, val_loader, device,
                methods=['youden', 'f1', 'balanced', 'target_specificity'],
                save_dir=str(results_dir / f'threshold_{model_name}')
            )

            results_summary['threshold_optimization'][model_name] = {}

            for method, result in threshold_results.items():
                results_summary['threshold_optimization'][model_name][method] = {
                    'threshold': float(result['threshold']),
                    'sensitivity': float(result['metrics']['sensitivity']),
                    'specificity': float(result['metrics']['specificity'])
                }

            # Save thresholds
            threshold_file = results_dir / \
                f'{model_name}_optimal_thresholds.json'
            with open(threshold_file, 'w') as f:
                json.dump(threshold_results, f, indent=2, default=float)

            print(f"\n✅ Thresholds saved to: {threshold_file}")

        print("\n✅ Threshold Optimization Complete!")
    else:
        print("\n⏭️  Skipping Threshold Optimization")

    # =========================================================================
    # PHASE 3: TEST-TIME AUGMENTATION EVALUATION
    # =========================================================================

    if not args.skip_tta:
        print("\n" + "#"*80)
        print("# PHASE 3: TEST-TIME AUGMENTATION EVALUATION")
        print("#"*80)

        # Load test data
        _, _, test_loader = get_dataloaders(config, use_existing_splits=True)

        results_summary['improvements_applied'].append(
            'test_time_augmentation')
        results_summary['tta_evaluation'] = {}

        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Evaluating TTA: {model_name}")
            print(f"{'='*60}")

            # Load model
            model_path = f'models/cv_models/{model_name}_fold1.pth'

            if not os.path.exists(model_path):
                model_path = f'models/{model_name}_final.pth'

            if not os.path.exists(model_path):
                print(f"❌ No model found for {model_name}, skipping...")
                continue

            model = create_model(model_name, config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            # Compare with/without TTA
            tta_comparison = compare_with_without_tta(
                model, test_loader, config, device, n_augmentations=5
            )

            results_summary['tta_evaluation'][model_name] = tta_comparison

        print("\n✅ TTA Evaluation Complete!")
    else:
        print("\n⏭️  Skipping TTA Evaluation")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "#"*80)
    print("# FINAL SUMMARY")
    print("#"*80)

    print("\n📊 Improvements Applied:")
    for improvement in results_summary['improvements_applied']:
        print(f"  ✅ {improvement.replace('_', ' ').title()}")

    # Save complete results
    summary_file = results_dir / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=float)

    print(f"\n💾 Complete results saved to: {summary_file}")

    # Print comparison table
    if 'cross_validation' in results_summary and results_summary['cross_validation']:
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy':<25} {'Specificity':<25}")
        print("-"*80)

        for model_name in models:
            if model_name in results_summary['cross_validation']:
                cv_results = results_summary['cross_validation'][model_name]

                acc_mean = cv_results['mean_metrics']['accuracy']
                acc_ci = cv_results['ci_95_metrics']['accuracy']

                spec_mean = cv_results['mean_metrics']['specificity']
                spec_ci = cv_results['ci_95_metrics']['specificity']

                print(f"{model_name:<20} "
                      f"{acc_mean:.4f} [{acc_ci[0]:.4f},{acc_ci[1]:.4f}]  "
                      f"{spec_mean:.4f} [{spec_ci[0]:.4f},{spec_ci[1]:.4f}]")

    if 'threshold_optimization' in results_summary and results_summary['threshold_optimization']:
        print("\n" + "="*80)
        print("OPTIMAL THRESHOLDS (Target Specificity ≥ 60%)")
        print("="*80)
        print(
            f"{'Model':<20} {'Threshold':<15} {'Sensitivity':<15} {'Specificity':<15}")
        print("-"*80)

        for model_name in models:
            if model_name in results_summary['threshold_optimization']:
                thresh_results = results_summary['threshold_optimization'][model_name]

                if 'target_specificity' in thresh_results:
                    result = thresh_results['target_specificity']
                    print(f"{model_name:<20} "
                          f"{result['threshold']:<15.4f} "
                          f"{result['sensitivity']:<15.4f} "
                          f"{result['specificity']:<15.4f}")

    print("\n" + "="*80)
    print("✅ ALL PHASES COMPLETE!")
    print("="*80)
    print(f"\nResults directory: {results_dir}")
    print(f"Summary file: {summary_file}")
    print("\n🎉 Ready for ensemble implementation!\n")


if __name__ == '__main__':
    main()
