#!/usr/bin/env python3
"""
Quick Start Script for Pre-Ensemble Fixes

This script guides you through the critical fixes that must be
implemented before ensemble learning.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")


def print_step(num, title, description):
    """Print formatted step"""
    print(f"\n📋 STEP {num}: {title}")
    print(f"   {description}")
    print("-" * 70)


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")

    required = [
        'torch',
        'torchvision',
        'numpy',
        'sklearn',
        'albumentations',
        'matplotlib',
        'tqdm',
        'pyyaml'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def check_data():
    """Check if data is available"""
    print_header("Checking Data")

    data_dir = Path("data/raw/chest_xray")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease download the Chest X-Ray dataset and place it in:")
        print(f"   {data_dir.absolute()}")
        return False

    # Check subdirectories
    required_dirs = ['train', 'val', 'test']
    for subdir in required_dirs:
        path = data_dir / subdir
        if path.exists():
            # Count samples
            normal = len(list((path / 'NORMAL').glob('*.*'))
                         ) if (path / 'NORMAL').exists() else 0
            pneumonia = len(list((path / 'PNEUMONIA').glob('*.*'))
                            ) if (path / 'PNEUMONIA').exists() else 0
            total = normal + pneumonia
            print(
                f"✅ {subdir:10s}: {total:5d} samples (Normal: {normal}, Pneumonia: {pneumonia})")
        else:
            print(f"❌ {subdir:10s}: NOT FOUND")
            return False

    print("\n✅ Data structure is valid!")
    return True


def check_models():
    """Check if trained models exist"""
    print_header("Checking Trained Models")

    models_dir = Path("models")
    models = ['efficientnet_b0_final.pth',
              'resnet50_final.pth', 'densenet121_final.pth']

    found = []
    missing = []

    for model_file in models:
        path = models_dir / model_file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_file:30s} ({size_mb:.1f} MB)")
            found.append(model_file)
        else:
            print(f"❌ {model_file:30s} (NOT FOUND)")
            missing.append(model_file)

    if missing:
        print(f"\n⚠️  Missing models: {len(missing)}/{len(models)}")
        print("Models must be trained before proceeding with corrections.")
        return False

    print(f"\n✅ All {len(found)} models found!")
    return True


def create_directory_structure():
    """Create necessary directories"""
    print_header("Creating Directory Structure")

    directories = [
        'src',
        'models/cv_models',
        'results',
        'results/figures',
        'results/metrics',
        'results/logs',
        'scripts'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

    print("\n✅ Directory structure ready!")


def show_implementation_plan():
    """Show step-by-step implementation plan"""
    print_header("Pre-Ensemble Fixes: Implementation Plan")

    print("""
🎯 OBJECTIVE: Resolve critical gaps before ensemble implementation

📊 Current Problems:
   • Validation set too small (16 samples) → Unstable metrics
   • Specificity too low (12-48%)       → Too many false positives
   • No cross-validation                → Uncertain generalization
   • Limited augmentation               → May underfit
   • Imbalanced loss                    → Biased towards majority

✅ Solutions to Implement:
""")

    print_step(
        1,
        "Cross-Validation (5-Fold Stratified)",
        "Implement K-Fold CV to get robust metrics with confidence intervals"
    )
    print("   📄 File: src/cross_validation.py")
    print("   ⏱️  Time: ~2 days (includes training 5 folds per model)")
    print("   🎯 Output: Mean ± Std ± CI(95%) for all metrics")
    print("   💡 Run: python -m src.cross_validation")

    print_step(
        2,
        "Threshold Optimization",
        "Find optimal threshold to maximize specificity while maintaining sensitivity"
    )
    print("   📄 File: src/threshold_optimization.py")
    print("   ⏱️  Time: ~1 day")
    print("   🎯 Target: Specificity ≥ 60%")
    print("   💡 Run: python -m src.threshold_optimization")

    print_step(
        3,
        "Advanced Augmentation",
        "Add medical imaging-specific augmentations (CLAHE, elastic deformation)"
    )
    print("   📄 File: src/data_loader.py (update)")
    print("   ⏱️  Time: ~0.5 days")
    print("   🎯 Output: 10+ augmentation types")

    print_step(
        4,
        "Focal Loss Implementation",
        "Replace Cross-Entropy with Focal Loss to better handle imbalance"
    )
    print("   📄 File: src/losses.py")
    print("   ⏱️  Time: ~0.5 days")
    print("   🎯 Output: Better class balance")

    print_step(
        5,
        "Test-Time Augmentation (TTA)",
        "Apply augmentation at inference time to reduce variance"
    )
    print("   📄 File: src/tta.py")
    print("   ⏱️  Time: ~1 day")
    print("   🎯 Output: More stable predictions")

    print_step(
        6,
        "Validation & Consolidation",
        "Verify all fixes and generate consolidated report"
    )
    print("   ⏱️  Time: ~2 days")
    print("   🎯 Output: Ready for ensemble implementation")

    print("\n" + "="*70)
    print("📊 EXPECTED IMPROVEMENTS:")
    print("="*70)
    print("   Specificity:      12-48%  →  ≥60%")
    print("   Validation Size:  16      →  ~1000 (across folds)")
    print("   Confidence:       None    →  95% CI for all metrics")
    print("   Balanced Acc:     ~56%    →  ≥75%")
    print("   Robustness:       Low     →  High (with TTA)")

    print("\n" + "="*70)
    print("⏱️  TOTAL TIME ESTIMATE: 7-10 days")
    print("="*70)


def show_next_steps():
    """Show immediate next steps"""
    print_header("🚀 Next Steps")

    print("""
IMMEDIATE ACTIONS:

1️⃣  Review the detailed plan:
   📄 Open: PRE_ENSEMBLE_FIXES.md
   👀 Read: Complete implementation details for each fix

2️⃣  Set up the code structure:
   ✅ All directories created
   ✅ Dependencies checked
   ✅ Data verified

3️⃣  Start with Cross-Validation:
   📝 Copy the code from PRE_ENSEMBLE_FIXES.md
   📄 Create: src/cross_validation.py
   ▶️  Run: python -m src.cross_validation

4️⃣  Monitor progress:
   📊 Track metrics in results/
   📈 Compare before/after
   ✅ Verify improvements

5️⃣  After ALL fixes are complete:
   📄 Proceed to: IMPLEMENTATION_GUIDE.md
   🎯 Implement: Ensemble learning

⚠️  CRITICAL: Do NOT skip to ensemble before completing these fixes!
   The ensemble will be unreliable without a solid statistical foundation.

📚 Documentation Structure:
   • progress.md              - Overall project status
   • PRE_ENSEMBLE_FIXES.md    - Detailed solutions (START HERE)
   • IMPLEMENTATION_GUIDE.md  - Ensemble implementation (AFTER fixes)
   • QUICKSTART.md           - General project guide
""")


def main():
    """Main function"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║        Pre-Ensemble Fixes - Quick Start Guide                     ║")
    print("║                                                                    ║")
    print("║        Chest X-Ray Classification Project                         ║")
    print("║        Authors: Jéssica A. L. de Macêdo & Matheus Borges F.       ║")
    print("║                                                                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    # Run checks
    deps_ok = check_dependencies()
    data_ok = check_data()
    models_ok = check_models()

    # Create directories
    if deps_ok and data_ok:
        create_directory_structure()

    # Show plan
    show_implementation_plan()

    # Show next steps
    show_next_steps()

    # Final status
    print_header("System Status")
    print(
        f"Dependencies:     {'✅ READY' if deps_ok else '❌ MISSING PACKAGES'}")
    print(f"Data:             {'✅ READY' if data_ok else '❌ DATA NOT FOUND'}")
    print(
        f"Trained Models:   {'✅ READY' if models_ok else '⚠️  NEED TRAINING'}")

    if deps_ok and data_ok and models_ok:
        print("\n✅ ALL CHECKS PASSED - Ready to implement fixes!")
        print("🚀 Start with: Review PRE_ENSEMBLE_FIXES.md")
    elif deps_ok and data_ok and not models_ok:
        print("\n⚠️  Models not trained yet")
        print("🔧 Train models first: python train.py --model efficientnet_b0")
        print("   Then re-run this script")
    else:
        print("\n❌ SETUP INCOMPLETE - Please resolve issues above")
        print("📖 See README.md for setup instructions")

    print("\n" + "="*70)
    print("For questions or issues, refer to:")
    print("  • PRE_ENSEMBLE_FIXES.md  - Complete fix documentation")
    print("  • progress.md            - Project status and roadmap")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
