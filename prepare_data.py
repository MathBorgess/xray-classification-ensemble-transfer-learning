"""
Quick start script to download and prepare the Chest X-Ray dataset

Authors:
    Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)
    Matheus Borges Figueirôa (mbf3@cin.ufpe.br)
    CIn - UFPE
"""

import os
import argparse
from pathlib import Path


def create_data_structure():
    """Create the expected data directory structure"""
    base_dir = Path("data/raw/chest_xray")

    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']

    for split in splits:
        for cls in classes:
            dir_path = base_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")


def print_instructions():
    """Print instructions for downloading the dataset"""
    print("\n" + "=" * 80)
    print("Chest X-Ray Dataset Setup")
    print("=" * 80)
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("\n1. Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("\n2. Extract the downloaded archive")
    print("\n3. Move the contents to: data/raw/chest_xray/")
    print("\n   Expected structure:")
    print("   data/raw/chest_xray/")
    print("   ├── train/")
    print("   │   ├── NORMAL/")
    print("   │   └── PNEUMONIA/")
    print("   ├── val/")
    print("   │   ├── NORMAL/")
    print("   │   └── PNEUMONIA/")
    print("   └── test/")
    print("       ├── NORMAL/")
    print("       └── PNEUMONIA/")
    print("\n4. Run data exploration notebook:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print("\n" + "=" * 80)


def check_dataset():
    """Check if dataset exists and print statistics"""
    base_dir = Path("data/raw/chest_xray")

    if not base_dir.exists():
        print("Dataset directory not found!")
        return False

    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)

    total_images = 0
    for split in ['train', 'val', 'test']:
        split_path = base_dir / split
        if not split_path.exists():
            continue

        normal_path = split_path / 'NORMAL'
        pneumonia_path = split_path / 'PNEUMONIA'

        normal_count = len(list(normal_path.glob('*.jpeg'))
                           ) if normal_path.exists() else 0
        pneumonia_count = len(list(pneumonia_path.glob(
            '*.jpeg'))) if pneumonia_path.exists() else 0
        split_total = normal_count + pneumonia_count

        print(f"\n{split.upper()}:")
        print(f"  Normal: {normal_count}")
        print(f"  Pneumonia: {pneumonia_count}")
        print(f"  Total: {split_total}")

        total_images += split_total

    print(f"\nTotal images: {total_images}")
    print("=" * 80 + "\n")

    return total_images > 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare Chest X-Ray dataset')
    parser.add_argument('--check', action='store_true',
                        help='Check if dataset exists and print statistics')
    args = parser.parse_args()

    if args.check:
        if check_dataset():
            print("✓ Dataset found and ready to use!")
        else:
            print("✗ Dataset not found. Please follow the download instructions.")
            create_data_structure()
            print_instructions()
    else:
        create_data_structure()
        print_instructions()


if __name__ == '__main__':
    main()
