"""
Dataset and DataLoader for Chest X-Ray classification
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        augmentation=None
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0: Normal, 1: Pneumonia)
            transform: PyTorch transforms
            augmentation: Albumentations augmentation pipeline
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        # Apply augmentation
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def get_transforms(config: Dict[str, Any], train: bool = True) -> transforms.Compose:
    """
    Get image transforms based on configuration

    Args:
        config: Configuration dictionary
        train: Whether transforms are for training

    Returns:
        Transforms composition
    """
    data_config = config.get('data', {})
    image_size = tuple(data_config.get('image_size', [224, 224]))
    norm_config = data_config.get('normalization', {})
    mean = norm_config.get('mean', [0.485, 0.456, 0.406])
    std = norm_config.get('std', [0.229, 0.224, 0.225])

    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    return transforms.Compose(transform_list)


def get_augmentation(config: Dict[str, Any]) -> A.Compose:
    """
    Get Albumentations augmentation pipeline

    Args:
        config: Configuration dictionary

    Returns:
        Augmentation composition
    """
    data_config = config.get('data', {})
    aug_config = data_config.get('augmentation', {})
    image_size = tuple(data_config.get('image_size', [224, 224]))

    rotation_range = aug_config.get('rotation_range', 10)
    horizontal_flip = aug_config.get('horizontal_flip', True)
    brightness_range = aug_config.get('brightness_range', 0.1)
    zoom_range = aug_config.get('zoom_range', 0.1)

    augmentation = A.Compose([
        A.Rotate(limit=rotation_range, p=0.5),
        A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_range,
            contrast_limit=brightness_range,
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=zoom_range,
            rotate_limit=0,
            p=0.5
        ),
        A.Resize(height=image_size[0], width=image_size[1])
    ])

    return augmentation


def load_data_from_directory(
    data_dir: str,
    config: Dict[str, Any]
) -> Tuple[List[str], List[int]]:
    """
    Load image paths and labels from directory structure

    Expected structure:
        data_dir/
            NORMAL/
                image1.jpeg
                image2.jpeg
            PNEUMONIA/
                image1.jpeg
                image2.jpeg

    Args:
        data_dir: Path to data directory
        config: Configuration dictionary

    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []

    # Class mapping: 0 = Normal, 1 = Pneumonia
    class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}

    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)

    return image_paths, labels


def create_data_splits(
    image_paths: List[str],
    labels: List[int],
    config: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Create stratified train/val/test splits

    Args:
        image_paths: List of image paths
        labels: List of labels
        config: Configuration dictionary

    Returns:
        Tuple of (train_paths, val_paths, test_paths, train_labels, val_labels, test_labels)
    """
    data_config = config.get('data', {})
    train_split = data_config.get('train_split', 0.70)
    val_split = data_config.get('val_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    seed = config.get('seed', 42)

    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_split,
        stratify=labels,
        random_state=seed
    )

    # Second split: separate train and validation
    val_size_adjusted = val_split / (train_split + val_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=seed
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def get_dataloaders(
    config: Dict[str, Any],
    data_dir: str = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets

    Args:
        config: Configuration dictionary
        data_dir: Path to data directory (optional, uses config if not provided)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get('data', {})
    if data_dir is None:
        data_dir = data_config.get('data_dir', 'data/raw/chest_xray')

    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)

    # Load data from existing train/val/test folders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    print(f"Loading data from:")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    print(f"  Test: {test_dir}")

    train_paths, train_labels = load_data_from_directory(train_dir, config)
    val_paths, val_labels = load_data_from_directory(val_dir, config)
    test_paths, test_labels = load_data_from_directory(test_dir, config)

    # Get transforms and augmentation
    train_transform = get_transforms(config, train=True)
    val_test_transform = get_transforms(config, train=False)
    train_augmentation = get_augmentation(config)

    # Create datasets
    train_dataset = ChestXRayDataset(
        train_paths, train_labels,
        transform=train_transform,
        augmentation=train_augmentation
    )

    val_dataset = ChestXRayDataset(
        val_paths, val_labels,
        transform=val_test_transform,
        augmentation=None
    )

    test_dataset = ChestXRayDataset(
        test_paths, test_labels,
        transform=val_test_transform,
        augmentation=None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Calculate class distribution
    train_normal = sum(1 for label in train_labels if label == 0)
    train_pneumonia = sum(1 for label in train_labels if label == 1)
    print(
        f"\nTrain distribution - Normal: {train_normal}, Pneumonia: {train_pneumonia}")
    print(
        f"Imbalance ratio: {max(train_normal, train_pneumonia) / min(train_normal, train_pneumonia):.2f}:1")

    return train_loader, val_loader, test_loader


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets

    Args:
        labels: List of labels

    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)
