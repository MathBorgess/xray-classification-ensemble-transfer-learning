"""
Advanced Augmentation for Medical Imaging

Implements medical imaging-specific augmentations including elastic
deformation, CLAHE, and other domain-specific transformations.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any


def get_augmentation_advanced(config: Dict[str, Any], p: float = 0.8) -> A.Compose:
    """
    Get advanced augmentation pipeline with medical imaging-specific transforms

    Args:
        config: Configuration dictionary
        p: Probability of applying each augmentation

    Returns:
        Albumentations composition
    """
    img_size = config.get('data', {}).get('img_size', 224)

    augmentation = A.Compose([
        # Geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=0,
            p=p * 0.7
        ),

        A.HorizontalFlip(p=0.5),

        # Elastic deformation - simulates tissue deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=0,
            p=p * 0.3
        ),

        # Grid distortion - simulates X-ray projection variations
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=p * 0.3
        ),

        # Optical distortion
        A.OpticalDistortion(
            distort_limit=0.2,
            shift_limit=0.2,
            border_mode=0,
            p=p * 0.2
        ),

        # Intensity transformations
        # CLAHE - Contrast Limited Adaptive Histogram Equalization
        # Very important for X-ray images to enhance local contrast
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=p * 0.5
        ),

        # Random brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p * 0.6
        ),

        # Gamma correction - simulates different exposure levels
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=p * 0.4
        ),

        # Gaussian noise - simulates sensor noise
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=p * 0.3
        ),

        # Blur transformations
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=p * 0.2),

        # Sharpening
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=p * 0.3
        ),

        # CoarseDropout (Cutout) - Random rectangular regions dropped
        A.CoarseDropout(
            max_holes=8,
            max_height=int(img_size * 0.1),
            max_width=int(img_size * 0.1),
            min_holes=1,
            fill_value=0,
            p=p * 0.3
        ),

        # Normalization (always apply)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            p=1.0
        ),

        ToTensorV2()
    ])

    return augmentation


def get_augmentation_basic(config: Dict[str, Any]) -> A.Compose:
    """
    Get basic augmentation pipeline (original)

    Args:
        config: Configuration dictionary

    Returns:
        Albumentations composition
    """
    img_size = config.get('data', {}).get('img_size', 224)

    augmentation = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            p=1.0
        ),
        ToTensorV2()
    ])

    return augmentation


def get_test_time_augmentation(config: Dict[str, Any], n_augmentations: int = 5) -> list:
    """
    Get list of augmentation pipelines for Test-Time Augmentation (TTA)

    Args:
        config: Configuration dictionary
        n_augmentations: Number of augmentation variants

    Returns:
        List of augmentation pipelines
    """
    img_size = config.get('data', {}).get('img_size', 224)

    base_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            p=1.0
        ),
        ToTensorV2()
    ])

    augmentations = [base_transform]  # Original (no augmentation)

    # Add variations
    if n_augmentations >= 2:
        augmentations.append(A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 3:
        augmentations.append(A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 4:
        augmentations.append(A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 5:
        augmentations.append(A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 6:
        augmentations.append(A.Compose([
            A.HorizontalFlip(p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 7:
        augmentations.append(A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=-5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    if n_augmentations >= 8:
        augmentations.append(A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225], p=1.0),
            ToTensorV2()
        ]))

    return augmentations[:n_augmentations]


if __name__ == '__main__':
    # Example usage
    config = {'data': {'img_size': 224}}

    print("Advanced Augmentation Module")
    print("="*60)

    aug = get_augmentation_advanced(config)
    print(
        f"\nAdvanced augmentation pipeline created with {len(aug)} transforms")

    tta_augs = get_test_time_augmentation(config, n_augmentations=8)
    print(f"\nTest-Time Augmentation: {len(tta_augs)} variants created")
