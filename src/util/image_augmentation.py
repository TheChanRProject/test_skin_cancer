"""
Image augmentation utilities for skin cancer lesion images (HAM10000 dataset).

This module provides functions to augment dermoscopic images while preserving
image properties and maintaining compatibility with torchvision ImageFolder datasets.
"""

import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def get_skin_lesion_augmentation_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    preserve_original_size: bool = True,
    intensity: str = "medium"
) -> A.Compose:
    """
    Create an albumentations augmentation pipeline specifically for skin lesion images.

    The augmentations are carefully chosen to preserve lesion characteristics while
    adding realistic variations that might occur in clinical settings (different
    lighting, angles, slight color variations, etc.).

    Args:
        image_size: Target image size (height, width). Used only if preserve_original_size is False.
        preserve_original_size: If True, doesn't resize images. If False, resizes to image_size.
        intensity: Augmentation intensity - "light", "medium", or "strong"

    Returns:
        Albumentations Compose object with the augmentation pipeline
    """
    intensity_params = {
        "light": {
            "rotate_limit": 20,
            "shift_scale_rotate_prob": 0.3,
            "brightness_contrast_limit": 0.1,
            "hue_saturation_limit": 10,
            "blur_limit": 3,
        },
        "medium": {
            "rotate_limit": 45,
            "shift_scale_rotate_prob": 0.5,
            "brightness_contrast_limit": 0.2,
            "hue_saturation_limit": 20,
            "blur_limit": 5,
        },
        "strong": {
            "rotate_limit": 90,
            "shift_scale_rotate_prob": 0.7,
            "brightness_contrast_limit": 0.3,
            "hue_saturation_limit": 30,
            "blur_limit": 7,
        }
    }

    params = intensity_params.get(intensity, intensity_params["medium"])

    transforms = [
        # Geometric transformations - common in dermatoscopy due to camera angle
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=params["rotate_limit"],
            border_mode=cv2.BORDER_CONSTANT,
            p=params["shift_scale_rotate_prob"]
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Color/lighting variations - realistic for different imaging conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=params["brightness_contrast_limit"],
                contrast_limit=params["brightness_contrast_limit"],
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=params["hue_saturation_limit"],
                sat_shift_limit=params["hue_saturation_limit"],
                val_shift_limit=params["hue_saturation_limit"],
                p=1.0
            ),
        ], p=0.5),

        # Slight blur - can occur due to focus issues
        A.OneOf([
            A.Blur(blur_limit=params["blur_limit"], p=1.0),
            A.GaussianBlur(blur_limit=params["blur_limit"], p=1.0),
        ], p=0.2),

        # Elastic transformations - subtle skin deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),

        # Optical distortion - lens effects
        A.OpticalDistortion(
            distort_limit=0.05,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),

        # Grid distortion - subtle geometric variations
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
    ]

    # Add resize if not preserving original size
    if not preserve_original_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1], p=1.0))

    return A.Compose(transforms)


def analyze_class_distribution(root_path: Union[str, Path]) -> Dict[str, int]:
    """
    Analyze the class distribution in an ImageFolder-style directory.

    Args:
        root_path: Path to the root directory containing class subdirectories

    Returns:
        Dictionary mapping class names to image counts
    """
    root_path = Path(root_path)
    class_counts = {}

    for class_dir in sorted(root_path.iterdir()):
        if class_dir.is_dir():
            # Count image files (common extensions)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            count = sum(
                1 for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            )
            class_counts[class_dir.name] = count

    return class_counts


def calculate_augmentation_targets(
    class_counts: Dict[str, int],
    balance_strategy: str = "match_max",
    target_count: Optional[int] = None
) -> Dict[str, int]:
    """
    Calculate how many augmented images to generate per class.

    Args:
        class_counts: Dictionary mapping class names to current counts
        balance_strategy: Strategy for balancing classes:
            - "match_max": Balance all classes to match the largest class
            - "match_median": Balance all classes to match the median class size
            - "match_target": Balance all classes to match target_count
        target_count: Target count per class (used with "match_target" strategy)

    Returns:
        Dictionary mapping class names to number of augmentations needed
    """
    if balance_strategy == "match_max":
        max_count = max(class_counts.values())
        targets = {cls: max_count - count for cls, count in class_counts.items()}
    elif balance_strategy == "match_median":
        median_count = int(np.median(list(class_counts.values())))
        targets = {cls: max(0, median_count - count) for cls, count in class_counts.items()}
    elif balance_strategy == "match_target":
        if target_count is None:
            raise ValueError("target_count must be provided for 'match_target' strategy")
        targets = {cls: max(0, target_count - count) for cls, count in class_counts.items()}
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")

    return targets


def generate_augmented_dataset(
    source_root: Union[str, Path],
    output_root: Union[str, Path],
    augmentations_per_image: int = 5,
    balance_strategy: str = "match_max",
    target_count: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    preserve_original_size: bool = True,
    intensity: str = "medium",
    include_originals: bool = True,
    seed: int = 42
) -> Dict[str, int]:
    """
    Generate augmented images to balance class distribution for torchvision ImageFolder.

    This function:
    1. Analyzes class distribution in source directory
    2. Calculates how many augmentations are needed per class
    3. Generates augmented images using albumentations
    4. Saves them in ImageFolder-compatible structure

    Args:
        source_root: Path to source directory with subdirectories per class
        output_root: Path to output directory (will be created if doesn't exist)
        augmentations_per_image: Number of augmented versions to create per original image
        balance_strategy: How to balance classes ("match_max", "match_median", "match_target")
        target_count: Target count per class (for "match_target" strategy)
        image_size: Target image size (height, width). None to preserve original
        preserve_original_size: Whether to preserve original image dimensions
        intensity: Augmentation intensity ("light", "medium", "strong")
        include_originals: If True, copy original images to output directory
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics about the augmentation process
    """
    np.random.seed(seed)

    source_root = Path(source_root)
    output_root = Path(output_root)

    # Validate source directory
    if not source_root.exists():
        raise ValueError(f"Source directory does not exist: {source_root}")

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Analyze current distribution
    print("Analyzing class distribution...")
    class_counts = analyze_class_distribution(source_root)
    print(f"Found {len(class_counts)} classes:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"  {cls_name}: {count} images")

    # Calculate augmentation targets
    aug_targets = calculate_augmentation_targets(
        class_counts,
        balance_strategy=balance_strategy,
        target_count=target_count
    )
    print(f"\nAugmentation targets (strategy: {balance_strategy}):")
    for cls_name, target in sorted(aug_targets.items()):
        print(f"  {cls_name}: {target} augmentations needed")

    # Create augmentation pipeline
    aug_pipeline = get_skin_lesion_augmentation_pipeline(
        image_size=image_size or (224, 224),
        preserve_original_size=preserve_original_size,
        intensity=intensity
    )

    # Statistics
    stats = {
        'original_counts': class_counts.copy(),
        'augmentation_targets': aug_targets.copy(),
        'generated_counts': {},
        'final_counts': {}
    }

    # Process each class
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for class_name in sorted(class_counts.keys()):
        print(f"\nProcessing class: {class_name}")

        # Create output class directory
        output_class_dir = output_root / class_name
        output_class_dir.mkdir(exist_ok=True)

        # Get all images in this class
        source_class_dir = source_root / class_name
        image_files = [
            f for f in source_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        # Copy originals if requested
        if include_originals:
            print(f"  Copying {len(image_files)} original images...")
            for img_file in tqdm(image_files, desc="  Copying"):
                shutil.copy2(img_file, output_class_dir / img_file.name)

        # Generate augmentations
        num_augmentations_needed = aug_targets[class_name]
        if num_augmentations_needed <= 0:
            print(f"  No augmentations needed for {class_name}")
            stats['generated_counts'][class_name] = 0
            continue

        # Calculate how many augmentations per image
        num_original_images = len(image_files)
        augs_per_image_this_class = max(1, num_augmentations_needed // num_original_images)

        print(f"  Generating {num_augmentations_needed} augmentations...")
        print(f"  (~{augs_per_image_this_class} augmentations per image)")

        generated_count = 0

        for img_file in tqdm(image_files, desc="  Augmenting"):
            # Read image
            image_bgr = cv2.imread(str(img_file))
            if image_bgr is None:
                print(f"  Warning: Could not read image {img_file}, skipping...")
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Generate augmentations for this image
            num_augs_for_this_image = min(
                augs_per_image_this_class,
                num_augmentations_needed - generated_count
            )

            for aug_idx in range(num_augs_for_this_image):
                # Apply augmentation
                augmented = aug_pipeline(image=image)
                aug_image = augmented['image']

                # Save augmented image
                base_name = img_file.stem
                aug_filename = f"{base_name}_aug_{aug_idx}{img_file.suffix}"
                aug_path = output_class_dir / aug_filename

                # Convert back to BGR for saving with cv2
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_path), aug_image_bgr)

                generated_count += 1

                if generated_count >= num_augmentations_needed:
                    break

            if generated_count >= num_augmentations_needed:
                break

        stats['generated_counts'][class_name] = generated_count

        # Count final images in output directory
        final_count = sum(
            1 for f in output_class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        stats['final_counts'][class_name] = final_count

    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"{'Class':<20} {'Original':<12} {'Generated':<12} {'Final':<12}")
    print("-"*60)
    for class_name in sorted(class_counts.keys()):
        print(
            f"{class_name:<20} "
            f"{stats['original_counts'][class_name]:<12} "
            f"{stats['generated_counts'].get(class_name, 0):<12} "
            f"{stats['final_counts'][class_name]:<12}"
        )
    print("="*60)

    return stats


def create_augmentation_transform_for_dataloader(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True
) -> A.Compose:
    """
    Create an albumentations transform that can be used with PyTorch DataLoader.

    This is meant to be used as a custom transform in torchvision datasets.
    Unlike generate_augmented_dataset which creates augmented files,
    this performs augmentation on-the-fly during training.

    Args:
        image_size: Target image size (height, width)
        is_training: If True, applies augmentations. If False, only resizes/normalizes.

    Returns:
        Albumentations Compose object

    Example:
        >>> from torchvision import datasets
        >>> from PIL import Image
        >>> import numpy as np
        >>>
        >>> # Create transform
        >>> aug_transform = create_augmentation_transform_for_dataloader(
        >>>     image_size=(224, 224),
        >>>     is_training=True
        >>> )
        >>>
        >>> # Custom transform function for ImageFolder
        >>> def transform_fn(image):
        >>>     # Convert PIL to numpy
        >>>     image_np = np.array(image)
        >>>     # Apply augmentations
        >>>     augmented = aug_transform(image=image_np)
        >>>     return augmented['image']
        >>>
        >>> # Use with ImageFolder
        >>> dataset = datasets.ImageFolder(root='path/to/data', transform=transform_fn)
    """
    if is_training:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
        ])
    else:
        # Validation/test: only resize
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
        ])


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate augmented skin lesion images for class balancing"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source directory with class subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="match_max",
        choices=["match_max", "match_median", "match_target"],
        help="Class balancing strategy"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target count per class (for match_target strategy)"
    )
    parser.add_argument(
        "--intensity",
        type=str,
        default="medium",
        choices=["light", "medium", "strong"],
        help="Augmentation intensity"
    )
    parser.add_argument(
        "--no-originals",
        action="store_true",
        help="Don't copy original images to output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    stats = generate_augmented_dataset(
        source_root=args.source,
        output_root=args.output,
        balance_strategy=args.strategy,
        target_count=args.target_count,
        intensity=args.intensity,
        include_originals=not args.no_originals,
        seed=args.seed
    )

    print("\nDone! Augmented dataset created at:", args.output)
