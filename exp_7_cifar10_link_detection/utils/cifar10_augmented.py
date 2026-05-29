#!/usr/bin/env python3
"""
CIFAR-10 with fast data augmentation for denser manifold sampling.
Uses numpy-based transforms and multiprocessing for speed.
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict
from multiprocessing import Pool, cpu_count
import time

# Ensure reproducibility
np.random.seed(42)


def fast_augment_batch(args):
    """
    Fast augmentation of a batch of images using numpy operations.
    Called by multiprocessing pool.

    Args:
        args: (images, n_aug, start_seed) tuple

    Returns:
        Augmented images array
    """
    images, n_aug, start_seed = args
    n_images = len(images)

    # Preallocate output
    augmented = np.zeros((n_images * n_aug, 3072), dtype=np.float32)

    for i in range(n_images):
        img = images[i].reshape(32, 32, 3)

        for aug_idx in range(n_aug):
            np.random.seed(start_seed + i * n_aug + aug_idx)
            aug_img = img.copy()

            # Random horizontal flip (50% chance)
            if np.random.random() < 0.5:
                aug_img = aug_img[:, ::-1, :]

            # Random brightness adjustment (±15%)
            brightness = np.random.uniform(0.85, 1.15)
            aug_img = aug_img * brightness

            # Random contrast adjustment (±15%)
            contrast = np.random.uniform(0.85, 1.15)
            mean = aug_img.mean()
            aug_img = (aug_img - mean) * contrast + mean

            # Small random noise (±2%)
            noise = np.random.uniform(-0.02, 0.02, aug_img.shape)
            aug_img = aug_img + noise

            # Random small translation (±2 pixels) using roll
            tx = np.random.randint(-2, 3)
            ty = np.random.randint(-2, 3)
            aug_img = np.roll(aug_img, tx, axis=0)
            aug_img = np.roll(aug_img, ty, axis=1)

            # Clip to valid range
            aug_img = np.clip(aug_img, 0, 1)

            augmented[i * n_aug + aug_idx] = aug_img.flatten()

    return augmented


def enhanced_augment_batch(args):
    """
    Enhanced augmentation with more techniques for denser manifold sampling.
    Includes: flip, brightness, contrast, noise, translation, rotation,
              scale, color channel shifts, gaussian blur, random crop, cutout.
    """
    from scipy.ndimage import rotate as scipy_rotate, zoom as scipy_zoom
    from scipy.ndimage import gaussian_filter

    images, n_aug, start_seed = args
    n_images = len(images)

    # Preallocate output
    augmented = np.zeros((n_images * n_aug, 3072), dtype=np.float32)

    for i in range(n_images):
        img = images[i].reshape(32, 32, 3)

        for aug_idx in range(n_aug):
            np.random.seed(start_seed + i * n_aug + aug_idx)
            aug_img = img.copy()

            # 1. Random horizontal flip (50% chance)
            if np.random.random() < 0.5:
                aug_img = aug_img[:, ::-1, :]

            # 2. Random Crop with Padding (VERY COMMON for CIFAR-10)
            # Pad to 40x40, then random crop back to 32x32
            if np.random.random() < 0.8:
                pad = 4
                padded = np.pad(aug_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
                # Random crop position
                crop_y = np.random.randint(0, 2 * pad + 1)
                crop_x = np.random.randint(0, 2 * pad + 1)
                aug_img = padded[crop_y:crop_y+32, crop_x:crop_x+32, :]

            # 3. Random rotation (±10 degrees)
            if np.random.random() < 0.5:
                angle = np.random.uniform(-10, 10)
                aug_img = scipy_rotate(aug_img, angle, axes=(0, 1),
                                       reshape=False, order=1, mode='nearest')

            # 4. Random scale/zoom (0.9 to 1.1)
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.9, 1.1)
                h, w = aug_img.shape[:2]
                scaled = scipy_zoom(aug_img, (scale, scale, 1), order=1)
                sh, sw = scaled.shape[:2]
                # Center crop or pad
                if scale > 1:
                    start_h = (sh - h) // 2
                    start_w = (sw - w) // 2
                    aug_img = scaled[start_h:start_h+h, start_w:start_w+w, :]
                else:
                    pad_h = (h - sh) // 2
                    pad_w = (w - sw) // 2
                    padded = np.zeros_like(aug_img)
                    padded[pad_h:pad_h+sh, pad_w:pad_w+sw, :] = scaled
                    aug_img = padded

            # 5. Random brightness (±20%)
            brightness = np.random.uniform(0.8, 1.2)
            aug_img = aug_img * brightness

            # 6. Random contrast (±20%)
            contrast = np.random.uniform(0.8, 1.2)
            mean = aug_img.mean()
            aug_img = (aug_img - mean) * contrast + mean

            # 7. Random saturation adjustment
            if np.random.random() < 0.4:
                saturation = np.random.uniform(0.8, 1.2)
                gray = aug_img.mean(axis=2, keepdims=True)
                aug_img = gray + saturation * (aug_img - gray)

            # 8. Per-channel color shifts (±5%)
            if np.random.random() < 0.4:
                for c in range(3):
                    shift = np.random.uniform(-0.05, 0.05)
                    aug_img[:, :, c] = aug_img[:, :, c] + shift

            # 9. Small random noise (±3%)
            noise = np.random.uniform(-0.03, 0.03, aug_img.shape)
            aug_img = aug_img + noise

            # 10. Random translation (±3 pixels) - reduced probability since we have random crop
            if np.random.random() < 0.3:
                tx = np.random.randint(-3, 4)
                ty = np.random.randint(-3, 4)
                aug_img = np.roll(aug_img, tx, axis=0)
                aug_img = np.roll(aug_img, ty, axis=1)

            # 11. Optional gaussian blur (20% chance)
            if np.random.random() < 0.2:
                sigma = np.random.uniform(0.3, 0.8)
                for c in range(3):
                    aug_img[:, :, c] = gaussian_filter(aug_img[:, :, c], sigma)

            # 12. Cutout / Random Erasing (30% chance)
            # Randomly mask a square region
            if np.random.random() < 0.3:
                cutout_size = np.random.randint(4, 12)  # 4-12 pixel squares
                cx = np.random.randint(0, 32)
                cy = np.random.randint(0, 32)
                x1 = max(0, cx - cutout_size // 2)
                x2 = min(32, cx + cutout_size // 2)
                y1 = max(0, cy - cutout_size // 2)
                y2 = min(32, cy + cutout_size // 2)
                # Fill with mean color (helps maintain manifold structure)
                aug_img[y1:y2, x1:x2, :] = aug_img.mean()

            # Clip to valid range
            aug_img = np.clip(aug_img, 0, 1)

            augmented[i * n_aug + aug_idx] = aug_img.flatten()

    return augmented


def create_augmented_cifar10(
    n_augmentations: int = 5,
    data_dir: str = 'data',
    force_recreate: bool = False,
    n_workers: int = None,
    enhanced: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented CIFAR-10 dataset using parallel processing.

    Args:
        n_augmentations: Number of augmented copies per original image
        data_dir: Directory for data storage
        force_recreate: Force recreation even if cache exists
        n_workers: Number of parallel workers (default: CPU count)
        enhanced: Use enhanced augmentation (more techniques, slower)

    Returns:
        X: Augmented images, shape (N * (1 + n_aug), 3072)
        y: Labels, shape (N * (1 + n_aug),)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    suffix = 'enhanced' if enhanced else 'fast'
    cache_file = os.path.join(data_dir, f'cifar10_aug{n_augmentations}x_{suffix}.npz')

    if os.path.exists(cache_file) and not force_recreate:
        print(f"Loading augmented CIFAR-10 from cache: {cache_file}")
        data = np.load(cache_file)
        return data['X'], data['y']

    # Load original data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cifar10_data_loader import download_cifar10
    X_orig, y_orig = download_cifar10(data_dir)

    n_orig = len(X_orig)
    n_total = n_orig * (1 + n_augmentations)

    print(f"Creating augmented CIFAR-10 with {n_workers} workers:")
    print(f"  Original: {n_orig}")
    print(f"  Augmentations per image: {n_augmentations}")
    print(f"  Total output: {n_total}")

    # Split data into batches for parallel processing
    batch_size = n_orig // n_workers
    batches = []
    for i in range(n_workers):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < n_workers - 1 else n_orig
        batch_images = X_orig[start_idx:end_idx]
        batches.append((batch_images, n_augmentations, i * batch_size * n_augmentations))

    # Parallel augmentation
    aug_func = enhanced_augment_batch if enhanced else fast_augment_batch
    aug_type = "enhanced" if enhanced else "fast"
    print(f"  Starting parallel {aug_type} augmentation...")
    start_time = time.time()

    with Pool(n_workers) as pool:
        results = pool.map(aug_func, batches)

    aug_time = time.time() - start_time
    print(f"  Augmentation completed in {aug_time:.1f}s")

    # Combine results
    X_augmented = np.vstack(results)

    # Create labels for augmented data
    y_augmented = np.repeat(y_orig, n_augmentations)

    # Combine original and augmented
    X_combined = np.vstack([X_orig, X_augmented])
    y_combined = np.concatenate([y_orig, y_augmented])

    # Shuffle
    print(f"  Shuffling {len(X_combined)} samples...")
    perm = np.random.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    # Save to cache
    print(f"  Saving to: {cache_file}")
    np.savez_compressed(cache_file, X=X_combined, y=y_combined)

    total_time = time.time() - start_time
    print(f"  Total augmentation time: {total_time:.1f}s")

    return X_combined, y_combined


def create_augmented_pca(
    n_augmentations: int = 5,
    n_components: int = 3,
    data_dir: str = 'data',
    force_recreate: bool = False,
    enhanced: bool = False
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    Create PCA projection of augmented CIFAR-10.

    Args:
        n_augmentations: Number of augmented copies per image
        n_components: PCA components
        data_dir: Data directory
        force_recreate: Force recreation
        enhanced: Use enhanced augmentation

    Returns:
        pca_data: Dict mapping class_id to PCA-projected points
        metadata: Dict with PCA statistics
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import time

    suffix = 'enhanced' if enhanced else 'fast'
    cache_file = os.path.join(data_dir, f'cifar10_aug{n_augmentations}x_{suffix}_pca{n_components}d.npz')

    if os.path.exists(cache_file) and not force_recreate:
        print(f"Loading augmented PCA from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        pca_data = {i: data[f'class_{i}'] for i in range(10)}
        metadata = data['metadata'].item()
        print(f"  Total samples: {sum(len(pca_data[i]) for i in range(10))}")
        print(f"  Explained variance: {metadata['explained_variance_ratio']}")
        return pca_data, metadata

    # Load augmented data
    X, y = create_augmented_cifar10(n_augmentations, data_dir, force_recreate, enhanced=enhanced)

    print(f"\nApplying StandardScaler + PCA to {len(X)} samples...")
    sys.stdout.flush()
    start_time = time.time()

    # StandardScaler - use partial_fit for large datasets
    print("  StandardScaler...")
    sys.stdout.flush()
    scaler = StandardScaler()

    # Process in chunks for memory efficiency
    chunk_size = 50000
    n_samples = len(X)

    if n_samples > 100000:
        # Incremental approach for large datasets
        print(f"    Using incremental fitting ({n_samples} samples, {chunk_size} per chunk)...")
        sys.stdout.flush()
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            scaler.partial_fit(X[i:end_idx])
            print(f"    Scaler: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")
            sys.stdout.flush()
        X_scaled = np.zeros_like(X)
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            X_scaled[i:end_idx] = scaler.transform(X[i:end_idx])
    else:
        X_scaled = scaler.fit_transform(X)

    # Use IncrementalPCA for large datasets (more memory efficient)
    print("  PCA...")
    sys.stdout.flush()

    if n_samples > 100000:
        from sklearn.decomposition import IncrementalPCA
        print(f"    Using IncrementalPCA ({n_samples} samples, batch_size={chunk_size})...")
        sys.stdout.flush()
        pca = IncrementalPCA(n_components=n_components, batch_size=chunk_size)
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            pca.partial_fit(X_scaled[i:end_idx])
            print(f"    PCA fit: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")
            sys.stdout.flush()
        X_pca = pca.transform(X_scaled)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        X_pca = pca.fit_transform(X_scaled)

    # Normalize PCA output
    pca_std = X_pca.std(axis=0)
    X_pca_normalized = X_pca / pca_std

    pca_time = time.time() - start_time
    print(f"  PCA completed in {pca_time:.1f}s")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")

    # Split by class
    pca_data = {}
    for class_id in range(10):
        mask = (y == class_id)
        pca_data[class_id] = X_pca_normalized[mask]
        print(f"    Class {class_id}: {len(pca_data[class_id])} samples")

    # Metadata
    metadata = {
        'n_components': n_components,
        'n_augmentations': n_augmentations,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_variance': sum(pca.explained_variance_ratio_),
        'pca_std': pca_std,
        'class_sizes': {i: len(pca_data[i]) for i in range(10)},
        'total_samples': len(X)
    }

    # Save
    save_dict = {f'class_{i}': pca_data[i] for i in range(10)}
    save_dict['metadata'] = metadata
    np.savez_compressed(cache_file, **save_dict)
    print(f"  Saved augmented PCA to: {cache_file}")

    return pca_data, metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create augmented CIFAR-10')
    parser.add_argument('--n-aug', type=int, default=5, help='Augmentations per image')
    parser.add_argument('--force', action='store_true', help='Force recreation')
    args = parser.parse_args()

    pca_data, metadata = create_augmented_pca(
        n_augmentations=args.n_aug,
        force_recreate=args.force
    )

    print("\nClass sizes:")
    for i in range(10):
        print(f"  Class {i}: {len(pca_data[i])} samples")
