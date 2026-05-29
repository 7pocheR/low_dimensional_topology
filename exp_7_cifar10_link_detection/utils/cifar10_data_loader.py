"""
CIFAR-10 data loading and PCA preprocessing for topological linking detection.

This module handles:
- Loading CIFAR-10 dataset
- Flattening images (32x32x3 -> 3072D)
- StandardScaler normalization (critical for meaningful Euclidean distances)
- Global PCA projection to 3D
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def download_cifar10(data_dir: str = '../data') -> Tuple[np.ndarray, np.ndarray]:
    """
    Download CIFAR-10 dataset and return flattened images with labels.

    Args:
        data_dir: Directory to store/load data

    Returns:
        X: Array of shape (50000, 3072) - flattened training images
        y: Array of shape (50000,) - class labels (0-9)
    """
    cache_file = os.path.join(data_dir, 'cifar10_full.npz')

    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"Loading CIFAR-10 from cache: {cache_file}")
        data = np.load(cache_file)
        return data['X'], data['y']

    print("Downloading CIFAR-10 dataset...")
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Try torchvision first (usually faster)
        import torchvision
        import torchvision.transforms as transforms

        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(data_dir, 'cifar10_raw'),
            train=True,
            download=True
        )

        # Convert to numpy arrays
        X = trainset.data.reshape(-1, 3072).astype(np.float32) / 255.0
        y = np.array(trainset.targets)

    except ImportError:
        # Fallback to keras/tensorflow
        try:
            from tensorflow.keras.datasets import cifar10
            (X_train, y_train), _ = cifar10.load_data()
            X = X_train.reshape(-1, 3072).astype(np.float32) / 255.0
            y = y_train.flatten()
        except ImportError:
            raise ImportError(
                "Neither torchvision nor tensorflow found. "
                "Please install one of them: pip install torchvision or pip install tensorflow"
            )

    # Save to cache
    np.savez(cache_file, X=X, y=y)
    print(f"Cached CIFAR-10 to: {cache_file}")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    return X, y


def load_cifar10_by_class(data_dir: str = '../data') -> Dict[int, np.ndarray]:
    """
    Load CIFAR-10 and organize by class.

    Returns:
        Dict mapping class_id (0-9) to array of shape (N, 3072)
    """
    X, y = download_cifar10(data_dir)

    class_data = {}
    for class_id in range(10):
        mask = y == class_id
        class_data[class_id] = X[mask]
        print(f"Class {class_id} ({CIFAR10_CLASSES[class_id]}): {class_data[class_id].shape[0]} samples")

    return class_data


def apply_global_pca(
    class_data: Dict[int, np.ndarray],
    n_components: int = 3,
    use_scaler: bool = True
) -> Tuple[Dict[int, np.ndarray], PCA, Optional[StandardScaler]]:
    """
    Apply global PCA to all classes combined.

    IMPORTANT: Uses StandardScaler before PCA to ensure Euclidean distances
    in PCA space are meaningful.

    Args:
        class_data: Dict mapping class_id to raw data arrays
        n_components: Number of PCA components (default 3)
        use_scaler: Whether to apply StandardScaler (recommended True)

    Returns:
        pca_data: Dict mapping class_id to PCA-projected data (N, 3)
        pca: Fitted PCA object
        scaler: Fitted StandardScaler (or None if use_scaler=False)
    """
    # Combine all class data
    all_data = np.vstack([class_data[i] for i in range(10)])
    class_sizes = [len(class_data[i]) for i in range(10)]

    print(f"\nApplying global PCA:")
    print(f"  Combined data shape: {all_data.shape}")
    print(f"  Using StandardScaler: {use_scaler}")

    # StandardScaler: zero mean, unit variance per feature
    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        all_data_scaled = scaler.fit_transform(all_data)
        print(f"  After scaling - mean: {all_data_scaled.mean():.6f}, std: {all_data_scaled.std():.6f}")
    else:
        all_data_scaled = all_data

    # Fit PCA
    pca = PCA(n_components=n_components)
    all_pca = pca.fit_transform(all_data_scaled)

    # Report explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    print(f"  Explained variance per component: {explained_var}")
    print(f"  Cumulative explained variance: {cumulative_var}")
    print(f"  Total variance captured in {n_components}D: {cumulative_var[-1]:.4f}")

    # Normalize PCA output to unit variance per component
    # This ensures Euclidean distances are meaningful and comparable to MNIST scale
    pca_std = all_pca.std(axis=0)
    all_pca_normalized = all_pca / pca_std
    print(f"  PCA output std before normalization: {pca_std}")
    print(f"  After PCA normalization - std per component: {all_pca_normalized.std(axis=0)}")

    # Split back into classes
    pca_data = {}
    start_idx = 0
    for class_id in range(10):
        end_idx = start_idx + class_sizes[class_id]
        pca_data[class_id] = all_pca_normalized[start_idx:end_idx]
        start_idx = end_idx

    return pca_data, pca, scaler


def load_or_create_cifar10_pca(
    data_dir: str = '../data',
    n_components: int = 3,
    force_recreate: bool = False
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    Load cached PCA data or create it if not exists.

    Returns:
        pca_data: Dict mapping class_id to PCA-projected data
        metadata: Dict with PCA stats and class info
    """
    cache_file = os.path.join(data_dir, f'cifar10_pca{n_components}d.npz')

    if os.path.exists(cache_file) and not force_recreate:
        print(f"Loading cached PCA data from: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)

        pca_data = {i: data[f'class_{i}'] for i in range(10)}
        metadata = data['metadata'].item()

        print(f"  Explained variance: {metadata['explained_variance_ratio']}")
        print(f"  Total variance captured: {metadata['total_variance']:.4f}")

        return pca_data, metadata

    # Load raw data and apply PCA
    print("Creating new PCA projection...")
    class_data = load_cifar10_by_class(data_dir)
    pca_data, pca, scaler = apply_global_pca(class_data, n_components, use_scaler=True)

    # Prepare metadata
    metadata = {
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_variance': np.sum(pca.explained_variance_ratio_),
        'use_scaler': True,
        'class_sizes': {i: len(pca_data[i]) for i in range(10)},
        'class_names': CIFAR10_CLASSES
    }

    # Save to cache
    save_dict = {f'class_{i}': pca_data[i] for i in range(10)}
    save_dict['metadata'] = metadata
    np.savez(cache_file, **save_dict)
    print(f"Saved PCA data to: {cache_file}")

    return pca_data, metadata


def get_class_name(class_id: int) -> str:
    """Get human-readable class name."""
    return CIFAR10_CLASSES[class_id]


def get_class_pair_name(class_i: int, class_j: int) -> str:
    """Get human-readable name for a class pair."""
    return f"{CIFAR10_CLASSES[class_i]}_vs_{CIFAR10_CLASSES[class_j]}"


def compute_distance_statistics(pca_data: Dict[int, np.ndarray]) -> Dict:
    """
    Compute distance statistics for each class in PCA space.

    This helps calibrate epsilon for k-NN graph construction.
    """
    from scipy.spatial.distance import pdist

    stats = {}

    for class_id in range(10):
        X = pca_data[class_id]

        # Sample if too large
        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        distances = pdist(X_sample)

        stats[class_id] = {
            'n_samples': len(pca_data[class_id]),
            'min': np.min(distances),
            'p10': np.percentile(distances, 10),
            'p25': np.percentile(distances, 25),
            'median': np.median(distances),
            'p75': np.percentile(distances, 75),
            'p90': np.percentile(distances, 90),
            'p95': np.percentile(distances, 95),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances)
        }

    return stats


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("CIFAR-10 DATA LOADER TEST")
    print("=" * 60)

    # Load PCA data
    pca_data, metadata = load_or_create_cifar10_pca()

    print("\n" + "=" * 60)
    print("PCA DATA SUMMARY")
    print("=" * 60)

    for class_id in range(10):
        X = pca_data[class_id]
        print(f"Class {class_id} ({CIFAR10_CLASSES[class_id]}): "
              f"shape={X.shape}, "
              f"range=[{X.min():.2f}, {X.max():.2f}]")

    print("\n" + "=" * 60)
    print("DISTANCE STATISTICS")
    print("=" * 60)

    stats = compute_distance_statistics(pca_data)

    # Print summary
    print(f"\n{'Class':<12} {'Median':<8} {'P25':<8} {'P75':<8} {'P90':<8} {'P95':<8}")
    print("-" * 60)
    for class_id in range(10):
        s = stats[class_id]
        print(f"{CIFAR10_CLASSES[class_id]:<12} "
              f"{s['median']:<8.3f} "
              f"{s['p25']:<8.3f} "
              f"{s['p75']:<8.3f} "
              f"{s['p90']:<8.3f} "
              f"{s['p95']:<8.3f}")

    # Suggest epsilon values based on statistics
    all_medians = [stats[i]['median'] for i in range(10)]
    all_p90 = [stats[i]['p90'] for i in range(10)]

    print(f"\n" + "=" * 60)
    print("SUGGESTED EPSILON VALUES")
    print("=" * 60)
    print(f"Based on median distances: {np.mean(all_medians):.3f}")
    print(f"Based on 90th percentile: {np.mean(all_p90):.3f}")
    print(f"Suggested range: [{np.mean(all_medians)*0.3:.3f}, {np.mean(all_p90)*0.5:.3f}]")
