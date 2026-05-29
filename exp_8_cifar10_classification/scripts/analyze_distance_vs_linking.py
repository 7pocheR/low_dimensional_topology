#!/usr/bin/env python3
"""
Compare inter-class distance correlation vs linking consistency correlation
with classifier confusion rates.

Distance metrics:
1. Centroid distance: Euclidean distance between class means
2. Average pairwise distance: Mean distance between samples from two classes
3. Minimum distance: Minimum distance between any two samples (proxy for margin)
4. Centroid distance in PCA space: Same as (1) but in 50D PCA space

We compare Spearman correlation of each metric with confusion rates,
and compare against linking consistency correlation.
"""

import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path
from itertools import combinations

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results" / "cnn_10class"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "results" / "distance_analysis"))
DATA_DIR = Path(os.environ.get("CIFAR10_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Linking consistency matrix from paper (Table~\ref{tab:linking-matrix})
# Values are fraction of 11 runs detecting lk != 0
# Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
LINKING_CONSISTENCY = np.array([
    [0.00, 0.27, 0.73, 0.27, 0.82, 0.36, 0.27, 0.18, 0.91, 0.27],  # airplane
    [0.27, 0.00, 0.82, 0.82, 0.91, 0.82, 0.82, 0.82, 0.55, 0.91],  # automobile
    [0.73, 0.82, 0.00, 0.91, 0.82, 0.91, 0.91, 0.82, 0.45, 0.73],  # bird
    [0.27, 0.82, 0.91, 0.00, 0.91, 0.82, 0.91, 0.82, 0.27, 0.45],  # cat
    [0.82, 0.91, 0.82, 0.91, 0.00, 0.91, 0.82, 0.91, 0.45, 0.27],  # deer
    [0.36, 0.82, 0.91, 0.82, 0.91, 0.00, 0.82, 0.64, 0.36, 0.55],  # dog
    [0.27, 0.82, 0.91, 0.91, 0.82, 0.82, 0.00, 0.82, 0.18, 0.55],  # frog
    [0.18, 0.82, 0.82, 0.82, 0.91, 0.64, 0.82, 0.00, 0.36, 0.82],  # horse
    [0.91, 0.55, 0.45, 0.27, 0.45, 0.36, 0.18, 0.36, 0.00, 0.82],  # ship
    [0.27, 0.91, 0.73, 0.45, 0.27, 0.55, 0.55, 0.82, 0.82, 0.00],  # truck
])


def load_cifar10():
    """Load CIFAR-10 training data."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    # Convert to numpy
    X = np.array([img.numpy().flatten() for img, _ in trainset])
    y = np.array([label for _, label in trainset])

    return X, y


def compute_class_data(X, y):
    """Split data by class."""
    class_data = {}
    for c in range(10):
        class_data[c] = X[y == c]
    return class_data


def compute_centroid_distances(class_data):
    """Compute pairwise centroid distances."""
    centroids = np.array([class_data[c].mean(axis=0) for c in range(10)])
    distances = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    return distances


def compute_avg_pairwise_distances(class_data, n_samples=500):
    """Compute average pairwise distances between classes (subsampled)."""
    distances = np.zeros((10, 10))

    for i in range(10):
        for j in range(i+1, 10):
            # Subsample for efficiency
            idx_i = np.random.choice(len(class_data[i]), min(n_samples, len(class_data[i])), replace=False)
            idx_j = np.random.choice(len(class_data[j]), min(n_samples, len(class_data[j])), replace=False)

            # Compute pairwise distances
            dists = cdist(class_data[i][idx_i], class_data[j][idx_j], metric='euclidean')
            distances[i, j] = distances[j, i] = dists.mean()

    return distances


def compute_min_distances(class_data, n_samples=1000):
    """Compute minimum distances between classes (subsampled)."""
    distances = np.zeros((10, 10))

    for i in range(10):
        for j in range(i+1, 10):
            # Subsample for efficiency
            idx_i = np.random.choice(len(class_data[i]), min(n_samples, len(class_data[i])), replace=False)
            idx_j = np.random.choice(len(class_data[j]), min(n_samples, len(class_data[j])), replace=False)

            # Compute pairwise distances
            dists = cdist(class_data[i][idx_i], class_data[j][idx_j], metric='euclidean')
            distances[i, j] = distances[j, i] = dists.min()

    return distances


def compute_pca_centroid_distances(class_data, n_components=50):
    """Compute centroid distances in PCA space."""
    # Combine all data for PCA fitting
    all_data = np.vstack([class_data[c] for c in range(10)])

    # Standardize and apply PCA
    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    pca = PCA(n_components=n_components)
    all_data_pca = pca.fit_transform(all_data_scaled)

    # Split back by class
    idx = 0
    class_data_pca = {}
    for c in range(10):
        n = len(class_data[c])
        class_data_pca[c] = all_data_pca[idx:idx+n]
        idx += n

    # Compute centroid distances
    centroids = np.array([class_data_pca[c].mean(axis=0) for c in range(10)])
    distances = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    return distances


def load_confusion_matrices():
    """Load confusion matrices from experiment results."""
    confusion_data = {}

    for result_file in RESULTS_DIR.glob("result_*.json"):
        with open(result_file) as f:
            data = json.load(f)

        # Extract config from filename: result_cnn10_{act}_L{depth}_B{skip}.json
        name = result_file.stem
        parts = name.split('_')
        act = parts[2]
        depth = parts[3]  # L5, L8, L11
        skip = parts[4]   # B0 (no skip), B1 (skip)

        key = f"{act}_{depth}_{skip}"

        if 'confusion_matrix' in data:
            confusion_data[key] = {
                'confusion_matrix': np.array(data['confusion_matrix']),
                'test_acc': data.get('test_acc', data.get('test_accuracy', 0)),
                'activation': act,
                'depth': depth,
                'skip': skip
            }

    return confusion_data


def extract_pairwise_confusion(confusion_matrix):
    """Extract symmetric pairwise confusion rates."""
    confusion = np.zeros((10, 10))
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = confusion_matrix / row_sums

    for i in range(10):
        for j in range(10):
            if i != j:
                confusion[i, j] = normalized[i, j] + normalized[j, i]

    return confusion


def matrix_to_pairs(matrix):
    """Convert 10x10 matrix to list of 45 pair values (upper triangle)."""
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append(matrix[i, j])
    return np.array(pairs)


def compute_correlations(confusion_pairs, distance_pairs, linking_pairs):
    """Compute Spearman correlations."""
    # Distance vs confusion (should be NEGATIVE - closer = more confusion)
    r_dist, p_dist = stats.spearmanr(distance_pairs, confusion_pairs)

    # Linking vs confusion (should be POSITIVE - more linked = more confusion)
    r_link, p_link = stats.spearmanr(linking_pairs, confusion_pairs)

    return {
        'distance_r': r_dist,
        'distance_p': p_dist,
        'linking_r': r_link,
        'linking_p': p_link
    }


def main():
    print("=" * 60)
    print("Inter-class Distance vs Linking Consistency Analysis")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    print("\nLoading CIFAR-10 data...")
    X, y = load_cifar10()
    print(f"Data shape: {X.shape}")

    # Split by class
    class_data = compute_class_data(X, y)

    # Compute distance metrics
    print("\nComputing distance metrics...")

    print("  - Centroid distances (raw)...")
    centroid_dist = compute_centroid_distances(class_data)

    print("  - Average pairwise distances...")
    avg_dist = compute_avg_pairwise_distances(class_data, n_samples=500)

    print("  - Minimum distances...")
    min_dist = compute_min_distances(class_data, n_samples=1000)

    print("  - Centroid distances (PCA-50)...")
    pca_centroid_dist = compute_pca_centroid_distances(class_data, n_components=50)

    # Convert to pair lists
    centroid_pairs = matrix_to_pairs(centroid_dist)
    avg_pairs = matrix_to_pairs(avg_dist)
    min_pairs = matrix_to_pairs(min_dist)
    pca_centroid_pairs = matrix_to_pairs(pca_centroid_dist)
    linking_pairs = matrix_to_pairs(LINKING_CONSISTENCY)

    # Load confusion matrices
    print("\nLoading confusion matrices...")
    confusion_data = load_confusion_matrices()
    print(f"Found {len(confusion_data)} experiment results")

    # Analyze each model
    results = []

    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'Acc':>6} | {'Centroid r':>11} {'Avg r':>11} {'Min r':>11} {'PCA r':>11} | {'Linking r':>11}")
    print("=" * 100)

    for key, data in sorted(confusion_data.items()):
        confusion_matrix = data['confusion_matrix']
        pairwise_confusion = extract_pairwise_confusion(confusion_matrix)
        confusion_pairs = matrix_to_pairs(pairwise_confusion)

        # Compute correlations for each distance metric
        r_centroid, p_centroid = stats.spearmanr(centroid_pairs, confusion_pairs)
        r_avg, p_avg = stats.spearmanr(avg_pairs, confusion_pairs)
        r_min, p_min = stats.spearmanr(min_pairs, confusion_pairs)
        r_pca, p_pca = stats.spearmanr(pca_centroid_pairs, confusion_pairs)
        r_link, p_link = stats.spearmanr(linking_pairs, confusion_pairs)

        result = {
            'model': key,
            'test_acc': data['test_acc'],
            'activation': data['activation'],
            'depth': data['depth'],
            'skip': data['skip'],
            'centroid_r': r_centroid,
            'centroid_p': p_centroid,
            'avg_r': r_avg,
            'avg_p': p_avg,
            'min_r': r_min,
            'min_p': p_min,
            'pca_centroid_r': r_pca,
            'pca_centroid_p': p_pca,
            'linking_r': r_link,
            'linking_p': p_link
        }
        results.append(result)

        print(f"{key:<25} {data['test_acc']*100:>5.1f}% | {r_centroid:>+11.3f} {r_avg:>+11.3f} {r_min:>+11.3f} {r_pca:>+11.3f} | {r_link:>+11.3f}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY (mean across all models)")
    print("=" * 100)

    metrics = ['centroid_r', 'avg_r', 'min_r', 'pca_centroid_r', 'linking_r']
    metric_names = ['Centroid (raw)', 'Avg pairwise', 'Minimum', 'Centroid (PCA)', 'Linking consistency']

    print(f"\n{'Metric':<25} {'Mean r':>10} {'Std r':>10} {'Mean |r|':>10} {'Interpretation':<30}")
    print("-" * 85)

    for metric, name in zip(metrics, metric_names):
        vals = [r[metric] for r in results]
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        mean_abs_r = np.mean(np.abs(vals))

        # Interpretation
        if 'linking' in metric:
            interp = "Higher link -> more confusion" if mean_r > 0 else "Lower link -> more confusion"
        else:
            interp = "Closer -> more confusion" if mean_r < 0 else "Farther -> more confusion"

        print(f"{name:<25} {mean_r:>+10.3f} {std_r:>10.3f} {mean_abs_r:>10.3f} {interp:<30}")

    # Statistical comparison: is linking correlation significantly different from distance correlation?
    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON: Linking vs Distance metrics")
    print("=" * 100)

    linking_rs = [r['linking_r'] for r in results]

    for metric, name in zip(['centroid_r', 'avg_r', 'min_r', 'pca_centroid_r'],
                           ['Centroid (raw)', 'Avg pairwise', 'Minimum', 'Centroid (PCA)']):
        dist_rs = [r[metric] for r in results]
        # Note: distance correlations are negative (closer = more confusion)
        # So we compare |linking_r| vs |distance_r|

        linking_abs = np.abs(linking_rs)
        dist_abs = np.abs(dist_rs)

        # Paired t-test on absolute correlations
        t_stat, p_val = stats.ttest_rel(linking_abs, dist_abs)

        mean_diff = np.mean(linking_abs) - np.mean(dist_abs)

        print(f"\nLinking vs {name}:")
        print(f"  Mean |r| linking:  {np.mean(linking_abs):.3f}")
        print(f"  Mean |r| distance: {np.mean(dist_abs):.3f}")
        print(f"  Difference:        {mean_diff:+.3f}")
        print(f"  Paired t-test:     t={t_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            winner = "Linking" if mean_diff > 0 else name
            print(f"  Result: {winner} is significantly better predictor (p < 0.05)")
        else:
            print(f"  Result: No significant difference")

    # Save results
    output_file = OUTPUT_DIR / "distance_vs_linking_correlation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'distance_matrices': {
                'centroid': centroid_dist.tolist(),
                'avg_pairwise': avg_dist.tolist(),
                'minimum': min_dist.tolist(),
                'pca_centroid': pca_centroid_dist.tolist()
            },
            'linking_consistency': LINKING_CONSISTENCY.tolist()
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Generate LaTeX table
    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)

    print(r"""
\begin{table}[h]
\centering
\caption{Comparison of confusion predictors: inter-class distance vs linking consistency.
Distance metrics show negative correlation (closer $\to$ more confusion),
while linking shows positive correlation (more linked $\to$ more confusion).
Values are mean Spearman $r$ across all 42 models.}
\label{tab:distance-vs-linking}
\small
\begin{tabular}{lccl}
\toprule
Predictor & Mean $r$ & Mean $|r|$ & Interpretation \\
\midrule""")

    for metric, name in zip(metrics, metric_names):
        vals = [r[metric] for r in results]
        mean_r = np.mean(vals)
        mean_abs_r = np.mean(np.abs(vals))

        if 'linking' in metric:
            interp = "linked $\\to$ confused"
        else:
            interp = "close $\\to$ confused"

        print(f"{name} & {mean_r:+.3f} & {mean_abs_r:.3f} & {interp} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
