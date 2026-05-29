#!/usr/bin/env python3
"""
Distance vs Linking Analysis using the CORRECT linking matrix from results.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('CIFAR10_DATA_DIR', PROJECT_ROOT / 'data'))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', PROJECT_ROOT / 'results' / 'cnn_10class'))
ANALYSIS_DIR = Path(os.environ.get('ANALYSIS_DIR', PROJECT_ROOT / 'results' / 'analysis'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', PROJECT_ROOT / 'results' / 'distance_analysis'))

print('='*70)
print('Distance vs Linking Analysis (using correct linking matrix)')
print('='*70)

# Load CIFAR-10
print('\nLoading CIFAR-10...')
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True, download=True, transform=transform
)
X = np.array([img.numpy().flatten() for img, _ in trainset])
y = np.array([label for _, label in trainset])

# Split by class
class_data = {c: X[y == c] for c in range(10)}
print(f'Loaded {len(X)} samples')

# Load correct linking matrix
link_df = pd.read_csv(ANALYSIS_DIR / 'linking_consistency.csv', index_col=0)
LINKING = link_df.values
print('Loaded linking consistency matrix')

# Compute distance metrics
print('\nComputing distance metrics...')

# 1. Centroid distances
print('  - Centroid distances...')
centroids = np.array([class_data[c].mean(axis=0) for c in range(10)])
centroid_dist = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        centroid_dist[i, j] = np.linalg.norm(centroids[i] - centroids[j])

# 2. Average pairwise distances (subsampled)
print('  - Average pairwise distances...')
np.random.seed(42)
avg_dist = np.zeros((10, 10))
for i in range(10):
    for j in range(i+1, 10):
        idx_i = np.random.choice(len(class_data[i]), 500, replace=False)
        idx_j = np.random.choice(len(class_data[j]), 500, replace=False)
        dists = cdist(class_data[i][idx_i], class_data[j][idx_j])
        avg_dist[i, j] = avg_dist[j, i] = dists.mean()

# 3. Minimum distances (subsampled)
print('  - Minimum distances...')
min_dist = np.zeros((10, 10))
for i in range(10):
    for j in range(i+1, 10):
        idx_i = np.random.choice(len(class_data[i]), 1000, replace=False)
        idx_j = np.random.choice(len(class_data[j]), 1000, replace=False)
        dists = cdist(class_data[i][idx_i], class_data[j][idx_j])
        min_dist[i, j] = min_dist[j, i] = dists.min()

# Convert to pairs
def to_pairs(m):
    return [m[i,j] for i in range(10) for j in range(i+1, 10)]

centroid_pairs = np.array(to_pairs(centroid_dist))
avg_pairs = np.array(to_pairs(avg_dist))
min_pairs = np.array(to_pairs(min_dist))
link_pairs = np.array(to_pairs(LINKING))

# Load all confusion matrices and compute correlations
print('\nLoading confusion matrices and computing correlations...')
results_dir = RESULTS_DIR
results = []

for f in sorted(results_dir.glob('result_*.json')):
    with open(f) as fp:
        data = json.load(fp)

    cm = np.array(data['confusion_matrix'])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = cm / row_sums

    conf_pairs = np.array([norm[i,j] + norm[j,i] for i in range(10) for j in range(i+1, 10)])

    r_cent, p_cent = stats.spearmanr(centroid_pairs, conf_pairs)
    r_avg, p_avg = stats.spearmanr(avg_pairs, conf_pairs)
    r_min, p_min = stats.spearmanr(min_pairs, conf_pairs)
    r_link, p_link = stats.spearmanr(link_pairs, conf_pairs)

    results.append({
        'model': f.stem,
        'r_centroid': r_cent,
        'r_avg': r_avg,
        'r_min': r_min,
        'r_linking': r_link,
        'p_linking': p_link
    })

print(f'Processed {len(results)} models')

# Print individual results
print('\n' + '='*100)
header = f"{'Model':<30} {'Centroid r':>12} {'Avg r':>12} {'Min r':>12} {'Linking r':>12}"
print(header)
print('='*100)

for r in results:
    line = f"{r['model']:<30} {r['r_centroid']:>+12.3f} {r['r_avg']:>+12.3f} {r['r_min']:>+12.3f} {r['r_linking']:>+12.3f}"
    print(line)

# Summary
print('\n' + '='*70)
print('SUMMARY (mean across all models)')
print('='*70)

metrics = ['r_centroid', 'r_avg', 'r_min', 'r_linking']
names = ['Centroid distance', 'Avg pairwise distance', 'Minimum distance', 'Linking consistency']

print(f"\n{'Metric':<25} {'Mean r':>12} {'Mean |r|':>12}")
print('-'*55)
for m, n in zip(metrics, names):
    vals = [r[m] for r in results]
    mean_r = np.mean(vals)
    mean_abs = np.mean(np.abs(vals))
    print(f'{n:<25} {mean_r:>+12.3f} {mean_abs:>12.3f}')

# Statistical comparison
print('\n' + '='*70)
print('STATISTICAL COMPARISON: Linking vs Distance metrics')
print('='*70)

link_abs = np.abs([r['r_linking'] for r in results])

comparisons = [
    ('r_centroid', 'Centroid distance'),
    ('r_avg', 'Avg pairwise distance'),
    ('r_min', 'Minimum distance')
]

for m, n in comparisons:
    dist_abs = np.abs([r[m] for r in results])
    t, p = stats.ttest_rel(link_abs, dist_abs)
    diff = np.mean(link_abs) - np.mean(dist_abs)

    print(f'\nLinking vs {n}:')
    print(f'  Mean |r| linking:  {np.mean(link_abs):.3f}')
    print(f'  Mean |r| distance: {np.mean(dist_abs):.3f}')
    print(f'  Difference:        {diff:+.3f}')
    print(f'  Paired t-test:     t={t:.2f}, p={p:.6f}')

    if p < 0.05:
        winner = 'Linking' if diff > 0 else n
        print(f'  Result: {winner} is significantly better (p < 0.05)')
    else:
        print(f'  Result: No significant difference')

# Save results
output = {
    'summary': {
        'centroid_mean_r': float(np.mean([r['r_centroid'] for r in results])),
        'centroid_mean_abs_r': float(np.mean(np.abs([r['r_centroid'] for r in results]))),
        'avg_mean_r': float(np.mean([r['r_avg'] for r in results])),
        'avg_mean_abs_r': float(np.mean(np.abs([r['r_avg'] for r in results]))),
        'min_mean_r': float(np.mean([r['r_min'] for r in results])),
        'min_mean_abs_r': float(np.mean(np.abs([r['r_min'] for r in results]))),
        'linking_mean_r': float(np.mean([r['r_linking'] for r in results])),
        'linking_mean_abs_r': float(np.mean(np.abs([r['r_linking'] for r in results]))),
    },
    'per_model': results
}

output_file = OUTPUT_DIR / 'distance_vs_linking_correct.json'
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nResults saved to: {output_file}')
