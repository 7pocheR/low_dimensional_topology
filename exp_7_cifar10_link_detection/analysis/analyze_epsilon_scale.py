#!/usr/bin/env python3
"""Analyze epsilon relative to data distribution."""
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / 'data' / 'cifar10_aug20x_enhanced_pca3d.npz'

# Load the 20x PCA data
data = np.load(DATA_FILE, allow_pickle=True)

# Combine all classes
all_points = np.vstack([data[f'class_{i}'] for i in range(10)])
print(f'Total points: {len(all_points)}')

# Per-dimension statistics
print('\n=== Per-Dimension Statistics ===')
for dim in range(3):
    vals = all_points[:, dim]
    rng = vals.max() - vals.min()
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    print(f'Dim {dim}: Range={rng:.3f}, IQR={iqr:.3f}')

# 3D bounding box diagonal
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)
diagonal = np.sqrt(((maxs - mins)**2).sum())
print(f'\n3D Diagonal (max distance): {diagonal:.4f}')

# Sample pairwise distances
np.random.seed(42)
n = 100000
idx1 = np.random.randint(0, len(all_points), n)
idx2 = np.random.randint(0, len(all_points), n)
dists = np.sqrt(((all_points[idx1] - all_points[idx2])**2).sum(axis=1))

q1, q3 = np.percentile(dists, [25, 75])
iqr = q3 - q1
print(f'\n=== Pairwise Distances ===')
print(f'Range: [{dists.min():.4f}, {dists.max():.4f}]')
print(f'Mean: {dists.mean():.4f}, Median: {np.median(dists):.4f}')
print(f'IQR: {iqr:.4f}')

eps = 0.0338
print(f'\n=== Epsilon = {eps} ===')
print(f'eps / 3D Range = {eps/diagonal:.4f} = {eps/diagonal*100:.2f}%')
print(f'eps / Distance IQR = {eps/iqr:.4f} = {eps/iqr*100:.2f}%')
print(f'eps / Mean dist = {eps/dists.mean():.4f} = {eps/dists.mean()*100:.2f}%')
print(f'Pairs within eps: {(dists < eps).mean()*100:.4f}%')
