#!/usr/bin/env python3
"""Aggregate shard results from eps_min_all_pairs_augmented.py and compute correlations."""
import os, json, glob, csv
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = os.environ.get('RESULTS_DIR', str(PROJECT_ROOT / 'results' / 'eps_min_augmented'))

# Merge all shard files
merged = {}
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, 'eps_min_augmented*.json'))):
    merged.update(json.load(open(f)))

print(f"Merged {len(merged)} pairs from {len(glob.glob(os.path.join(RESULTS_DIR, 'eps_min_augmented*.json')))} files")

# Save merged
with open(os.path.join(RESULTS_DIR, 'eps_min_augmented_merged.json'), 'w') as f:
    json.dump(merged, f, indent=2)

# Print all pairs sorted by eps_min
pairs_sorted = sorted(merged.items(), key=lambda x: x[1] if x[1] else 999)
print(f"\n{'Pair':<25} {'eps_min':>10}")
print('-' * 37)
for name, val in pairs_sorted:
    print(f"{name:<25} {val:.4f}" if val else f"{name:<25} {'N/A':>10}")

# Compute correlations
pair_order = list(combinations(range(10), 2))
eps_vals = []
for i, j in pair_order:
    name = f"{CLASSES[i]}-{CLASSES[j]}"
    val = merged.get(name)
    eps_vals.append(1.0 / val if val and val > 0 else 0)
eps_vals = np.array(eps_vals)

# Load linking consistency
lc = np.zeros((10, 10))
lc_file = os.environ.get('LINKING_CONSISTENCY_CSV', str(PROJECT_ROOT / 'results' / 'linking_consistency.csv'))
if os.path.exists(lc_file):
    with open(lc_file) as f:
        reader = csv.reader(f)
        next(reader)
        for idx, row in enumerate(reader):
            for jdx, val in enumerate(row[1:]):
                if val: lc[idx][jdx] = float(val)
    lc_vals = np.array([lc[i][j] for i, j in pair_order])

    r_el, p_el = spearmanr(eps_vals, lc_vals)
    print(f"\nSpearman(1/ε_min, consistency): r = {r_el:.3f}, p = {p_el:.4f}")

# Load CNN confusion
cnn_files = sorted(glob.glob(str(PROJECT_ROOT / 'results' / 'cnn_10class' / 'result_cnn10_*.json')))
if cnn_files:
    eps_rs, lc_rs = [], []
    for rf in cnn_files:
        r = json.load(open(rf))
        cm = np.array(r['confusion_matrix'])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = cm / row_sums
        conf = np.array([norm[i][j] + norm[j][i] for i, j in pair_order])
        re, _ = spearmanr(eps_vals, conf)
        rl, _ = spearmanr(lc_vals, conf)
        eps_rs.append(abs(re))
        lc_rs.append(abs(rl))

    print(f"\nCorrelation with confusion ({len(cnn_files)} CNN models):")
    print(f"  1/ε_min (augmented) vs confusion: mean |r| = {np.mean(eps_rs):.3f}")
    print(f"  Linking consistency vs confusion:  mean |r| = {np.mean(lc_rs):.3f}")
