#!/usr/bin/env python3
"""
Step 4: Compute per-image distances and save for post-hoc analysis.
Also computes aggregate subset statistics and permutation tests.
"""

import os
import sys
import json
import numpy as np
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'witness_eval')
DEFAULT_DATASET = str(PROJECT_ROOT / 'data' / 'cifar10_bird_deer_40x.npz')


def eval_subsets(dataset_path, output_dir=DEFAULT_OUTPUT_DIR):
    print("=" * 60)
    print("Witness Subset Evaluation")
    print("=" * 60)

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    pca_3d = data['pca_3d']
    labels = data['labels']

    # Load link detection result
    link_path = os.path.join(output_dir, 'link_detection_result.json')
    with open(link_path) as f:
        link = json.load(f)

    # Support both single-link and multi-link format
    if 'all_cycle_global_indices' in link:
        # Multi-link format
        eps = link['working_eps']
        cycle_idx = np.array(link['all_cycle_global_indices'])
        bird_centroid = np.array(link['bird_centroid_3d'])
        deer_centroid = np.array(link['deer_centroid_3d'])
        n_links = link['n_links_found']
        print(f"{n_links} linked pairs found, {len(cycle_idx)} unique points on links")
    else:
        # Single-link format (backward compat)
        eps = link['eps']
        bird_cycle_idx = np.array(link['bird_cycle_global_indices'])
        deer_cycle_idx = np.array(link['deer_cycle_global_indices'])
        cycle_idx = np.concatenate([bird_cycle_idx, deer_cycle_idx])
        bird_centroid = np.array(link['bird_centroid_3d'])
        deer_centroid = np.array(link['deer_centroid_3d'])
        n_links = 1

    cycle_idx = np.unique(cycle_idx)
    cycle_pts_3d = pca_3d[cycle_idx]
    print(f"Unique witness points: {len(cycle_idx)}, eps = {eps:.4f}")

    # ---- Per-image distances ----
    print("\nComputing per-image distances to witness cycle...")
    n = len(pca_3d)
    chunk_size = 10000
    min_dist_to_cycle = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = cdist(pca_3d[start:end], cycle_pts_3d).min(axis=1)
        min_dist_to_cycle[start:end] = dists

    dist_to_bird_centroid = np.linalg.norm(pca_3d - bird_centroid, axis=1).astype(np.float32)
    dist_to_deer_centroid = np.linalg.norm(pca_3d - deer_centroid, axis=1).astype(np.float32)
    dist_to_nearest_centroid = np.minimum(dist_to_bird_centroid, dist_to_deer_centroid)

    on_link_mask = np.isin(np.arange(n), cycle_idx)

    # Save per-image distance data (for post-hoc analysis)
    dist_path = os.path.join(output_dir, 'per_image_distances.npz')
    np.savez_compressed(dist_path,
        min_dist_to_cycle=min_dist_to_cycle,
        dist_to_bird_centroid=dist_to_bird_centroid,
        dist_to_deer_centroid=dist_to_deer_centroid,
        dist_to_nearest_centroid=dist_to_nearest_centroid,
        on_link_mask=on_link_mask,
        labels=labels,
        eps=np.float32(eps),
    )
    print(f"Saved per-image distances to {dist_path}")

    # ---- Aggregate subset statistics ----
    radii_in_eps = [0, 1, 3, 5, 10, 20, 50, 100]
    subsets = {}
    for r in radii_in_eps:
        if r == 0:
            subsets[f'on_link'] = cycle_idx
        else:
            mask = min_dist_to_cycle <= r * eps
            subsets[f'within_{r}eps'] = np.where(mask)[0]
    subsets['all'] = np.arange(n)

    for name, idx in subsets.items():
        n_bird = (labels[idx] == 0).sum()
        n_deer = (labels[idx] == 1).sum()
        print(f"  {name}: {len(idx)} points ({n_bird} bird, {n_deer} deer)")

    # ---- Load predictions and compute stats ----
    all_results = {}
    for act in ['relu', 'gelu']:
        pred_path = os.path.join(output_dir, f'{act}_L8_noskip', 'predictions.npz')
        if not os.path.exists(pred_path):
            print(f"\n  WARNING: {pred_path} not found, skipping {act}")
            continue

        pred_data = np.load(pred_path)
        preds = pred_data['preds']
        correct = pred_data['correct']
        probs = pred_data['probs']

        print(f"\n=== {act.upper()} ===")
        act_results = {}
        for name, idx in subsets.items():
            if len(idx) == 0:
                continue
            acc = correct[idx].mean()
            bird_mask = labels[idx] == 0
            deer_mask = labels[idx] == 1
            bird_acc = correct[idx][bird_mask].mean() if bird_mask.any() else None
            deer_acc = correct[idx][deer_mask].mean() if deer_mask.any() else None
            correct_probs = np.array([probs[i, labels[i]] for i in idx])
            mean_conf = correct_probs.mean()

            bal_acc = np.mean([x for x in [bird_acc, deer_acc] if x is not None])
            act_results[name] = {
                'acc': float(acc), 'bal_acc': float(bal_acc), 'n': int(len(idx)),
                'bird_acc': float(bird_acc) if bird_acc is not None else None,
                'deer_acc': float(deer_acc) if deer_acc is not None else None,
                'mean_confidence': float(mean_conf),
            }
            bird_str = f"{bird_acc:.4f}" if bird_acc is not None else "N/A"
            deer_str = f"{deer_acc:.4f}" if deer_acc is not None else "N/A"
            print(f"  {name:20s}: acc={acc:.4f} bal={bal_acc:.4f} n={len(idx):6d} bird={bird_str} deer={deer_str} conf={mean_conf:.4f}")

        all_results[act] = act_results

    # ---- Accuracy vs distance profile (binned) ----
    print("\n=== ACCURACY vs DISTANCE PROFILE ===")
    dist_bins = [0, 1, 3, 5, 10, 20, 50, 100, 500, np.inf]
    for act in ['relu', 'gelu']:
        pred_path = os.path.join(output_dir, f'{act}_L8_noskip', 'predictions.npz')
        if not os.path.exists(pred_path):
            continue
        correct = np.load(pred_path)['correct']
        print(f"\n{act.upper()} accuracy by distance (in eps units):")
        for i in range(len(dist_bins) - 1):
            lo, hi = dist_bins[i] * eps, dist_bins[i+1] * eps
            mask = (min_dist_to_cycle >= lo) & (min_dist_to_cycle < hi)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            print(f"  [{dist_bins[i]:6.0f}, {dist_bins[i+1]:6.0f}) eps: acc={acc:.4f} n={mask.sum()}")

    # ---- Permutation test ----
    print("\n=== PERMUTATION TEST (gap near link vs random) ===")
    if 'relu' in all_results and 'gelu' in all_results:
        relu_correct = np.load(os.path.join(output_dir, 'relu_L8_noskip', 'predictions.npz'))['correct']
        gelu_correct = np.load(os.path.join(output_dir, 'gelu_L8_noskip', 'predictions.npz'))['correct']

        # Observed gap on witness points
        n_witness = len(cycle_idx)
        obs_gap = gelu_correct[cycle_idx].mean() - relu_correct[cycle_idx].mean()
        print(f"Observed gap on witness points: {obs_gap:+.4f} (n={n_witness})")

        # Observed gap within 10eps
        near_10 = np.where(min_dist_to_cycle <= 10 * eps)[0]
        obs_gap_10 = gelu_correct[near_10].mean() - relu_correct[near_10].mean()
        print(f"Observed gap within 10eps: {obs_gap_10:+.4f} (n={len(near_10)})")

        # Permutation: random subsets of same size
        np.random.seed(42)
        n_perms = 1000
        random_gaps = np.zeros(n_perms)
        random_gaps_10 = np.zeros(n_perms)
        for p in range(n_perms):
            rand_idx = np.random.choice(n, n_witness, replace=False)
            random_gaps[p] = gelu_correct[rand_idx].mean() - relu_correct[rand_idx].mean()
            rand_idx_10 = np.random.choice(n, len(near_10), replace=False)
            random_gaps_10[p] = gelu_correct[rand_idx_10].mean() - relu_correct[rand_idx_10].mean()

        p_value = (random_gaps >= obs_gap).mean()
        p_value_10 = (random_gaps_10 >= obs_gap_10).mean()
        print(f"Permutation p-value (witness): {p_value:.4f} (random gap mean={random_gaps.mean():+.4f})")
        print(f"Permutation p-value (10eps): {p_value_10:.4f} (random gap mean={random_gaps_10.mean():+.4f})")

    # ---- Test set: partition by distance to link ----
    print("\n=== TEST SET (unaugmented) — partitioned by distance ===")
    test_pca_3d = data['test_pca_3d']
    test_labels = data['test_labels']
    n_test = len(test_pca_3d)

    # Compute test-point distances to witness cycle in 3D
    test_dist_to_cycle = np.full(n_test, np.inf, dtype=np.float32)
    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        dists = cdist(test_pca_3d[start:end], cycle_pts_3d).min(axis=1)
        test_dist_to_cycle[start:end] = dists

    test_subsets = {}
    for r in radii_in_eps:
        if r == 0:
            continue  # no test points exactly on the training witness cycle
        test_subsets[f'test_within_{r}eps'] = np.where(test_dist_to_cycle <= r * eps)[0]
    test_subsets['test_all'] = np.arange(n_test)
    test_subsets['test_far'] = np.where(test_dist_to_cycle > 50 * eps)[0]

    for name, idx in test_subsets.items():
        n_bird = (test_labels[idx] == 0).sum()
        n_deer = (test_labels[idx] == 1).sum()
        print(f"  {name}: {len(idx)} pts ({n_bird} bird, {n_deer} deer)")

    test_results = {}
    for act in ['relu', 'gelu']:
        test_path = os.path.join(output_dir, f'{act}_L8_noskip', 'test_predictions.npz')
        if not os.path.exists(test_path):
            continue
        td = np.load(test_path)
        test_preds = td['preds']
        test_correct = (test_preds == test_labels).astype(np.int8)

        print(f"\n  {act.upper()}:")
        act_test = {}
        for name, idx in test_subsets.items():
            if len(idx) == 0:
                continue
            acc = test_correct[idx].mean()
            bird_mask = test_labels[idx] == 0
            deer_mask = test_labels[idx] == 1
            bird_acc = test_correct[idx][bird_mask].mean() if bird_mask.any() else None
            deer_acc = test_correct[idx][deer_mask].mean() if deer_mask.any() else None
            bal_acc = np.mean([x for x in [bird_acc, deer_acc] if x is not None])
            act_test[name] = {'acc': float(acc), 'bal_acc': float(bal_acc), 'n': int(len(idx)),
                              'bird_acc': float(bird_acc) if bird_acc is not None else None,
                              'deer_acc': float(deer_acc) if deer_acc is not None else None}
            print(f"    {name:25s}: acc={acc:.4f} bal={bal_acc:.4f} n={len(idx)}")
        test_results[act] = act_test

    # Test set gaps
    if 'relu' in test_results and 'gelu' in test_results:
        print(f"\n  TEST GAPS (GELU - ReLU):")
        for name in test_subsets:
            if name in test_results['relu'] and name in test_results['gelu']:
                r_acc = test_results['relu'][name]['bal_acc']
                g_acc = test_results['gelu'][name]['bal_acc']
                print(f"    {name:25s}: {g_acc - r_acc:+.4f} (balanced)")

    # ---- Save everything ----
    output = {
        'train_per_activation': all_results,
        'test_per_activation': test_results,
        'link_info': link,
    }
    save_path = os.path.join(output_dir, 'subset_comparison.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', default=os.environ.get('WITNESS_OUTPUT_DIR', DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    eval_subsets(args.dataset, args.output_dir)
