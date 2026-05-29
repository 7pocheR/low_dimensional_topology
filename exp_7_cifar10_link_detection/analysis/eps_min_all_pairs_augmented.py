#!/usr/bin/env python3
"""
Binary search for minimum epsilon for EACH of the 45 CIFAR-10 class pairs.
Uses 20x AUGMENTED data (matching the consistency computation).
Uses TOP-K longest cycles (K=500) instead of random sampling to avoid
signal dilution at large epsilon.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from link_detector import build_knn_graph, fundamental_cycle_basis, gauss_linking_number_numeric
from cifar10_data_loader import CIFAR10_CLASSES
from cifar10_augmented import create_augmented_pca


def test_linking_at_epsilon_for_pair(points_i, points_j, epsilon, k=15,
                                      min_cycle_length=20, top_k=500,
                                      tolerance=0.1, class_i=0, class_j=1,
                                      verbose=False):
    """Test if linking exists between two point clouds at given epsilon.
    Uses TOP-K longest cycles from each class instead of random sampling."""
    t0 = time.time()

    adj_i = build_knn_graph(points_i, k=k, epsilon=epsilon, mutual=True)
    adj_j = build_knn_graph(points_j, k=k, epsilon=epsilon, mutual=True)
    t_graph = time.time() - t0

    cycles_i = fundamental_cycle_basis(adj_i)
    cycles_j = fundamental_cycle_basis(adj_j)
    t_cycles = time.time() - t0 - t_graph

    # Filter by minimum length
    cycles_i = [c for c in cycles_i if len(c) >= min_cycle_length]
    cycles_j = [c for c in cycles_j if len(c) >= min_cycle_length]

    if not cycles_i or not cycles_j:
        if verbose:
            print(f"      eps={epsilon:.4f}: no valid cycles "
                  f"(raw: {len(fundamental_cycle_basis(adj_i))}/{len(fundamental_cycle_basis(adj_j))}, "
                  f"filtered: {len(cycles_i)}/{len(cycles_j)}) [{t_graph+t_cycles:.1f}s]")
        return False, None, {}

    # Sort by length descending, take top K
    cycles_i.sort(key=len, reverse=True)
    cycles_j.sort(key=len, reverse=True)
    cycles_i = cycles_i[:top_k]
    cycles_j = cycles_j[:top_k]

    # Convert to polylines
    polylines_i = [points_i[np.array(c, dtype=int)] for c in cycles_i]
    polylines_j = [points_j[np.array(c, dtype=int)] for c in cycles_j]

    n_i, n_j = len(polylines_i), len(polylines_j)
    avg_len_i = np.mean([len(c) for c in cycles_i])
    avg_len_j = np.mean([len(c) for c in cycles_j])
    max_len_i = len(cycles_i[0])
    max_len_j = len(cycles_j[0])

    # Test all top_k x top_k pairs
    n_tested = 0
    n_linked = 0
    best_lk = None
    best_pair = None
    t_gauss_start = time.time()

    for idx_i in range(n_i):
        for idx_j in range(n_j):
            gauss = gauss_linking_number_numeric(polylines_i[idx_i], polylines_j[idx_j])
            lk = round(gauss)
            n_tested += 1
            if abs(gauss - lk) < tolerance and lk != 0:
                n_linked += 1
                if best_lk is None:
                    best_lk = lk
                    best_pair = (idx_i, idx_j)
                    # Early exit on first find for binary search efficiency
                    t_total = time.time() - t0
                    stats = {
                        'n_cycles_i': n_i, 'n_cycles_j': n_j,
                        'avg_len_i': avg_len_i, 'avg_len_j': avg_len_j,
                        'max_len_i': max_len_i, 'max_len_j': max_len_j,
                        'n_tested': n_tested, 'n_linked': n_linked,
                        't_graph': t_graph, 't_cycles': t_cycles,
                        't_gauss': time.time() - t_gauss_start,
                        't_total': t_total,
                        'early_exit': True
                    }
                    if verbose:
                        print(f"      eps={epsilon:.4f}: LINKED (lk={lk}) at pair ({idx_i},{idx_j}), "
                              f"tested {n_tested}/{n_i*n_j}, "
                              f"cycles {n_i}×{n_j} (avg len {avg_len_i:.0f}×{avg_len_j:.0f}), "
                              f"[{t_total:.1f}s]")
                    return True, lk, stats

    t_total = time.time() - t0
    stats = {
        'n_cycles_i': n_i, 'n_cycles_j': n_j,
        'avg_len_i': avg_len_i, 'avg_len_j': avg_len_j,
        'max_len_i': max_len_i, 'max_len_j': max_len_j,
        'n_tested': n_tested, 'n_linked': 0,
        't_graph': t_graph, 't_cycles': t_cycles,
        't_gauss': time.time() - t_gauss_start,
        't_total': t_total,
        'early_exit': False
    }
    if verbose:
        print(f"      eps={epsilon:.4f}: not found, "
              f"tested {n_tested}/{n_i*n_j}, "
              f"cycles {n_i}×{n_j} (avg len {avg_len_i:.0f}×{avg_len_j:.0f}, "
              f"max {max_len_i}×{max_len_j}), "
              f"[{t_total:.1f}s]")
    return False, None, stats


def binary_search_pair(points_i, points_j, eps_start=0.05, eps_cap=1.0,
                        precision=0.005, k=15, min_cycle_length=20,
                        top_k=500, class_i=0, class_j=1, pair_name=""):
    """Find minimum epsilon where linking exists using exponential probe + binary search.

    Phase 1 (exponential probe): Start at eps_start, multiply by 2 until linking
    is found or eps_cap is reached.

    Phase 2 (binary search): Once found at some eps, search [0, eps] to refine.
    Uses [0, eps] not [eps/2, eps] because detection is stochastic.
    """
    print(f"    [{pair_name}] Phase 1: exponential probe from {eps_start} (cap={eps_cap})")
    t_pair_start = time.time()

    # Phase 1: exponential probe upward
    eps = eps_start
    found_eps = None
    probe_history = []
    while eps <= eps_cap:
        found, lk, stats = test_linking_at_epsilon_for_pair(
            points_i, points_j, eps, k, min_cycle_length, top_k,
            class_i=class_i, class_j=class_j, verbose=True)
        probe_history.append({'eps': eps, 'found': found, 'stats': stats})
        if found:
            found_eps = eps
            break
        eps *= 2

    if found_eps is None:
        t_total = time.time() - t_pair_start
        print(f"    [{pair_name}] NO LINKING FOUND (probed up to {min(eps, eps_cap):.3f}) [{t_total:.1f}s]")
        return None, {'phase1': probe_history, 'phase2': []}

    # Phase 2: binary search in [0, found_eps]
    print(f"    [{pair_name}] Phase 2: binary search in [0, {found_eps:.4f}]")
    eps_low = 0.0
    eps_high = found_eps
    search_history = []
    iteration = 0
    while eps_high - eps_low > precision:
        iteration += 1
        eps_mid = (eps_low + eps_high) / 2
        found, lk, stats = test_linking_at_epsilon_for_pair(
            points_i, points_j, eps_mid, k, min_cycle_length, top_k,
            class_i=class_i, class_j=class_j, verbose=True)
        search_history.append({'eps': eps_mid, 'found': found, 'iteration': iteration, 'stats': stats})
        if found:
            eps_high = eps_mid
        else:
            eps_low = eps_mid

    t_total = time.time() - t_pair_start
    print(f"    [{pair_name}] RESULT: ε_min = {eps_high:.4f} "
          f"(range [{eps_low:.4f}, {eps_high:.4f}], {iteration} iterations) [{t_total:.1f}s]")
    return eps_high, {'phase1': probe_history, 'phase2': search_history}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ε_min for all 45 pairs (augmented data)')
    parser.add_argument('--n-aug', type=int, default=20)
    parser.add_argument('--eps-start', type=float, default=0.05, help='Initial probe epsilon')
    parser.add_argument('--eps-cap', type=float, default=1.0, help='Maximum epsilon to try')
    parser.add_argument('--precision', type=float, default=0.005)
    parser.add_argument('--top-k', type=int, default=500, help='Use top-K longest cycles')
    parser.add_argument('--pair-start', type=int, default=0)
    parser.add_argument('--pair-end', type=int, default=45)
    parser.add_argument('--pair-indices', type=str, default=None,
                        help='Comma-separated pair indices (overrides start/end)')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'eps_min_augmented'))
    parser.add_argument('--data-dir', type=str,
                        default=str(PROJECT_ROOT / 'data'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the SAME cached augmented PCA file used by consistency runs
    aug_pca_file = os.path.join(args.data_dir, 'cifar10_aug20x_enhanced_pca3d.npz')
    if os.path.exists(aug_pca_file):
        print(f"Loading cached augmented PCA: {aug_pca_file}")
        raw = np.load(aug_pca_file, allow_pickle=True)
        keys = sorted(raw.keys())
        pca_data = {}
        for idx, cls in enumerate(CIFAR10_CLASSES):
            key = f'class_{idx}' if f'class_{idx}' in keys else cls
            pca_data[cls] = raw[key]
        total = sum(len(v) for v in pca_data.values())
        print(f"Total samples: {total}, per class: {total // 10}")
    else:
        print(f"Cached file not found, recomputing {args.n_aug}x augmented PCA...")
        pca_data, metadata = create_augmented_pca(
            n_augmentations=args.n_aug,
            data_dir=args.data_dir
        )
        print(f"Total samples: {metadata['total_samples']}")

    print(f"\nε_min search (top-{args.top_k} longest cycles)")
    print(f"Parameters: k=15, mutual=True, min_cycle_length=20, top_k={args.top_k}")
    print(f"Search: exponential probe from {args.eps_start}, cap={args.eps_cap}, precision={args.precision}")

    all_pairs = list(combinations(range(10), 2))
    if args.pair_indices:
        indices = [int(x) for x in args.pair_indices.split(',')]
        my_pairs = [all_pairs[i] for i in indices]
        shard_label = f"custom_{len(indices)}"
        print(f"Processing {len(my_pairs)} pairs at indices: {indices}")
    else:
        my_pairs = all_pairs[args.pair_start:args.pair_end]
        shard_label = f"p{args.pair_start}-{args.pair_end}"
        print(f"Processing pairs {args.pair_start} to {args.pair_end-1} ({len(my_pairs)} pairs)")
    print()

    results = {}
    all_histories = {}
    for pair_idx, (i, j) in enumerate(my_pairs):
        pair_name = f"{CIFAR10_CLASSES[i]}-{CIFAR10_CLASSES[j]}"
        print(f"  === Pair {pair_idx+1}/{len(my_pairs)}: {pair_name} ===")
        points_i = pca_data[CIFAR10_CLASSES[i]]
        points_j = pca_data[CIFAR10_CLASSES[j]]

        eps_min, history = binary_search_pair(
            points_i, points_j,
            eps_start=args.eps_start,
            eps_cap=args.eps_cap,
            precision=args.precision,
            top_k=args.top_k,
            class_i=i, class_j=j,
            pair_name=pair_name
        )

        results[pair_name] = eps_min
        # Save stats but not full numpy arrays
        all_histories[pair_name] = {
            'eps_min': eps_min,
            'n_phase1_probes': len(history['phase1']),
            'n_phase2_iters': len(history['phase2']),
            'phase1_eps': [h['eps'] for h in history['phase1']],
            'phase1_found': [h['found'] for h in history['phase1']],
            'phase2_eps': [h['eps'] for h in history['phase2']],
            'phase2_found': [h['found'] for h in history['phase2']],
        }

        # Save intermediate results after each pair
        shard_tag = f"_{shard_label}" if len(my_pairs) < 45 else ""
        output_file = os.path.join(args.output_dir, f'eps_min_augmented{shard_tag}.json')
        with open(output_file, 'w') as f:
            json.dump({'results': results, 'histories': all_histories}, f, indent=2)
        print(f"    [saved intermediate results to {output_file}]")
        print()

    # Final summary
    print("=" * 60)
    print("SHARD SUMMARY")
    print("=" * 60)
    for name, val in results.items():
        status = f"ε_min = {val:.4f}" if val is not None else "no linking"
        print(f"  {name:<25} {status}")

    # Compute correlation with confusion (only for full runs)
    import glob
    cnn_files = sorted(glob.glob(str(PROJECT_ROOT / 'results' / 'cnn_10class' / 'result_cnn10_*.json')))
    if cnn_files and len(results) == 45:
        pair_order = list(combinations(range(10), 2))
        eps_vals = []
        for i, j in pair_order:
            name = f"{CIFAR10_CLASSES[i]}-{CIFAR10_CLASSES[j]}"
            val = results.get(name)
            eps_vals.append(1.0 / val if val and val > 0 else 0)
        eps_vals = np.array(eps_vals)

        import csv
        lc = np.zeros((10, 10))
        with open(PROJECT_ROOT / 'results' / 'linking_consistency.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for idx, row in enumerate(reader):
                for jdx, val in enumerate(row[1:]):
                    if val:
                        lc[idx][jdx] = float(val)
        lc_vals = np.array([lc[i][j] for i, j in pair_order])

        print(f"\nCorrelation with confusion ({len(cnn_files)} CNN models):")
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

        print(f"  1/ε_min (augmented) vs confusion: mean |r| = {np.mean(eps_rs):.3f}")
        print(f"  Linking consistency vs confusion:  mean |r| = {np.mean(lc_rs):.3f}")
        print(f"  Spearman(1/ε_min, consistency):    r = {spearmanr(eps_vals, lc_vals)[0]:.3f}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
