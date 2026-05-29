#!/usr/bin/env python3
"""
Refine ε_min by continuing binary search from previous results.
Loads existing ε_min values as upper bounds, searches [0, ε_min] with finer precision.
"""

import os
import sys
import json
import time
import glob
import numpy as np
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from link_detector import build_knn_graph, fundamental_cycle_basis, gauss_linking_number_numeric
from cifar10_data_loader import CIFAR10_CLASSES


def test_linking_at_epsilon(points_i, points_j, epsilon, k=15,
                             min_cycle_length=20, top_k=500,
                             tolerance=0.1, class_i=0, class_j=1,
                             verbose=False):
    """Test if linking exists at given epsilon using top-K longest cycles."""
    t0 = time.time()

    adj_i = build_knn_graph(points_i, k=k, epsilon=epsilon, mutual=True)
    adj_j = build_knn_graph(points_j, k=k, epsilon=epsilon, mutual=True)

    cycles_i = fundamental_cycle_basis(adj_i)
    cycles_j = fundamental_cycle_basis(adj_j)

    cycles_i = [c for c in cycles_i if len(c) >= min_cycle_length]
    cycles_j = [c for c in cycles_j if len(c) >= min_cycle_length]

    if not cycles_i or not cycles_j:
        if verbose:
            print(f"      eps={epsilon:.5f}: no valid cycles ({time.time()-t0:.1f}s)")
        return False

    cycles_i.sort(key=len, reverse=True)
    cycles_j.sort(key=len, reverse=True)
    cycles_i = cycles_i[:top_k]
    cycles_j = cycles_j[:top_k]

    polylines_i = [points_i[np.array(c, dtype=int)] for c in cycles_i]
    polylines_j = [points_j[np.array(c, dtype=int)] for c in cycles_j]

    n_i, n_j = len(polylines_i), len(polylines_j)
    n_tested = 0

    for idx_i in range(n_i):
        for idx_j in range(n_j):
            gauss = gauss_linking_number_numeric(polylines_i[idx_i], polylines_j[idx_j])
            lk = round(gauss)
            n_tested += 1
            if abs(gauss - lk) < tolerance and lk != 0:
                if verbose:
                    print(f"      eps={epsilon:.5f}: LINKED (lk={lk}) pair ({idx_i},{idx_j}), "
                          f"tested {n_tested}/{n_i*n_j}, cycles {n_i}x{n_j} ({time.time()-t0:.1f}s)")
                return True

    if verbose:
        print(f"      eps={epsilon:.5f}: not found, "
              f"tested {n_tested}/{n_i*n_j}, cycles {n_i}x{n_j} ({time.time()-t0:.1f}s)")
    return False


def refine_pair(points_i, points_j, prev_eps_min, precision=0.001,
                k=15, min_cycle_length=20, top_k=500,
                class_i=0, class_j=1, pair_name=""):
    """Refine ε_min by binary search in [0, prev_eps_min]."""
    print(f"    [{pair_name}] Refining from prev ε_min={prev_eps_min:.4f}, precision={precision}")
    t0 = time.time()

    # Verify linking still exists at prev_eps_min
    found = test_linking_at_epsilon(
        points_i, points_j, prev_eps_min, k, min_cycle_length, top_k,
        class_i=class_i, class_j=class_j, verbose=True)

    if not found:
        print(f"    [{pair_name}] WARNING: linking not reproduced at prev ε_min={prev_eps_min:.4f}")
        print(f"    [{pair_name}] Trying 1.5x = {prev_eps_min*1.5:.4f}")
        found = test_linking_at_epsilon(
            points_i, points_j, prev_eps_min * 1.5, k, min_cycle_length, top_k,
            class_i=class_i, class_j=class_j, verbose=True)
        if found:
            prev_eps_min = prev_eps_min * 1.5
            print(f"    [{pair_name}] Found at {prev_eps_min:.4f}, using as upper bound")
        else:
            print(f"    [{pair_name}] FAILED — skipping")
            return None, []

    # Binary search in [0, prev_eps_min]
    eps_low = 0.0
    eps_high = prev_eps_min
    history = []
    iteration = 0

    while eps_high - eps_low > precision:
        iteration += 1
        eps_mid = (eps_low + eps_high) / 2
        found = test_linking_at_epsilon(
            points_i, points_j, eps_mid, k, min_cycle_length, top_k,
            class_i=class_i, class_j=class_j, verbose=True)
        history.append({'eps': eps_mid, 'found': found, 'iteration': iteration})
        if found:
            eps_high = eps_mid
        else:
            eps_low = eps_mid

    t_total = time.time() - t0
    print(f"    [{pair_name}] RESULT: ε_min = {eps_high:.5f} "
          f"(range [{eps_low:.5f}, {eps_high:.5f}], {iteration} iters) [{t_total:.1f}s]")
    return eps_high, history


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Refine ε_min from previous results')
    parser.add_argument('--precision', type=float, default=0.001)
    parser.add_argument('--top-k', type=int, default=500)
    parser.add_argument('--pair-indices', type=str, default=None,
                        help='Comma-separated pair indices (default: all with existing results)')
    parser.add_argument('--results-dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'eps_min_augmented'))
    parser.add_argument('--data-dir', type=str,
                        default=str(PROJECT_ROOT / 'data'))
    args = parser.parse_args()

    # Load augmented PCA data
    aug_pca_file = os.path.join(args.data_dir, 'cifar10_aug20x_enhanced_pca3d.npz')
    print(f"Loading: {aug_pca_file}")
    raw = np.load(aug_pca_file, allow_pickle=True)
    pca_data = {}
    for idx, cls in enumerate(CIFAR10_CLASSES):
        key = f'class_{idx}' if f'class_{idx}' in sorted(raw.keys()) else cls
        pca_data[cls] = raw[key]
    print(f"Total samples: {sum(len(v) for v in pca_data.values())}")

    # Load previous results
    prev_results = {}
    for f in glob.glob(os.path.join(args.results_dir, 'eps_min_augmented*.json')):
        data = json.load(open(f))
        r = data.get('results', data)
        prev_results.update(r)
    print(f"Loaded {len(prev_results)} previous results")

    # Determine which pairs to refine
    all_pairs = list(combinations(range(10), 2))
    if args.pair_indices:
        indices = [int(x) for x in args.pair_indices.split(',')]
        my_pairs = [all_pairs[i] for i in indices]
    else:
        my_pairs = [p for p in all_pairs
                    if f"{CIFAR10_CLASSES[p[0]]}-{CIFAR10_CLASSES[p[1]]}" in prev_results
                    and prev_results[f"{CIFAR10_CLASSES[p[0]]}-{CIFAR10_CLASSES[p[1]]}"] is not None]

    print(f"Refining {len(my_pairs)} pairs with precision={args.precision}")
    print()

    results = {}
    for pair_idx, (i, j) in enumerate(my_pairs):
        pair_name = f"{CIFAR10_CLASSES[i]}-{CIFAR10_CLASSES[j]}"
        prev_val = prev_results.get(pair_name)
        if prev_val is None:
            print(f"  === Pair {pair_idx+1}/{len(my_pairs)}: {pair_name} — no previous result, skipping ===")
            continue

        print(f"  === Pair {pair_idx+1}/{len(my_pairs)}: {pair_name} (prev={prev_val:.4f}) ===")
        eps_min, history = refine_pair(
            pca_data[CIFAR10_CLASSES[i]], pca_data[CIFAR10_CLASSES[j]],
            prev_val, precision=args.precision, top_k=args.top_k,
            class_i=i, class_j=j, pair_name=pair_name
        )
        results[pair_name] = eps_min

        # Save intermediate
        output_file = os.path.join(args.results_dir, 'eps_min_refined.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"    [saved to {output_file}]")
        print()

    # Summary
    print("=" * 60)
    print("REFINEMENT SUMMARY")
    print("=" * 60)
    print(f"{'Pair':<25} {'prev':>8} {'refined':>8} {'improved':>8}")
    print("-" * 52)
    for name, val in sorted(results.items()):
        prev = prev_results.get(name, None)
        prev_s = f"{prev:.4f}" if prev else "N/A"
        val_s = f"{val:.5f}" if val else "FAILED"
        imp = f"{prev-val:.5f}" if (prev and val) else ""
        print(f"  {name:<25} {prev_s:>8} {val_s:>8} {imp:>8}")


if __name__ == "__main__":
    main()
