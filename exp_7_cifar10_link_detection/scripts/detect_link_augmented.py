#!/usr/bin/env python3
"""
Step 3: Detect linking in the pre-computed augmented dataset.
Uses same 3D PCA coordinates from the NPZ.
Saves witness cycle vertex indices (indexing into the augmented dataset).
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from link_detector import build_knn_graph, fundamental_cycle_basis, gauss_linking_number_numeric

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'witness_eval')
DEFAULT_DATASET = str(PROJECT_ROOT / 'data' / 'cifar10_bird_deer_40x.npz')


def detect_link(dataset_path, k=15, mutual=True, min_cycle_length=30,
                eps_low=0.02, eps_high=0.10, eps_precision=0.002,
                output_dir=DEFAULT_OUTPUT_DIR, max_links=10):
    print("=" * 60)
    print("Linking Detection on Augmented Dataset")
    print("=" * 60)

    data = np.load(dataset_path, allow_pickle=True)
    pca_3d = data['pca_3d']
    labels = data['labels']

    bird_idx = np.where(labels == 0)[0]
    deer_idx = np.where(labels == 1)[0]
    bird_pts = pca_3d[bird_idx]
    deer_pts = pca_3d[deer_idx]

    print(f"Bird: {len(bird_pts)} points, Deer: {len(deer_pts)} points")

    # Binary search for eps_min
    print(f"\nBinary search for eps_min in [{eps_low}, {eps_high}]...")

    def try_epsilon(eps):
        adj_b = build_knn_graph(bird_pts, k=k, epsilon=eps, mutual=mutual)
        adj_d = build_knn_graph(deer_pts, k=k, epsilon=eps, mutual=mutual)
        cycles_b = fundamental_cycle_basis(adj_b)
        cycles_d = fundamental_cycle_basis(adj_d)

        # Filter by length
        cycles_b = [c for c in cycles_b if len(c) >= min_cycle_length]
        cycles_d = [c for c in cycles_d if len(c) >= min_cycle_length]

        if not cycles_b or not cycles_d:
            return None, cycles_b, cycles_d

        # Test cycle pairs (limit to top 200 longest each)
        cycles_b.sort(key=len, reverse=True)
        cycles_d.sort(key=len, reverse=True)
        cycles_b = cycles_b[:200]
        cycles_d = cycles_d[:200]

        for i, cb in enumerate(cycles_b):
            pts_b = bird_pts[cb]
            for j, cd in enumerate(cycles_d):
                pts_d = deer_pts[cd]
                lk = gauss_linking_number_numeric(pts_b, pts_d, nsub=4)
                lk_rounded = round(lk)
                if abs(lk - lk_rounded) < 0.1 and lk_rounded != 0:
                    return {
                        'lk': lk_rounded,
                        'lk_raw': float(lk),
                        'bird_cycle_idx': i,
                        'deer_cycle_idx': j,
                        'bird_cycle_vertices': [int(v) for v in cb],  # indices into bird_pts
                        'deer_cycle_vertices': [int(v) for v in cd],
                        'bird_cycle_length': len(cb),
                        'deer_cycle_length': len(cd),
                    }, cycles_b, cycles_d
        return None, cycles_b, cycles_d

    # First verify eps_high works
    print(f"  Testing eps_high={eps_high}...")
    result_high, _, _ = try_epsilon(eps_high)
    if result_high is None:
        print(f"  No link at eps_high={eps_high}, trying larger...")
        for eps_try in [0.15, 0.2, 0.3, 0.5, 1.0]:
            result_high, _, _ = try_epsilon(eps_try)
            if result_high is not None:
                eps_high = eps_try
                print(f"  Found link at eps={eps_try}")
                break
        if result_high is None:
            print("  ERROR: No link found at any epsilon!")
            return None

    # Binary search
    while eps_high - eps_low > eps_precision:
        eps_mid = (eps_low + eps_high) / 2
        print(f"  [{eps_low:.4f}, {eps_high:.4f}] testing {eps_mid:.4f}...")
        result, _, _ = try_epsilon(eps_mid)
        if result is not None:
            eps_high = eps_mid
            result_high = result
        else:
            eps_low = eps_mid

    eps_min = eps_high
    result = result_high
    print(f"\neps_min = {eps_min:.4f}")
    print(f"Link 1: lk={result['lk']}, bird={result['bird_cycle_length']}pts, deer={result['deer_cycle_length']}pts")

    # Find more links at eps_min by continuing through remaining cycle pairs
    all_links = [result]
    if max_links > 1:
        print(f"\nSearching for up to {max_links} links at eps_min={eps_min}...")
        # Rebuild cycles at eps_min
        adj_b = build_knn_graph(bird_pts, k=k, epsilon=eps_min, mutual=mutual)
        adj_d = build_knn_graph(deer_pts, k=k, epsilon=eps_min, mutual=mutual)
        cycles_b = [c for c in fundamental_cycle_basis(adj_b) if len(c) >= min_cycle_length]
        cycles_d = [c for c in fundamental_cycle_basis(adj_d) if len(c) >= min_cycle_length]
        cycles_b.sort(key=len, reverse=True)
        cycles_d.sort(key=len, reverse=True)
        cycles_b = cycles_b[:200]
        cycles_d = cycles_d[:200]

        used_pairs = {(result['bird_cycle_idx'], result['deer_cycle_idx'])}
        for i, cb in enumerate(cycles_b):
            if len(all_links) >= max_links:
                break
            for j, cd in enumerate(cycles_d):
                if len(all_links) >= max_links:
                    break
                if (i, j) in used_pairs:
                    continue
                lk = gauss_linking_number_numeric(bird_pts[cb], deer_pts[cd], nsub=4)
                lk_r = round(lk)
                if abs(lk - lk_r) < 0.1 and lk_r != 0:
                    all_links.append({
                        'lk': lk_r, 'lk_raw': float(lk),
                        'bird_cycle_idx': i, 'deer_cycle_idx': j,
                        'bird_cycle_vertices': [int(v) for v in cb],
                        'deer_cycle_vertices': [int(v) for v in cd],
                        'bird_cycle_length': len(cb), 'deer_cycle_length': len(cd),
                    })
                    used_pairs.add((i, j))
                    print(f"  Link {len(all_links)}: lk={lk_r}, bird={len(cb)}pts, deer={len(cd)}pts")

    print(f"Total: {len(all_links)} links found")

    # Build output
    bbox_diag = np.linalg.norm(pca_3d.max(axis=0) - pca_3d.min(axis=0))
    links_output = []
    all_bird_global = []
    all_deer_global = []
    for lnk in all_links:
        bg = bird_idx[lnk['bird_cycle_vertices']].tolist()
        dg = deer_idx[lnk['deer_cycle_vertices']].tolist()
        all_bird_global.extend(bg)
        all_deer_global.extend(dg)
        links_output.append({
            'lk': lnk['lk'], 'bird_len': lnk['bird_cycle_length'],
            'deer_len': lnk['deer_cycle_length'],
            'bird_global': bg, 'deer_global': dg,
            'bird_centroid': pca_3d[bg].mean(axis=0).tolist(),
            'deer_centroid': pca_3d[dg].mean(axis=0).tolist(),
        })

    first_bird = bird_idx[all_links[0]['bird_cycle_vertices']]
    first_deer = deer_idx[all_links[0]['deer_cycle_vertices']]

    output = {
        'eps_min': float(eps_min),
        'working_eps': float(eps_min),
        'eps_fraction': float(eps_min / bbox_diag),
        'bbox_diagonal': float(bbox_diag),
        'n_links_found': len(all_links),
        'links': links_output,
        'all_bird_global_indices': np.unique(all_bird_global).tolist(),
        'all_deer_global_indices': np.unique(all_deer_global).tolist(),
        'all_cycle_global_indices': np.unique(all_bird_global + all_deer_global).tolist(),
        # First link (backward compat)
        'linking_number': all_links[0]['lk'],
        'bird_cycle_global_indices': first_bird.tolist(),
        'deer_cycle_global_indices': first_deer.tolist(),
        'bird_centroid_3d': pca_3d[first_bird].mean(axis=0).tolist(),
        'deer_centroid_3d': pca_3d[first_deer].mean(axis=0).tolist(),
        'parameters': {'k': k, 'mutual': mutual, 'min_cycle_length': min_cycle_length,
                       'max_links': max_links},
    }

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'link_detection_result.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {save_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--eps-low', type=float, default=0.02)
    parser.add_argument('--eps-high', type=float, default=0.10)
    parser.add_argument('--max-links', type=int, default=10)
    parser.add_argument('--output-dir', default=os.environ.get('WITNESS_OUTPUT_DIR', DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    detect_link(args.dataset, k=args.k, eps_low=args.eps_low, eps_high=args.eps_high,
                output_dir=args.output_dir, max_links=args.max_links)
