#!/usr/bin/env python3
"""
Per-link-pair evaluation: for each detected link independently,
compute the accuracy gap at various distances, then aggregate.
Supports using predictions from a different output dir than the detection.
"""

import os
import sys
import json
import numpy as np
from scipy.spatial.distance import cdist
import argparse


def eval_per_link(dataset_path, detection_dir, prediction_dir):
    print("=" * 60)
    print("Per-Link-Pair Evaluation")
    print(f"Detection: {detection_dir}")
    print(f"Predictions: {prediction_dir}")
    print("=" * 60)

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    pca_3d = data['pca_3d']
    labels = data['labels']
    test_pca_3d = data['test_pca_3d']
    test_labels = data['test_labels']
    n_test = len(test_pca_3d)

    # Load detection result
    link_path = os.path.join(detection_dir, 'link_detection_result.json')
    with open(link_path) as f:
        link = json.load(f)

    if 'links' in link:
        links = link['links']
        eps = link['working_eps']
    else:
        links = [{
            'bird_global': link.get('bird_cycle_global_indices', []),
            'deer_global': link.get('deer_cycle_global_indices', []),
            'lk': link.get('linking_number', 1),
        }]
        eps = link.get('working_eps', link.get('eps_min', 1.0))

    print(f"{len(links)} links, eps={eps}")

    # Load test predictions
    preds = {}
    for act in ['relu', 'gelu']:
        test_path = os.path.join(prediction_dir, f'{act}_L8_noskip', 'test_predictions.npz')
        if os.path.exists(test_path):
            td = np.load(test_path)
            preds[act] = (td['preds'] == td['labels']).astype(np.int8)

    if len(preds) < 2:
        print("ERROR: Need both relu and gelu test predictions")
        return

    relu_correct = preds['relu']
    gelu_correct = preds['gelu']

    overall_gap = gelu_correct.mean() - relu_correct.mean()
    print(f"Overall test: ReLU={relu_correct.mean():.4f}, GELU={gelu_correct.mean():.4f}, gap={overall_gap:+.4f}")

    # Radii to evaluate (in multiples of eps)
    radii = [1, 3, 5, 10, 20, 50]

    # Per-link evaluation
    per_link_results = []
    print(f"\n{'Link':>5} {'lk':>3} {'b_len':>5} {'d_len':>5}", end='')
    for r in radii:
        print(f" {'gap@'+str(r)+'eps':>10}", end='')
    print(f" {'n@5eps':>7}")

    for li, lnk in enumerate(links):
        b_idx = np.array(lnk['bird_global'])
        d_idx = np.array(lnk['deer_global'])
        cycle_idx = np.unique(np.concatenate([b_idx, d_idx]))
        cycle_pts = pca_3d[cycle_idx]
        lk = lnk.get('lk', '?')

        # Compute test distances to THIS link's cycle points
        test_dists = np.full(n_test, np.inf, dtype=np.float32)
        chunk = 5000
        for start in range(0, n_test, chunk):
            end = min(start + chunk, n_test)
            test_dists[start:end] = cdist(test_pca_3d[start:end], cycle_pts).min(axis=1)

        link_result = {
            'link_idx': li, 'lk': int(lk) if isinstance(lk, (int, float)) else lk,
            'bird_len': len(b_idx), 'deer_len': len(d_idx),
        }

        print(f"{li:5d} {lk:>3}", end='')
        print(f" {len(b_idx):5d} {len(d_idx):5d}", end='')

        for r in radii:
            mask = test_dists <= r * eps
            n_nearby = mask.sum()
            if n_nearby == 0:
                link_result[f'gap_{r}eps'] = None
                link_result[f'n_{r}eps'] = 0
                print(f" {'N/A':>10}", end='')
            else:
                relu_acc = relu_correct[mask].mean()
                gelu_acc = gelu_correct[mask].mean()
                gap = gelu_acc - relu_acc
                link_result[f'gap_{r}eps'] = float(gap)
                link_result[f'relu_{r}eps'] = float(relu_acc)
                link_result[f'gelu_{r}eps'] = float(gelu_acc)
                link_result[f'n_{r}eps'] = int(n_nearby)
                print(f" {gap:>+10.4f}", end='')

        n5 = link_result.get(f'n_5eps', 0)
        print(f" {n5:7d}")
        per_link_results.append(link_result)

    # Aggregate statistics across links
    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS LINKS")
    print(f"{'='*60}")

    for r in radii:
        gaps = [p[f'gap_{r}eps'] for p in per_link_results if p.get(f'gap_{r}eps') is not None]
        if not gaps:
            continue
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        n_pos = sum(1 for g in gaps if g > 0)
        n_neg = sum(1 for g in gaps if g < 0)
        median_gap = np.median(gaps)
        total_n = sum(p.get(f'n_{r}eps', 0) for p in per_link_results)
        print(f"  {r}eps: mean_gap={mean_gap:+.4f} ± {std_gap:.4f}, median={median_gap:+.4f}, "
              f"pos/neg={n_pos}/{n_neg}, total_n={total_n}, n_links={len(gaps)}")

    # Also compute pooled gap (all test points near ANY link)
    print(f"\nPOOLED (all links combined):")
    all_cycle_idx = []
    for lnk in links:
        all_cycle_idx.extend(lnk['bird_global'])
        all_cycle_idx.extend(lnk['deer_global'])
    all_cycle_pts = pca_3d[np.unique(all_cycle_idx)]

    pooled_dists = np.full(n_test, np.inf, dtype=np.float32)
    for start in range(0, n_test, 5000):
        end = min(start + 5000, n_test)
        pooled_dists[start:end] = cdist(test_pca_3d[start:end], all_cycle_pts).min(axis=1)

    for r in radii:
        mask = pooled_dists <= r * eps
        n = mask.sum()
        if n < 5:
            continue
        relu_acc = relu_correct[mask].mean()
        gelu_acc = gelu_correct[mask].mean()
        gap = gelu_acc - relu_acc
        print(f"  {r}eps: gap={gap:+.4f}, relu={relu_acc:.4f}, gelu={gelu_acc:.4f}, n={n}")

    # Far from all links
    far_mask = pooled_dists > 50 * eps
    if far_mask.sum() > 0:
        relu_far = relu_correct[far_mask].mean()
        gelu_far = gelu_correct[far_mask].mean()
        print(f"  far(>50eps): gap={gelu_far-relu_far:+.4f}, relu={relu_far:.4f}, gelu={gelu_far:.4f}, n={far_mask.sum()}")

    # Save
    output = {
        'per_link': per_link_results,
        'aggregate': {
            r: {
                'mean_gap': float(np.mean([p[f'gap_{r}eps'] for p in per_link_results if p.get(f'gap_{r}eps') is not None])),
                'n_links': len([p for p in per_link_results if p.get(f'gap_{r}eps') is not None]),
            }
            for r in radii
            if any(p.get(f'gap_{r}eps') is not None for p in per_link_results)
        },
        'overall_gap': float(overall_gap),
        'n_links': len(links),
        'eps': float(eps),
    }

    save_path = os.path.join(detection_dir, 'per_link_eval.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--detection-dir', required=True, help='Dir with link_detection_result.json')
    parser.add_argument('--prediction-dir', required=True, help='Dir with relu_L8_noskip/ and gelu_L8_noskip/')
    args = parser.parse_args()
    eval_per_link(args.dataset, args.detection_dir, args.prediction_dir)
