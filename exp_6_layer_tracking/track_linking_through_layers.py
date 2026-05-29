#!/usr/bin/env python3
"""
Track linking number through network layers.

For trained width-3 models on the Hopf link (S^1 ⊔ S^1 in R^3),
compute the Gauss linking integral of the two curves after each layer.

This shows:
- ReLU: lk ≈ ±1 throughout (linking preserved — Theorem 3.7)
- GELU: lk transitions ±1 → 0 at some layer (unlinking via folding)
- ResNet (ReLU+skip): lk transitions ±1 → 0 (unlinking via |x|=x+2ReLU(-x))
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_width_scaling_v7 import FFN, FFNBlock, get_activation, tildeA, tildeB, sample_on_sphere


def generate_ordered_hopf_link(n_points=200, seed=42):
    """Generate two closed polylines forming the Hopf link (S^1 ⊔ S^1 in R^3).
    Points are in parametric order so they form valid closed curves."""
    rng = np.random.default_rng(seed)
    d = 3  # R^3 for n=1

    # Sample uniformly on S^1 (the 1-sphere) in parametric order
    t_vals = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Curve A: tildeA applied to points on S^1
    # S^1 in R^2: (cos(t), sin(t))
    curve_A = []
    for t in t_vals:
        u = np.array([np.cos(t), np.sin(t)])  # point on S^1 ⊂ R^2
        curve_A.append(tildeA(u))
    curve_A = np.array(curve_A)

    # Curve B: tildeB applied to points on S^1
    curve_B = []
    for t in t_vals:
        v = np.array([np.cos(t), np.sin(t)])
        curve_B.append(tildeB(v))
    curve_B = np.array(curve_B)

    return curve_A, curve_B


def gauss_linking_number(P, Q, nsub=1):
    """Compute Gauss linking integral for two closed polylines.
    P, Q: arrays of shape (N, 3) and (M, 3) representing closed polylines."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    n = len(P)
    m = len(Q)
    total = 0.0

    for i in range(n):
        p1 = P[i]
        p2 = P[(i + 1) % n]
        dp = p2 - p1

        for j in range(m):
            q1 = Q[j]
            q2 = Q[(j + 1) % m]
            dq = q2 - q1

            # Subdivide for accuracy
            for si in range(nsub):
                for sj in range(nsub):
                    t = (si + 0.5) / nsub
                    s = (sj + 0.5) / nsub
                    p = p1 + t * dp
                    q = q1 + s * dq
                    r = p - q
                    norm_r = np.linalg.norm(r)
                    if norm_r < 1e-12:
                        continue
                    cross = np.cross(dp / nsub, dq / nsub)
                    total += np.dot(r, cross) / (norm_r ** 3)

    return total / (4.0 * np.pi)


def track_linking(model, curve_A, curve_B, device='cpu'):
    """Pass two curves through a model layer by layer, computing lk after each."""
    model.eval()
    model.to(device)

    A = torch.tensor(curve_A, dtype=torch.float32).to(device)
    B = torch.tensor(curve_B, dtype=torch.float32).to(device)

    results = []

    # Compute initial linking number
    lk0 = gauss_linking_number(curve_A, curve_B, nsub=2)
    results.append({
        'layer': 'input',
        'layer_idx': -1,
        'lk_gauss': float(lk0),
        'lk_rounded': int(round(lk0)),
        'A_range': float(np.ptp(curve_A)),
        'B_range': float(np.ptp(curve_B)),
        'disjoint': float(np.min(np.linalg.norm(
            curve_A[:, None, :] - curve_B[None, :, :], axis=2)))
    })
    print(f"  Input: lk = {lk0:.4f} (rounded: {round(lk0)}), "
          f"min_dist = {results[-1]['disjoint']:.4f}")

    # Pass through each layer in model.layers
    with torch.no_grad():
        for idx, layer in enumerate(model.layers):
            A = layer(A)
            B = layer(B)

            A_np = A.cpu().numpy()
            B_np = B.cpu().numpy()

            # Compute linking number
            lk = gauss_linking_number(A_np, B_np, nsub=2)

            # Compute minimum distance between curves
            dists = np.linalg.norm(A_np[:, None, :] - B_np[None, :, :], axis=2)
            min_dist = float(np.min(dists))

            layer_name = type(layer).__name__
            if hasattr(layer, 'skip') and layer.skip:
                layer_name += '+skip'

            results.append({
                'layer': layer_name,
                'layer_idx': idx,
                'lk_gauss': float(lk),
                'lk_rounded': int(round(lk)),
                'A_range': float(np.ptp(A_np)),
                'B_range': float(np.ptp(B_np)),
                'disjoint': min_dist
            })
            print(f"  Layer {idx} ({layer_name}): lk = {lk:.4f} "
                  f"(rounded: {round(lk)}), min_dist = {min_dist:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Track linking number through network layers')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing best_model.pt and results.json')
    parser.add_argument('--n-points', type=int, default=200,
                        help='Points per curve')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as results-dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load model config
    config_file = os.path.join(args.results_dir, 'results.json')
    with open(config_file) as f:
        config = json.load(f)

    n = config['n']
    depth = config['depth']
    activation = config['activation']
    skip = config['skip']
    input_dim = config['input_dim']
    hidden_dim = config.get('hidden_dim', input_dim)

    print(f"Model: n={n}, depth={depth}, act={activation}, skip={skip}, "
          f"width={hidden_dim}")
    print(f"Test acc: {config['test_acc']:.4f}")

    if n != 1:
        print(f"WARNING: n={n}, not 1. This script is designed for the Hopf link (n=1, R^3).")
        print("The Gauss linking integral requires 3D curves.")
        return

    # Load model
    model = FFN(input_dim, depth, activation, skip, hidden_dim=hidden_dim)
    model_path = os.path.join(args.results_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Loaded model from {model_path}")

    # Generate ordered Hopf link curves
    print(f"\nGenerating Hopf link with {args.n_points} points per curve...")
    curve_A, curve_B = generate_ordered_hopf_link(args.n_points, args.seed)
    print(f"  Curve A shape: {curve_A.shape}, range: {np.ptp(curve_A, axis=0)}")
    print(f"  Curve B shape: {curve_B.shape}, range: {np.ptp(curve_B, axis=0)}")

    # Track linking through layers
    print(f"\nTracking linking number through {depth} layers...")
    layer_results = track_linking(model, curve_A, curve_B)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {activation}{'_skip' if skip else ''} width={hidden_dim}")
    print(f"{'='*60}")
    print(f"{'Layer':<20} {'lk (Gauss)':>12} {'lk (round)':>10} {'min_dist':>10}")
    print('-' * 55)
    for r in layer_results:
        name = r['layer'] if r['layer_idx'] < 0 else f"Layer {r['layer_idx']} ({r['layer']})"
        print(f"{name:<20} {r['lk_gauss']:>12.4f} {r['lk_rounded']:>10d} {r['disjoint']:>10.4f}")

    # Save results
    output = {
        'config': config,
        'n_points': args.n_points,
        'seed': args.seed,
        'layer_tracking': layer_results
    }
    out_file = os.path.join(output_dir, 'linking_through_layers.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
