"""
Validate interpolation on the ACTUAL witness cycles that formed detected links.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.append('../src')
sys.path.append('../utils')

from cifar10_data_loader import download_cifar10, CIFAR10_CLASSES
from cifar10_batch_cycles import create_cifar10_cycle_data


def visualize_witness_interpolation(
    results_file: str,
    epsilon: float = 1.0,
    k: int = 15,
    n_edges: int = 6,
    n_interp: int = 5,
    n_pairs: int = 10,
    output_dir: str = '.',
    data_dir: str = '../data'
):
    """
    Visualize interpolations on actual witness cycles from linked pairs.
    """
    # Load linking results
    with open(results_file) as f:
        results = json.load(f)

    # Get parameters used
    params = results['parameters']
    print(f"Results from: epsilon={params['epsilon']}, k={params['k']}")

    # Load raw CIFAR-10 data
    print("Loading CIFAR-10 data...")
    X_raw, y_raw = download_cifar10(data_dir)

    # Build cycle data with SAME parameters as linking detection
    print(f"Building cycle data with k={params['k']}, epsilon={params['epsilon']}...")
    cycle_data = create_cifar10_cycle_data(
        k=params['k'],
        epsilon=params['epsilon'],
        mutual=params['mutual'],
        min_cycle_length=params['min_cycle_length'],
        data_dir=data_dir
    )

    # Find linked pairs with witnesses
    linked_pairs = [r for r in results['results']
                    if r.get('linking_found') and r.get('witness_cycles')]

    print(f"Found {len(linked_pairs)} linked pairs with witnesses")

    # Process each linked pair
    for pair_idx, pair_result in enumerate(linked_pairs[:n_pairs]):
        class_i = pair_result['class_i']
        class_j = pair_result['class_j']
        lk = pair_result['linking_number']
        witness = pair_result['witness_cycles']

        cycle_i_idx = witness['cycle_i_idx']
        cycle_j_idx = witness['cycle_j_idx']

        print(f"\n{'='*60}")
        print(f"Pair {pair_idx+1}: {CIFAR10_CLASSES[class_i]} vs {CIFAR10_CLASSES[class_j]}")
        print(f"Linking number: {lk}")
        print(f"Witness cycles: {cycle_i_idx} (class {class_i}), {cycle_j_idx} (class {class_j})")
        print('='*60)

        # Get the actual witness cycles
        cycles_i = cycle_data.get_class_cycles(class_i)
        cycles_j = cycle_data.get_class_cycles(class_j)

        if cycle_i_idx >= len(cycles_i) or cycle_j_idx >= len(cycles_j):
            print(f"Cycle index out of range, skipping")
            continue

        cycle_i = cycles_i[cycle_i_idx]
        cycle_j = cycles_j[cycle_j_idx]

        pca_i = cycle_data.get_class_points(class_i)
        pca_j = cycle_data.get_class_points(class_j)

        # Get raw images for each class
        mask_i = y_raw == class_i
        mask_j = y_raw == class_j
        X_class_i = X_raw[mask_i]
        X_class_j = X_raw[mask_j]

        # Create figure with both cycles
        fig, axes = plt.subplots(n_edges * 2, n_interp + 2,
                                 figsize=(2*(n_interp+2), 2*n_edges*2))

        # Visualize cycle i (first class)
        n_edges_i = min(n_edges, len(cycle_i) - 1)
        edge_indices_i = np.linspace(0, len(cycle_i) - 2, n_edges_i, dtype=int)

        for row, edge_idx in enumerate(edge_indices_i):
            idx_a = cycle_i[edge_idx]
            idx_b = cycle_i[edge_idx + 1]

            img_a = X_class_i[idx_a].reshape(32, 32, 3)
            img_b = X_class_i[idx_b].reshape(32, 32, 3)

            pca_dist = np.linalg.norm(pca_i[idx_a] - pca_i[idx_b])

            axes[row, 0].imshow(img_a)
            axes[row, 0].set_title(f'A (idx={idx_a})', fontsize=7)
            axes[row, 0].axis('off')

            for i in range(n_interp):
                t = (i + 1) / (n_interp + 1)
                img_interp = np.clip((1 - t) * img_a + t * img_b, 0, 1)
                axes[row, i + 1].imshow(img_interp)
                axes[row, i + 1].set_title(f't={t:.2f}', fontsize=7)
                axes[row, i + 1].axis('off')

            axes[row, n_interp + 1].imshow(img_b)
            axes[row, n_interp + 1].set_title(f'B (idx={idx_b})', fontsize=7)
            axes[row, n_interp + 1].axis('off')

            if row == 0:
                axes[row, 0].set_ylabel(f'{CIFAR10_CLASSES[class_i]}\nd={pca_dist:.3f}', fontsize=8)
            else:
                axes[row, 0].set_ylabel(f'd={pca_dist:.3f}', fontsize=8)

        # Visualize cycle j (second class)
        n_edges_j = min(n_edges, len(cycle_j) - 1)
        edge_indices_j = np.linspace(0, len(cycle_j) - 2, n_edges_j, dtype=int)

        for row_offset, edge_idx in enumerate(edge_indices_j):
            row = n_edges + row_offset
            idx_a = cycle_j[edge_idx]
            idx_b = cycle_j[edge_idx + 1]

            img_a = X_class_j[idx_a].reshape(32, 32, 3)
            img_b = X_class_j[idx_b].reshape(32, 32, 3)

            pca_dist = np.linalg.norm(pca_j[idx_a] - pca_j[idx_b])

            axes[row, 0].imshow(img_a)
            axes[row, 0].set_title(f'A (idx={idx_a})', fontsize=7)
            axes[row, 0].axis('off')

            for i in range(n_interp):
                t = (i + 1) / (n_interp + 1)
                img_interp = np.clip((1 - t) * img_a + t * img_b, 0, 1)
                axes[row, i + 1].imshow(img_interp)
                axes[row, i + 1].set_title(f't={t:.2f}', fontsize=7)
                axes[row, i + 1].axis('off')

            axes[row, n_interp + 1].imshow(img_b)
            axes[row, n_interp + 1].set_title(f'B (idx={idx_b})', fontsize=7)
            axes[row, n_interp + 1].axis('off')

            if row_offset == 0:
                axes[row, 0].set_ylabel(f'{CIFAR10_CLASSES[class_j]}\nd={pca_dist:.3f}', fontsize=8)
            else:
                axes[row, 0].set_ylabel(f'd={pca_dist:.3f}', fontsize=8)

        plt.suptitle(f'WITNESS CYCLES: {CIFAR10_CLASSES[class_i]} vs {CIFAR10_CLASSES[class_j]}\n'
                     f'Linking number = {lk}, epsilon={params["epsilon"]}, k={params["k"]}\n'
                     f'Cycle {cycle_i_idx} ({len(cycle_i)-1} pts) linked with Cycle {cycle_j_idx} ({len(cycle_j)-1} pts)',
                     fontsize=10)
        plt.tight_layout()

        output_file = os.path.join(output_dir,
            f'witness_{CIFAR10_CLASSES[class_i]}_{CIFAR10_CLASSES[class_j]}_lk{lk}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to linking results JSON')
    parser.add_argument('--n-pairs', type=int, default=10,
                       help='Number of linked pairs to visualize')
    parser.add_argument('--n-edges', type=int, default=6,
                       help='Edges per cycle to show')
    parser.add_argument('--n-interp', type=int, default=5,
                       help='Interpolation steps')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--data-dir', type=str, default='../data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    visualize_witness_interpolation(
        results_file=args.results_file,
        n_pairs=args.n_pairs,
        n_edges=args.n_edges,
        n_interp=args.n_interp,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
