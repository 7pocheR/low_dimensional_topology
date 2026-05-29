"""
Edge length analysis for CIFAR-10 to determine optimal k and epsilon parameters.

This script analyzes the distance distribution and graph connectivity
for various parameter settings to guide the linking detection pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import sys
import os

sys.path.append('../src')
sys.path.append('../utils')

from link_detector import build_knn_graph, fundamental_cycle_basis
from cifar10_data_loader import (
    load_or_create_cifar10_pca,
    CIFAR10_CLASSES,
    compute_distance_statistics
)


def analyze_graph_connectivity(
    points: np.ndarray,
    k_values: list,
    epsilon_values: list,
    class_name: str,
    mutual: bool = True
) -> dict:
    """
    Analyze k-NN graph connectivity for various parameter combinations.

    Returns statistics on edges, isolated nodes, and cycles.
    """
    n = len(points)
    results = {}

    for k in k_values:
        for epsilon in epsilon_values:
            # Build graph
            adj = build_knn_graph(points, k=k, epsilon=epsilon, mutual=mutual)

            # Count edges
            n_edges = sum(len(neighbors) for neighbors in adj.values()) // 2

            # Count isolated nodes
            isolated = sum(1 for node in adj if len(adj[node]) == 0)

            # Average degree
            avg_degree = 2 * n_edges / n if n > 0 else 0

            # Extract cycles
            cycles = fundamental_cycle_basis(adj)
            n_cycles = len(cycles)

            # Cycle length statistics
            if cycles:
                cycle_lengths = [len(c) - 1 for c in cycles]  # -1 because cycles repeat first vertex
                min_cycle_len = min(cycle_lengths)
                max_cycle_len = max(cycle_lengths)
                median_cycle_len = np.median(cycle_lengths)
            else:
                min_cycle_len = max_cycle_len = median_cycle_len = 0

            results[(k, epsilon)] = {
                'n_edges': n_edges,
                'isolated': isolated,
                'isolated_pct': 100 * isolated / n,
                'avg_degree': avg_degree,
                'n_cycles': n_cycles,
                'min_cycle_len': min_cycle_len,
                'max_cycle_len': max_cycle_len,
                'median_cycle_len': median_cycle_len
            }

    return results


def run_cifar10_edge_analysis(
    sample_size: int = 2000,
    k_values: list = None,
    epsilon_values: list = None,
    mutual: bool = True,
    output_dir: str = '.',
    data_dir: str = '../data'
):
    """
    Run comprehensive edge length analysis on CIFAR-10.

    Args:
        sample_size: Number of samples per class (for efficiency)
        k_values: List of k values to test
        epsilon_values: List of epsilon values to test
        mutual: Whether to use mutual k-NN
        output_dir: Directory to save results
        data_dir: Directory for CIFAR-10 data cache
    """
    if k_values is None:
        k_values = [8, 10, 15, 20]
    if epsilon_values is None:
        # Will be determined from distance statistics
        epsilon_values = None

    print("=" * 70)
    print("CIFAR-10 EDGE LENGTH ANALYSIS")
    print("=" * 70)

    # Load PCA data
    pca_data, metadata = load_or_create_cifar10_pca(data_dir=data_dir)

    # Compute distance statistics first
    print("\n" + "=" * 70)
    print("STEP 1: DISTANCE STATISTICS")
    print("=" * 70)

    dist_stats = compute_distance_statistics(pca_data)

    # Print distance summary
    print(f"\n{'Class':<12} {'Min':<7} {'P25':<7} {'Median':<7} {'P75':<7} {'P90':<7} {'Max':<7}")
    print("-" * 70)
    for class_id in range(10):
        s = dist_stats[class_id]
        print(f"{CIFAR10_CLASSES[class_id]:<12} "
              f"{s['min']:<7.2f} "
              f"{s['p25']:<7.2f} "
              f"{s['median']:<7.2f} "
              f"{s['p75']:<7.2f} "
              f"{s['p90']:<7.2f} "
              f"{s['max']:<7.2f}")

    # Determine epsilon values based on distance statistics
    all_medians = np.array([dist_stats[i]['median'] for i in range(10)])
    all_p25 = np.array([dist_stats[i]['p25'] for i in range(10)])
    all_p75 = np.array([dist_stats[i]['p75'] for i in range(10)])

    median_of_medians = np.median(all_medians)

    if epsilon_values is None:
        # Suggest epsilon values based on distance distribution
        epsilon_values = [
            round(median_of_medians * 0.3, 2),
            round(median_of_medians * 0.5, 2),
            round(median_of_medians * 0.7, 2),
            round(median_of_medians * 1.0, 2),
            round(median_of_medians * 1.5, 2),
        ]
        print(f"\nAuto-selected epsilon values based on median distance ({median_of_medians:.2f}):")
        print(f"  epsilon = {epsilon_values}")

    print(f"\nTesting parameters:")
    print(f"  k values: {k_values}")
    print(f"  epsilon values: {epsilon_values}")
    print(f"  mutual k-NN: {mutual}")
    print(f"  sample size: {sample_size}")

    # Analyze each class
    print("\n" + "=" * 70)
    print("STEP 2: GRAPH CONNECTIVITY ANALYSIS")
    print("=" * 70)

    all_results = {}

    for class_id in range(10):
        class_name = CIFAR10_CLASSES[class_id]
        X = pca_data[class_id]

        # Subsample if needed
        if len(X) > sample_size:
            np.random.seed(42 + class_id)
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        print(f"\n--- Class {class_id}: {class_name} ({len(X_sample)} samples) ---")

        results = analyze_graph_connectivity(
            X_sample, k_values, epsilon_values, class_name, mutual=mutual
        )
        all_results[class_id] = results

        # Print summary for this class
        print(f"{'k':<4} {'eps':<6} {'edges':<8} {'isolated':<10} {'avg_deg':<8} {'cycles':<8} {'med_cyc_len':<12}")
        print("-" * 70)
        for k in k_values:
            for epsilon in epsilon_values:
                r = results[(k, epsilon)]
                print(f"{k:<4} {epsilon:<6.2f} {r['n_edges']:<8} "
                      f"{r['isolated']:>4} ({r['isolated_pct']:>4.1f}%) "
                      f"{r['avg_degree']:<8.1f} {r['n_cycles']:<8} {r['median_cycle_len']:<12.0f}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("STEP 3: PARAMETER RECOMMENDATIONS")
    print("=" * 70)

    # Find parameters that work well across all classes
    good_params = []

    for k in k_values:
        for epsilon in epsilon_values:
            # Check if parameters are good for all classes
            all_have_cycles = True
            total_cycles = 0
            max_isolated_pct = 0
            min_cycles = float('inf')

            for class_id in range(10):
                r = all_results[class_id][(k, epsilon)]
                if r['n_cycles'] == 0:
                    all_have_cycles = False
                total_cycles += r['n_cycles']
                max_isolated_pct = max(max_isolated_pct, r['isolated_pct'])
                min_cycles = min(min_cycles, r['n_cycles'])

            if all_have_cycles and max_isolated_pct < 20:
                good_params.append({
                    'k': k,
                    'epsilon': epsilon,
                    'total_cycles': total_cycles,
                    'min_cycles': min_cycles,
                    'max_isolated_pct': max_isolated_pct
                })

    # Sort by number of cycles (more is better for finding linking)
    good_params.sort(key=lambda x: (-x['min_cycles'], x['max_isolated_pct']))

    if good_params:
        print("\nRecommended parameters (sorted by min cycles across classes):")
        print(f"{'Rank':<5} {'k':<4} {'epsilon':<8} {'min_cycles':<12} {'total_cycles':<14} {'max_isolated%':<14}")
        print("-" * 70)
        for i, p in enumerate(good_params[:10]):
            print(f"{i+1:<5} {p['k']:<4} {p['epsilon']:<8.2f} {p['min_cycles']:<12} "
                  f"{p['total_cycles']:<14} {p['max_isolated_pct']:<14.1f}")

        # Print suggested parameter sets to test
        print("\n" + "=" * 70)
        print("SUGGESTED PARAMETER SETS FOR LINKING DETECTION")
        print("=" * 70)
        top_params = good_params[:3]
        for i, p in enumerate(top_params):
            print(f"\nParameter Set {i+1}:")
            print(f"  k={p['k']}, epsilon={p['epsilon']}, mutual={mutual}")
    else:
        print("\nNo parameters found with cycles in all classes!")
        print("Consider increasing epsilon values or k values.")

    # Create visualization
    create_analysis_plots(all_results, k_values, epsilon_values, dist_stats, output_dir)

    return all_results, good_params


def create_analysis_plots(
    all_results: dict,
    k_values: list,
    epsilon_values: list,
    dist_stats: dict,
    output_dir: str
):
    """Create visualization plots for the analysis."""

    # Plot 1: Distance distribution for all classes
    fig1, axes1 = plt.subplots(2, 5, figsize=(20, 8))
    axes1 = axes1.flatten()

    for class_id in range(10):
        ax = axes1[class_id]
        s = dist_stats[class_id]

        # Create box plot-like visualization
        positions = [s['min'], s['p25'], s['median'], s['p75'], s['p90'], s['max']]
        labels = ['min', 'p25', 'med', 'p75', 'p90', 'max']

        ax.bar(range(len(positions)), positions, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f'{CIFAR10_CLASSES[class_id]}')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)

    fig1.suptitle('CIFAR-10 Distance Statistics by Class (3D PCA Space)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cifar10_distance_stats.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(output_dir, 'cifar10_distance_stats.png')}")

    # Plot 2: Heatmap of cycle counts for different parameters
    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
    axes2 = axes2.flatten()

    for class_id in range(10):
        ax = axes2[class_id]
        results = all_results[class_id]

        # Create matrix of cycle counts
        cycle_matrix = np.zeros((len(k_values), len(epsilon_values)))
        for i, k in enumerate(k_values):
            for j, epsilon in enumerate(epsilon_values):
                cycle_matrix[i, j] = results[(k, epsilon)]['n_cycles']

        im = ax.imshow(cycle_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(epsilon_values)))
        ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values], rotation=45)
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels(k_values)
        ax.set_xlabel('epsilon')
        ax.set_ylabel('k')
        ax.set_title(f'{CIFAR10_CLASSES[class_id]}')

        # Add text annotations
        for i in range(len(k_values)):
            for j in range(len(epsilon_values)):
                text = ax.text(j, i, f'{int(cycle_matrix[i, j])}',
                              ha='center', va='center', color='black', fontsize=8)

    fig2.suptitle('CIFAR-10 Cycle Counts by Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cifar10_cycle_counts.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'cifar10_cycle_counts.png')}")

    # Plot 3: Isolated node percentage
    fig3, axes3 = plt.subplots(2, 5, figsize=(20, 8))
    axes3 = axes3.flatten()

    for class_id in range(10):
        ax = axes3[class_id]
        results = all_results[class_id]

        # Create matrix of isolated percentages
        iso_matrix = np.zeros((len(k_values), len(epsilon_values)))
        for i, k in enumerate(k_values):
            for j, epsilon in enumerate(epsilon_values):
                iso_matrix[i, j] = results[(k, epsilon)]['isolated_pct']

        im = ax.imshow(iso_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
        ax.set_xticks(range(len(epsilon_values)))
        ax.set_xticklabels([f'{e:.2f}' for e in epsilon_values], rotation=45)
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels(k_values)
        ax.set_xlabel('epsilon')
        ax.set_ylabel('k')
        ax.set_title(f'{CIFAR10_CLASSES[class_id]}')

    fig3.suptitle('CIFAR-10 Isolated Node % by Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cifar10_isolated_nodes.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'cifar10_isolated_nodes.png')}")

    plt.close('all')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CIFAR-10 edge length analysis')
    parser.add_argument('--sample-size', type=int, default=2000,
                       help='Sample size per class (default: 2000)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[8, 10, 15, 20],
                       help='k values to test (default: 8 10 15 20)')
    parser.add_argument('--epsilon-values', type=float, nargs='+', default=None,
                       help='epsilon values to test (default: auto from distance stats)')
    parser.add_argument('--no-mutual', action='store_true',
                       help='Use standard k-NN instead of mutual k-NN')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Data directory for CIFAR-10 cache')
    args = parser.parse_args()

    results, good_params = run_cifar10_edge_analysis(
        sample_size=args.sample_size,
        k_values=args.k_values,
        epsilon_values=args.epsilon_values,
        mutual=not args.no_mutual,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
