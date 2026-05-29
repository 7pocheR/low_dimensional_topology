#!/usr/bin/env python3
"""
Analyze correlation between topological linking and CNN confusion errors.

This script:
1. Computes linking consistency across multiple detection runs
2. Loads CNN 10-class confusion matrices
3. Correlates confusion errors with linking status
4. Compares monotonic vs non-monotonic advantage on linked vs unlinked pairs
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_linking_matrices(results_dir):
    """Load all linking matrices from detection runs."""
    csv_files = glob.glob(os.path.join(results_dir, 'linking_matrix_*.csv'))

    matrices = []
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        matrices.append(df.values)

    print(f"Loaded {len(matrices)} linking matrices")
    return matrices


def compute_linking_consistency(matrices):
    """
    Compute linking consistency for each class pair.

    Consistency = fraction of runs where pair is detected as linked (|lk| >= 1).
    Returns 10x10 matrix of consistency values in [0, 1].
    """
    n_classes = 10
    n_runs = len(matrices)

    # Count how many times each pair is linked
    linked_count = np.zeros((n_classes, n_classes))

    for mat in matrices:
        linked_count += (np.abs(mat) >= 1).astype(float)

    consistency = linked_count / n_runs
    return consistency


def categorize_pairs(consistency_matrix, high_thresh=0.7, low_thresh=0.3):
    """
    Categorize class pairs into linked, unlinked, and ambiguous.

    Returns:
        linked_pairs: list of (i, j) tuples with consistency >= high_thresh
        unlinked_pairs: list of (i, j) tuples with consistency <= low_thresh
        ambiguous_pairs: list of (i, j) tuples in between
    """
    linked = []
    unlinked = []
    ambiguous = []

    n = len(consistency_matrix)
    for i in range(n):
        for j in range(i+1, n):
            cons = consistency_matrix[i, j]
            if cons >= high_thresh:
                linked.append((i, j, cons))
            elif cons <= low_thresh:
                unlinked.append((i, j, cons))
            else:
                ambiguous.append((i, j, cons))

    return linked, unlinked, ambiguous


def load_cnn_results(results_dir):
    """
    Load CNN 10-class results with confusion matrices.

    Returns:
        dict mapping config_name -> {
            'activation': str,
            'is_monotonic': bool,
            'use_skip': bool,
            'layers': int,
            'accuracy': float,
            'confusion_matrix': 10x10 array
        }
    """
    json_files = glob.glob(os.path.join(results_dir, 'result_*.json'))

    results = {}
    monotonic_activations = {'relu', 'elu', 'selu', 'leaky_relu'}

    for f in json_files:
        with open(f, 'r') as fp:
            data = json.load(fp)

        config = data.get('config', {})
        activation = config.get('activation', 'unknown')

        config_name = os.path.basename(f).replace('result_', '').replace('.json', '')

        results[config_name] = {
            'activation': activation,
            'is_monotonic': activation.lower() in monotonic_activations,
            'use_skip': config.get('use_skip', False),
            'layers': config.get('total_layers', 0),
            'accuracy': data.get('test_accuracy', 0),
            'confusion_matrix': np.array(data.get('confusion_matrix', []))
        }

    print(f"Loaded {len(results)} CNN results")
    return results


def compute_pairwise_errors(confusion_matrix, normalize=True):
    """
    Compute pairwise confusion errors from confusion matrix.

    Returns 10x10 matrix where entry (i,j) = errors between classes i and j.
    Symmetric: errors(i,j) = confusion[i,j] + confusion[j,i]
    """
    cm = np.array(confusion_matrix)
    n = cm.shape[0]

    # Symmetrize: bidirectional confusion
    errors = cm + cm.T
    np.fill_diagonal(errors, 0)

    if normalize:
        # Normalize by total samples in each pair
        totals = cm.sum(axis=1)
        for i in range(n):
            for j in range(i+1, n):
                pair_total = totals[i] + totals[j]
                if pair_total > 0:
                    errors[i, j] /= pair_total
                    errors[j, i] = errors[i, j]

    return errors


def analyze_linking_vs_errors(linking_consistency, error_matrices, results_info):
    """
    Analyze correlation between linking consistency and confusion errors.

    Returns analysis dict with correlations for monotonic vs non-monotonic.
    """
    analysis = {
        'monotonic': {'correlations': [], 'error_on_linked': [], 'error_on_unlinked': []},
        'non_monotonic': {'correlations': [], 'error_on_linked': [], 'error_on_unlinked': []}
    }

    linked_pairs, unlinked_pairs, _ = categorize_pairs(linking_consistency)

    for config_name, info in results_info.items():
        if info['confusion_matrix'].size == 0:
            continue

        errors = compute_pairwise_errors(info['confusion_matrix'])

        # Get upper triangle values (excluding diagonal)
        n = len(linking_consistency)
        upper_idx = np.triu_indices(n, k=1)

        link_vals = linking_consistency[upper_idx]
        error_vals = errors[upper_idx]

        # Compute correlation
        corr, p_val = stats.spearmanr(link_vals, error_vals)

        # Mean error on linked vs unlinked pairs
        linked_error = np.mean([errors[i, j] for i, j, _ in linked_pairs]) if linked_pairs else 0
        unlinked_error = np.mean([errors[i, j] for i, j, _ in unlinked_pairs]) if unlinked_pairs else 0

        category = 'monotonic' if info['is_monotonic'] else 'non_monotonic'
        analysis[category]['correlations'].append({
            'config': config_name,
            'correlation': corr,
            'p_value': p_val,
            'activation': info['activation']
        })
        analysis[category]['error_on_linked'].append(linked_error)
        analysis[category]['error_on_unlinked'].append(unlinked_error)

    return analysis


def compare_activations_on_pairs(results_info, linked_pairs, unlinked_pairs):
    """
    Compare monotonic vs non-monotonic activations specifically on linked vs unlinked pairs.

    Key hypothesis:
    - Non-monotonic advantage should be larger on linked pairs
    - No systematic advantage on unlinked pairs
    """
    comparisons = {
        'linked': {'monotonic_errors': [], 'nonmono_errors': []},
        'unlinked': {'monotonic_errors': [], 'nonmono_errors': []}
    }

    for config_name, info in results_info.items():
        if info['confusion_matrix'].size == 0:
            continue

        errors = compute_pairwise_errors(info['confusion_matrix'])

        # Mean error on linked pairs
        linked_err = np.mean([errors[i, j] for i, j, _ in linked_pairs]) if linked_pairs else 0
        # Mean error on unlinked pairs
        unlinked_err = np.mean([errors[i, j] for i, j, _ in unlinked_pairs]) if unlinked_pairs else 0

        if info['is_monotonic']:
            comparisons['linked']['monotonic_errors'].append(linked_err)
            comparisons['unlinked']['monotonic_errors'].append(unlinked_err)
        else:
            comparisons['linked']['nonmono_errors'].append(linked_err)
            comparisons['unlinked']['nonmono_errors'].append(unlinked_err)

    # Compute statistics
    results = {}
    for pair_type in ['linked', 'unlinked']:
        mono = comparisons[pair_type]['monotonic_errors']
        nonmono = comparisons[pair_type]['nonmono_errors']

        if mono and nonmono:
            mono_mean = np.mean(mono)
            nonmono_mean = np.mean(nonmono)
            advantage = mono_mean - nonmono_mean  # Positive = non-mono is better (lower error)

            # Statistical test
            t_stat, p_val = stats.ttest_ind(mono, nonmono)

            results[pair_type] = {
                'mono_mean_error': mono_mean,
                'nonmono_mean_error': nonmono_mean,
                'advantage': advantage,
                'advantage_percent': 100 * advantage / mono_mean if mono_mean > 0 else 0,
                't_statistic': t_stat,
                'p_value': p_val
            }

    return results


def plot_linking_consistency_matrix(consistency, output_path):
    """Plot heatmap of linking consistency."""
    plt.figure(figsize=(10, 8))

    df = pd.DataFrame(consistency, index=CLASS_NAMES, columns=CLASS_NAMES)

    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                vmin=0, vmax=1, center=0.5,
                square=True, linewidths=0.5)

    plt.title('Linking Consistency Across Detection Runs\n(1.0 = always linked, 0.0 = never linked)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved linking consistency heatmap to {output_path}")


def plot_comparison_summary(comparison_results, output_path):
    """Plot comparison of monotonic vs non-monotonic on linked vs unlinked pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    categories = ['linked', 'unlinked']

    for ax, cat in zip(axes, categories):
        if cat not in comparison_results:
            continue

        res = comparison_results[cat]

        x = ['Monotonic', 'Non-monotonic']
        y = [res['mono_mean_error'], res['nonmono_mean_error']]
        colors = ['#e74c3c', '#3498db']

        bars = ax.bar(x, y, color=colors)

        ax.set_ylabel('Mean Confusion Error Rate')
        ax.set_title(f'{cat.capitalize()} Pairs\n'
                    f'Advantage: {res["advantage_percent"]:.1f}%\n'
                    f'p-value: {res["p_value"]:.3f}')

        # Add value labels
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Non-monotonic Advantage: Linked vs Unlinked Pairs', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze linking vs confusion correlation')
    parser.add_argument('--linking-dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'results', 'cifar10'),
                       help='Directory with linking detection results')
    parser.add_argument('--cnn-dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'results', 'cnn_10class'),
                       help='Directory with CNN 10-class results')
    parser.add_argument('--output-dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'results', 'analysis'),
                       help='Output directory for analysis results')
    parser.add_argument('--high-thresh', type=float, default=0.7,
                       help='Threshold for strongly linked pairs')
    parser.add_argument('--low-thresh', type=float, default=0.3,
                       help='Threshold for unlinked pairs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("LINKING VS CONFUSION CORRELATION ANALYSIS")
    print("=" * 60)

    # Step 1: Load and analyze linking matrices
    print("\n1. Loading linking detection results...")
    matrices = load_linking_matrices(args.linking_dir)

    if not matrices:
        print("ERROR: No linking matrices found!")
        return

    consistency = compute_linking_consistency(matrices)

    # Save consistency matrix
    consistency_df = pd.DataFrame(consistency, index=CLASS_NAMES, columns=CLASS_NAMES)
    consistency_path = os.path.join(args.output_dir, 'linking_consistency.csv')
    consistency_df.to_csv(consistency_path)
    print(f"Saved linking consistency to {consistency_path}")

    # Plot consistency heatmap
    plot_linking_consistency_matrix(consistency,
                                    os.path.join(args.output_dir, 'linking_consistency_heatmap.png'))

    # Categorize pairs
    linked_pairs, unlinked_pairs, ambiguous_pairs = categorize_pairs(
        consistency, args.high_thresh, args.low_thresh)

    print(f"\n2. Pair categorization (high={args.high_thresh}, low={args.low_thresh}):")
    print(f"   Strongly linked pairs: {len(linked_pairs)}")
    print(f"   Unlinked pairs: {len(unlinked_pairs)}")
    print(f"   Ambiguous pairs: {len(ambiguous_pairs)}")

    print("\n   Strongly linked pairs (consistency >= {:.0%}):".format(args.high_thresh))
    for i, j, cons in sorted(linked_pairs, key=lambda x: -x[2])[:10]:
        print(f"      {CLASS_NAMES[i]} - {CLASS_NAMES[j]}: {cons:.1%}")

    print("\n   Unlinked pairs (consistency <= {:.0%}):".format(args.low_thresh))
    for i, j, cons in sorted(unlinked_pairs, key=lambda x: x[2]):
        print(f"      {CLASS_NAMES[i]} - {CLASS_NAMES[j]}: {cons:.1%}")

    # Step 2: Load CNN results if available
    print("\n3. Loading CNN 10-class results...")
    if not os.path.exists(args.cnn_dir):
        print(f"   CNN results directory not found: {args.cnn_dir}")
        print("   Skipping confusion matrix analysis (experiments may still be running)")

        # Save linking summary for later use
        summary = {
            'n_linking_runs': len(matrices),
            'strongly_linked_pairs': [(CLASS_NAMES[i], CLASS_NAMES[j], cons)
                                      for i, j, cons in linked_pairs],
            'unlinked_pairs': [(CLASS_NAMES[i], CLASS_NAMES[j], cons)
                               for i, j, cons in unlinked_pairs],
            'ambiguous_pairs': [(CLASS_NAMES[i], CLASS_NAMES[j], cons)
                                for i, j, cons in ambiguous_pairs]
        }

        with open(os.path.join(args.output_dir, 'linking_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n   Saved linking summary for later analysis")
        return

    cnn_results = load_cnn_results(args.cnn_dir)

    if not cnn_results:
        print("   No CNN results found yet. Run this script again when experiments complete.")
        return

    # Step 3: Analyze correlation
    print("\n4. Analyzing linking vs confusion correlation...")
    analysis = analyze_linking_vs_errors(consistency,
                                         {k: v for k, v in cnn_results.items()},
                                         cnn_results)

    # Step 4: Compare on linked vs unlinked pairs
    print("\n5. Comparing activations on linked vs unlinked pairs...")
    comparison = compare_activations_on_pairs(cnn_results, linked_pairs, unlinked_pairs)

    if comparison:
        plot_comparison_summary(comparison,
                               os.path.join(args.output_dir, 'activation_comparison.png'))

        print("\n   RESULTS:")
        print("   " + "=" * 50)

        for pair_type in ['linked', 'unlinked']:
            if pair_type in comparison:
                res = comparison[pair_type]
                print(f"\n   {pair_type.upper()} PAIRS:")
                print(f"      Monotonic mean error:     {res['mono_mean_error']:.4f}")
                print(f"      Non-monotonic mean error: {res['nonmono_mean_error']:.4f}")
                print(f"      Non-mono advantage:       {res['advantage_percent']:+.2f}%")
                print(f"      p-value:                  {res['p_value']:.4f}")

        # Key test: Is advantage larger on linked pairs?
        if 'linked' in comparison and 'unlinked' in comparison:
            linked_adv = comparison['linked']['advantage_percent']
            unlinked_adv = comparison['unlinked']['advantage_percent']

            print("\n   " + "=" * 50)
            print("   HYPOTHESIS TEST:")
            print(f"      Advantage on linked pairs:   {linked_adv:+.2f}%")
            print(f"      Advantage on unlinked pairs: {unlinked_adv:+.2f}%")
            print(f"      Difference:                  {linked_adv - unlinked_adv:+.2f}%")

            if linked_adv > unlinked_adv:
                print("\n   SUPPORTS topological barrier hypothesis!")
            else:
                print("\n   Does NOT support topological barrier hypothesis")

    # Save full analysis
    full_results = {
        'linking_analysis': {
            'n_runs': len(matrices),
            'n_linked_pairs': len(linked_pairs),
            'n_unlinked_pairs': len(unlinked_pairs)
        },
        'comparison': comparison,
        'per_config': analysis
    }

    with open(os.path.join(args.output_dir, 'full_analysis.json'), 'w') as f:
        json.dump(full_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print(f"\nSaved full analysis to {os.path.join(args.output_dir, 'full_analysis.json')}")


if __name__ == '__main__':
    main()
