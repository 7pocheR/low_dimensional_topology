#!/usr/bin/env python
"""
Analyze Linking Number Scaling Experiments

Creates publication-quality plots showing how accuracy varies with linking number
for different activation functions.

Expected result: Curves showing ReLU decreasing fastest toward chance (50%),
while GeLU/Swish decrease more slowly.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Style settings for publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Model configurations and colors
MODELS = {
    'relu_noskip': {'label': 'ReLU', 'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'relu_skip': {'label': 'ReLU + Skip', 'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'gelu_noskip': {'label': 'GeLU', 'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
    'swish_noskip': {'label': 'Swish', 'color': '#1f77b4', 'marker': 'D', 'linestyle': '-'},
}


def load_results(results_dir: Path, n: int = 2, depth: int = 3):
    """Load results from linking scaling experiments (single seed)."""
    results = {model: {} for model in MODELS}

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Parse directory name: n{N}_d{DEPTH}_{ACT}_{SKIP}_c{COPIES}
        parts = subdir.name.split('_')
        if len(parts) < 5:
            continue

        try:
            dir_n = int(parts[0][1:])
            dir_depth = int(parts[1][1:])
            activation = parts[2]
            skip_str = parts[3]
            num_copies = int(parts[4][1:])
        except (ValueError, IndexError):
            continue

        if dir_n != n or dir_depth != depth:
            continue

        model_key = f"{activation}_{skip_str}"
        if model_key not in MODELS:
            continue

        # Load results.json
        results_file = subdir / 'results.json'
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        results[model_key][num_copies] = {
            'test_acc': data.get('test_acc', data.get('val_acc', 0)),
            'val_acc': data.get('val_acc', 0),
            'best_epoch': data.get('best_epoch', 0),
            'total_epochs': data.get('total_epochs', 0),
        }

    return results


def load_results_multiseed(results_dir: Path, n: int = 2, depth: int = 3):
    """
    Load results from multi-seed linking scaling experiments.
    Returns max, mean, std across seeds for each (model, copies) pair.
    """
    # Collect all results by (model, copies)
    raw_results = {}  # {(model, copies): [acc1, acc2, ...]}

    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Parse: n{N}_d{DEPTH}_{ACT}_{SKIP}_c{COPIES}_s{SEED}
        parts = subdir.name.split('_')
        if len(parts) < 6:
            continue

        try:
            dir_n = int(parts[0][1:])
            dir_depth = int(parts[1][1:])
            activation = parts[2]
            skip_str = parts[3]
            num_copies = int(parts[4][1:])
            seed = int(parts[5][1:])
        except (ValueError, IndexError):
            continue

        if dir_n != n or dir_depth != depth:
            continue

        model_key = f"{activation}_{skip_str}"
        if model_key not in MODELS:
            continue

        # Load results.json
        results_file = subdir / 'results.json'
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)

        key = (model_key, num_copies)
        if key not in raw_results:
            raw_results[key] = []
        raw_results[key].append(data.get('test_acc', data.get('val_acc', 0)))

    # Aggregate: max, mean, std
    results = {model: {} for model in MODELS}
    for (model_key, num_copies), accs in raw_results.items():
        accs = np.array(accs)
        results[model_key][num_copies] = {
            'max_acc': float(np.max(accs)),
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'n_seeds': len(accs),
            'all_accs': accs.tolist(),
        }

    return results


def plot_linking_scaling(results: dict, output_path: Path = None, title: str = None):
    """Create the main linking number scaling plot."""
    fig, ax = plt.subplots()

    for model_key, model_config in MODELS.items():
        if model_key not in results or not results[model_key]:
            continue

        copies = sorted(results[model_key].keys())
        accs = [results[model_key][c]['test_acc'] * 100 for c in copies]

        ax.plot(copies, accs,
                label=model_config['label'],
                color=model_config['color'],
                marker=model_config['marker'],
                linestyle=model_config['linestyle'],
                markersize=8,
                linewidth=2)

    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, label='Chance')

    ax.set_xlabel('Linking Number (Number of Link Copies)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xscale('log')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Accuracy vs Linking Number\n(S² × S² linked in R⁵, Width-5 Networks)')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(45, 100)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")

    return fig, ax


def plot_gap_analysis(results: dict, output_path: Path = None):
    """Plot the accuracy gap between ReLU and non-monotonic activations."""
    fig, ax = plt.subplots()

    relu_results = results.get('relu_noskip', {})
    if not relu_results:
        print("No ReLU results found")
        return None, None

    copies = sorted(relu_results.keys())

    for model_key in ['gelu_noskip', 'swish_noskip', 'relu_skip']:
        if model_key not in results or not results[model_key]:
            continue

        model_config = MODELS[model_key]
        gaps = []
        valid_copies = []

        for c in copies:
            if c in results[model_key]:
                relu_acc = relu_results[c]['test_acc']
                model_acc = results[model_key][c]['test_acc']
                gap = (model_acc - relu_acc) * 100
                gaps.append(gap)
                valid_copies.append(c)

        ax.plot(valid_copies, gaps,
                label=f"{model_config['label']} - ReLU",
                color=model_config['color'],
                marker=model_config['marker'],
                linestyle=model_config['linestyle'],
                markersize=8,
                linewidth=2)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('Linking Number (Number of Link Copies)')
    ax.set_ylabel('Accuracy Gap vs ReLU (%)')
    ax.set_xscale('log')
    ax.set_title('Topological Advantage: Non-Monotonic vs Monotonic')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved gap plot to {output_path}")

    return fig, ax


def print_summary_table(results: dict):
    """Print a summary table of results."""
    all_copies = set()
    for model_results in results.values():
        all_copies.update(model_results.keys())
    all_copies = sorted(all_copies)

    if not all_copies:
        print("No results found")
        return

    # Header
    header = "Copies |"
    for model_key, config in MODELS.items():
        if model_key in results and results[model_key]:
            header += f" {config['label']:>12} |"
    print(header)
    print("-" * len(header))

    # Data rows
    for copies in all_copies:
        row = f"{copies:>6} |"
        for model_key in MODELS:
            if model_key in results and copies in results[model_key]:
                acc = results[model_key][copies]['test_acc'] * 100
                row += f" {acc:>11.1f}% |"
            elif model_key in results and results[model_key]:
                row += f" {'---':>12} |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='Analyze linking scaling experiments')
    parser.add_argument('--results-dir', type=str,
                        default='results/linking_seeds_v7',
                        help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--n', type=int, default=2, help='Dimension parameter')
    parser.add_argument('--depth', type=int, default=3, help='Network depth')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir, n=args.n, depth=args.depth)

    print("\n=== Summary Table ===\n")
    print_summary_table(results)

    print("\n=== Generating Plots ===\n")

    # Main scaling plot
    plot_linking_scaling(
        results,
        output_path=output_dir / 'linking_scaling.png',
        title=f'Width-{2*args.n+1} Networks on Linked S^{args.n} × S^{args.n}'
    )

    # Gap analysis plot
    plot_gap_analysis(
        results,
        output_path=output_dir / 'linking_gap.png'
    )

    plt.show()


if __name__ == '__main__':
    main()
