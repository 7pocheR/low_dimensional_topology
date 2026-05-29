#!/usr/bin/env python
"""
Generate publication-quality plots for linking number scaling experiments.

Plot 1: Line plot with min-max shading
Plot 2: Point plot with error bars (mean ± std, min-max whiskers)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style settings for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
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

# Model configurations
# Monotonic activations: ReLU, ELU, Leaky ReLU (dashed lines)
# Non-monotonic activations: GeLU, Swish (solid lines)
# Skip connections: enable |x| synthesis (dotted lines)
MODELS = {
    # Non-monotonic activations (solid lines)
    'gelu_noskip': {
        'label': 'GeLU',
        'color': '#2ca02c',  # green
        'marker': '^',
        'linestyle': '-',
        'zorder': 8,
    },
    'swish_noskip': {
        'label': 'Swish',
        'color': '#9467bd',  # purple
        'marker': 'D',
        'linestyle': '-',
        'zorder': 7,
    },
    # Skip connections (dotted lines) - can synthesize |x|
    'relu_skip': {
        'label': 'ReLU + Skip',
        'color': '#1f77b4',  # blue
        'marker': 's',
        'linestyle': ':',
        'zorder': 6,
    },
    'elu_skip': {
        'label': 'ELU + Skip',
        'color': '#17becf',  # cyan
        'marker': 'p',
        'linestyle': ':',
        'zorder': 5,
    },
    'leaky_relu_skip': {
        'label': 'LeakyReLU + Skip',
        'color': '#bcbd22',  # olive
        'marker': 'h',
        'linestyle': ':',
        'zorder': 4,
    },
    # Monotonic activations (dashed lines)
    'relu_noskip': {
        'label': 'ReLU',
        'color': '#d62728',  # red
        'marker': 'o',
        'linestyle': '--',
        'zorder': 3,
    },
    'elu_noskip': {
        'label': 'ELU',
        'color': '#ff7f0e',  # orange
        'marker': 'v',
        'linestyle': '--',
        'zorder': 2,
    },
    'leaky_relu_noskip': {
        'label': 'LeakyReLU',
        'color': '#8c564b',  # brown
        'marker': '<',
        'linestyle': '--',
        'zorder': 1,
    },
}

COPY_COUNTS = [1, 2, 5, 10, 20, 50]
COPY_COUNTS_EXTENDED = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def load_results(results_dir: Path, copy_counts: list = None):
    """Load all results and compute statistics."""
    if copy_counts is None:
        copy_counts = COPY_COUNTS_EXTENDED

    data = defaultdict(lambda: defaultdict(list))

    for d in sorted(results_dir.iterdir()):
        f = d / "results.json"
        if not f.exists():
            continue

        r = json.load(open(f))
        act = r["activation"]
        skip = "skip" if r["skip"] else "noskip"
        model = f"{act}_{skip}"
        copies = r["num_copies"]

        if model in MODELS:
            data[model][copies].append(r["test_acc"] * 100)  # Convert to percentage

    # Compute statistics
    stats = {}
    for model in MODELS:
        stats[model] = {}
        for copies in copy_counts:
            if copies in data[model] and len(data[model][copies]) > 0:
                accs = np.array(data[model][copies])
                stats[model][copies] = {
                    'mean': np.mean(accs),
                    'std': np.std(accs),
                    'min': np.min(accs),
                    'max': np.max(accs),
                    'median': np.median(accs),
                    'q25': np.percentile(accs, 25),
                    'q75': np.percentile(accs, 75),
                    'n_seeds': len(accs),
                    'all': accs,
                }

    return stats


def plot_line_with_shading(stats: dict, output_path: Path = None, copy_counts: list = None,
                           log_error: bool = False):
    """
    Plot 1: Line plot with min-max shading.
    - Solid line: mean across seeds
    - Shaded region: min to max range

    If log_error=True, plot error rate (100 - accuracy) on log scale.
    """
    if copy_counts is None:
        copy_counts = COPY_COUNTS

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_key, config in MODELS.items():
        if model_key not in stats:
            continue

        copies = []
        means = []
        mins = []
        maxs = []

        for c in copy_counts:
            if c in stats[model_key]:
                copies.append(c)
                if log_error:
                    # Error rate: lower is better, so min_acc -> max_error
                    means.append(100 - stats[model_key][c]['mean'])
                    mins.append(100 - stats[model_key][c]['max'])  # min error = max acc
                    maxs.append(100 - stats[model_key][c]['min'])  # max error = min acc
                else:
                    means.append(stats[model_key][c]['mean'])
                    mins.append(stats[model_key][c]['min'])
                    maxs.append(stats[model_key][c]['max'])

        if not copies:
            continue

        copies = np.array(copies)
        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        # Plot shaded region (min-max)
        ax.fill_between(copies, mins, maxs,
                        color=config['color'], alpha=0.2,
                        zorder=config['zorder'])

        # Plot mean line
        ax.plot(copies, means,
                label=config['label'],
                color=config['color'],
                marker=config['marker'],
                linestyle=config['linestyle'],
                linewidth=2,
                markersize=8,
                zorder=config['zorder'] + 10)

        # Add small markers at best values (max acc or min error)
        best_vals = mins if log_error else maxs
        ax.scatter(copies, best_vals,
                   color=config['color'],
                   marker='_',
                   s=100,
                   linewidths=2,
                   zorder=config['zorder'] + 5)

    ax.set_xlabel('Number of Link Copies (k)')
    ax.set_xscale('log')
    ax.set_xticks(copy_counts)
    ax.set_xticklabels([str(c) for c in copy_counts])

    if log_error:
        ax.set_ylabel('Error Rate (%)')
        ax.set_yscale('log')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, label='Chance', zorder=0)
        ax.set_ylim(0.01, 60)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_title('Error Rate vs Linking Number (log scale)\n(S² × S² in R⁵, Width-5, Depth-5 Networks)')
    else:
        ax.set_ylabel('Test Accuracy (%)')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, label='Chance', zorder=0)
        ax.set_ylim(45, 102)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.set_title('Accuracy vs Linking Number\n(S² × S² in R⁵, Width-5, Depth-5 Networks)')

    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig, ax


def plot_errorbar(stats: dict, output_path: Path = None, copy_counts: list = None,
                  log_error: bool = False):
    """
    Plot 2: Point plot with error bars on log-scale x-axis.
    - Central marker: median (small)
    - Thick inner bar: 25th to 75th percentile (IQR)
    - Thin whiskers: min to max
    - Line connecting max values to show best-seed trend

    If log_error=True, plot error rate (100 - accuracy) on log y-axis.
    """
    if copy_counts is None:
        copy_counts = COPY_COUNTS

    fig, ax = plt.subplots(figsize=(10, 6))

    # Only count models that actually have data for proper grouping
    models_with_data = [(k, v) for k, v in MODELS.items()
                        if k in stats and any(c in stats[k] for c in copy_counts)]
    n_models = len(models_with_data)

    if n_models == 0:
        print("No data to plot")
        return fig, ax

    # Use log-spaced offsets for grouping at each x position
    # Width in log space for grouping
    group_width = 0.15  # fraction of x position for total group width

    for i, (model_key, config) in enumerate(models_with_data):
        # Offset factor: centered around 0
        offset_factor = (i - (n_models - 1) / 2) / max(n_models - 1, 1) * group_width

        x_vals = []
        medians = []
        q25s = []
        q75s = []
        mins = []
        maxs = []

        for c in copy_counts:
            if c in stats[model_key]:
                # Apply multiplicative offset for log scale
                x_vals.append(c * (1 + offset_factor))
                if log_error:
                    # Error rate: 100 - accuracy
                    medians.append(100 - stats[model_key][c]['median'])
                    q25s.append(100 - stats[model_key][c]['q75'])  # Note: flipped for error
                    q75s.append(100 - stats[model_key][c]['q25'])  # Note: flipped for error
                    mins.append(100 - stats[model_key][c]['max'])  # min error = max acc
                    maxs.append(100 - stats[model_key][c]['min'])  # max error = min acc
                else:
                    medians.append(stats[model_key][c]['median'])
                    q25s.append(stats[model_key][c]['q25'])
                    q75s.append(stats[model_key][c]['q75'])
                    mins.append(stats[model_key][c]['min'])
                    maxs.append(stats[model_key][c]['max'])

        if not medians:
            continue

        x_vals = np.array(x_vals)
        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)
        mins = np.array(mins)
        maxs = np.array(maxs)

        # For log_error, we want to show "best" (lowest error) trend
        best_vals = mins if log_error else maxs

        # Min-max whiskers (thin)
        ax.errorbar(x_vals, medians,
                    yerr=[medians - mins, maxs - medians],
                    fmt='none',
                    ecolor=config['color'],
                    elinewidth=1,
                    capsize=3,
                    capthick=1,
                    alpha=0.6,
                    zorder=config['zorder'])

        # IQR bars (thick) - 25th to 75th percentile
        ax.errorbar(x_vals, medians,
                    yerr=[medians - q25s, q75s - medians],
                    fmt=config['marker'],
                    color=config['color'],
                    ecolor=config['color'],
                    elinewidth=2.5,
                    capsize=0,
                    markersize=4,
                    label=config['label'],
                    zorder=config['zorder'] + 10)

        # Line connecting best values (max acc or min error trend)
        ax.plot(x_vals, best_vals,
                color=config['color'],
                linestyle='-',
                linewidth=1.5,
                alpha=0.8,
                zorder=config['zorder'] + 5)
        # Small markers at best positions
        ax.scatter(x_vals, best_vals,
                   color=config['color'],
                   marker=config['marker'],
                   s=25,
                   zorder=config['zorder'] + 6)

    # Log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlabel('Number of Link Copies (k)')

    if log_error:
        ax.set_yscale('log')
        ax.set_ylabel('Error Rate (%)')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, label='Chance', zorder=0)
        ax.set_ylim(0.05, 60)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_title('Error Rate vs Linking Number (line = best, bar = IQR)\n(S² × S² in R⁵, Width-5, Depth-5 Networks)')
    else:
        ax.set_ylabel('Test Accuracy (%)')
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, label='Chance', zorder=0)
        ax.set_ylim(45, 102)
        ax.legend(loc='lower left', framealpha=0.9)
        ax.set_title('Accuracy vs Linking Number (line = max, bar = IQR)\n(S² × S² in R⁵, Width-5, Depth-5 Networks)')

    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig, ax


def print_table(stats: dict):
    """Print summary table."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Max Accuracy (Mean ± Std)")
    print("="*80)

    header = f"{'Model':<15} |"
    for c in COPY_COUNTS:
        header += f" c={c:<6} |"
    print(header)
    print("-" * len(header))

    for model_key, config in MODELS.items():
        if model_key not in stats:
            continue
        row = f"{config['label']:<15} |"
        for c in COPY_COUNTS:
            if c in stats[model_key]:
                s = stats[model_key][c]
                row += f" {s['max']:.1f}%   |"
            else:
                row += f" {'--':<7} |"
        print(row)

    print()
    print("Gap (ReLU+skip - ReLU no-skip):")
    for c in COPY_COUNTS:
        if c in stats['relu_skip'] and c in stats['relu_noskip']:
            gap = stats['relu_skip'][c]['max'] - stats['relu_noskip'][c]['max']
            print(f"  c={c}: {gap:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                        default='results/linking_seeds_v7',
                        help='Directory with results')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--extended', action='store_true',
                        help='Include c=100,200,500 if available')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    copy_counts = COPY_COUNTS_EXTENDED if args.extended else COPY_COUNTS

    print(f"Loading results from {results_dir}")
    stats = load_results(results_dir, copy_counts=copy_counts)

    print_table(stats)

    print("\nGenerating plots...")

    # Determine which copy counts have data
    available_copies = set()
    for model in stats:
        available_copies.update(stats[model].keys())
    available_copies = sorted([c for c in copy_counts if c in available_copies])

    print(f"Available copy counts: {available_copies}")

    # Plot 1: Line with shading (accuracy)
    plot_line_with_shading(
        stats,
        output_path=output_dir / 'linking_scaling_shaded.png',
        copy_counts=available_copies,
        log_error=False
    )
    plot_line_with_shading(
        stats,
        output_path=output_dir / 'linking_scaling_shaded.pdf',
        copy_counts=available_copies,
        log_error=False
    )

    # Plot 2: Line with shading (log error rate)
    plot_line_with_shading(
        stats,
        output_path=output_dir / 'linking_scaling_logerror.png',
        copy_counts=available_copies,
        log_error=True
    )
    plot_line_with_shading(
        stats,
        output_path=output_dir / 'linking_scaling_logerror.pdf',
        copy_counts=available_copies,
        log_error=True
    )

    # Plot 3: Error bars with quartiles (accuracy)
    plot_errorbar(
        stats,
        output_path=output_dir / 'linking_scaling_errorbar.png',
        copy_counts=available_copies,
        log_error=False
    )
    plot_errorbar(
        stats,
        output_path=output_dir / 'linking_scaling_errorbar.pdf',
        copy_counts=available_copies,
        log_error=False
    )

    # Plot 4: Error bars with log error rate
    plot_errorbar(
        stats,
        output_path=output_dir / 'linking_scaling_errorbar_logerror.png',
        copy_counts=available_copies,
        log_error=True
    )
    plot_errorbar(
        stats,
        output_path=output_dir / 'linking_scaling_errorbar_logerror.pdf',
        copy_counts=available_copies,
        log_error=True
    )

    print("\nDone!")
    plt.show()


if __name__ == '__main__':
    main()
