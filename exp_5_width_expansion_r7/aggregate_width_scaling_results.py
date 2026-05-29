#!/usr/bin/env python
"""
Aggregate width-scaling experiment results into summary tables.

Usage:
    python aggregate_width_scaling_results.py [--results_dir /path/to/results]
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_results(results_dir: Path):
    """Load all experiment results from directory."""
    results = []

    for exp_dir in results_dir.glob("n*"):
        result_file = exp_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
                result['exp_dir'] = str(exp_dir.name)
                results.append(result)

    return pd.DataFrame(results)


def categorize_activation(act):
    """Categorize activation as monotonic or non-monotonic."""
    monotonic = ['relu', 'elu', 'selu', 'leaky_relu']
    return 'monotonic' if act in monotonic else 'non-monotonic'


def create_summary_tables(df):
    """Create summary tables from results."""

    if df.empty:
        print("No results found!")
        return

    # Add derived columns
    df['activation_type'] = df['activation'].apply(categorize_activation)
    df['arch'] = df['skip'].apply(lambda x: 'FFN+skip' if x else 'FFN')

    print("=" * 80)
    print("WIDTH-SCALING EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Table 1: Test accuracy by n and activation type
    print("\n" + "-" * 80)
    print("TABLE 1: Mean Test Accuracy by Dimension and Activation Type")
    print("-" * 80)
    table1 = df.pivot_table(
        values='test_acc',
        index='n',
        columns='activation_type',
        aggfunc='mean'
    )
    table1['gap'] = table1['non-monotonic'] - table1['monotonic']
    print(table1.round(4).to_string())

    # Table 2: Test accuracy by n and architecture
    print("\n" + "-" * 80)
    print("TABLE 2: Mean Test Accuracy by Dimension and Architecture")
    print("-" * 80)
    table2 = df.pivot_table(
        values='test_acc',
        index='n',
        columns='arch',
        aggfunc='mean'
    )
    if 'FFN+skip' in table2.columns and 'FFN' in table2.columns:
        table2['skip_effect'] = table2['FFN+skip'] - table2['FFN']
    print(table2.round(4).to_string())

    # Table 3: Test accuracy by activation (detailed)
    print("\n" + "-" * 80)
    print("TABLE 3: Mean Test Accuracy by Activation (across all n)")
    print("-" * 80)
    table3 = df.groupby('activation').agg({
        'test_acc': ['mean', 'std', 'min', 'max']
    }).round(4)
    table3.columns = ['mean', 'std', 'min', 'max']
    table3 = table3.sort_values('mean', ascending=False)
    print(table3.to_string())

    # Table 4: Test accuracy by depth (effect of depth)
    print("\n" + "-" * 80)
    print("TABLE 4: Mean Test Accuracy by Depth and Activation Type")
    print("-" * 80)
    table4 = df.pivot_table(
        values='test_acc',
        index='depth',
        columns='activation_type',
        aggfunc='mean'
    )
    print(table4.round(4).to_string())

    # Table 5: Best configurations per n
    print("\n" + "-" * 80)
    print("TABLE 5: Best and Worst Configurations per Dimension")
    print("-" * 80)
    for n in sorted(df['n'].unique()):
        n_df = df[df['n'] == n]
        best = n_df.loc[n_df['test_acc'].idxmax()]
        worst = n_df.loc[n_df['test_acc'].idxmin()]

        print(f"\nn={n} (R^{2*n+1}, width {2*n+1}):")
        print(f"  Best:  {best['test_acc']:.4f} - {best['activation']}, depth={best['depth']}, skip={best['skip']}")
        print(f"  Worst: {worst['test_acc']:.4f} - {worst['activation']}, depth={worst['depth']}, skip={worst['skip']}")

        # Monotonic best vs non-mono best
        mono_best = n_df[n_df['activation_type'] == 'monotonic']['test_acc'].max()
        nonmono_best = n_df[n_df['activation_type'] == 'non-monotonic']['test_acc'].max()
        print(f"  Monotonic best:     {mono_best:.4f}")
        print(f"  Non-monotonic best: {nonmono_best:.4f}")
        print(f"  Gap:                {nonmono_best - mono_best:+.4f}")

    # Table 6: NegSlopeReLU dose-response
    print("\n" + "-" * 80)
    print("TABLE 6: NegSlopeReLU Dose-Response (α effect)")
    print("-" * 80)
    negslope_df = df[df['activation'].str.startswith('negslope')]
    if not negslope_df.empty:
        table6 = negslope_df.pivot_table(
            values='test_acc',
            index='n',
            columns='activation',
            aggfunc='mean'
        )
        # Reorder columns by α magnitude
        cols = ['negslope_001', 'negslope_01', 'negslope_05', 'negslope_10']
        cols = [c for c in cols if c in table6.columns]
        table6 = table6[cols]
        print(table6.round(4).to_string())
    else:
        print("No NegSlopeReLU results found")

    return {
        'by_type': table1,
        'by_arch': table2,
        'by_activation': table3,
        'by_depth': table4,
    }


def save_summary(df, output_path: Path):
    """Save summary to CSV and JSON."""
    df.to_csv(output_path / 'all_results.csv', index=False)

    # Also save pivot tables
    summary = {}
    for col in ['n', 'depth', 'activation', 'skip']:
        summary[col] = df.groupby(col)['test_acc'].mean().to_dict()

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='results_width_r7',
                        help='Directory containing experiment results')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading results from: {results_dir}")

    df = load_results(results_dir)
    print(f"Loaded {len(df)} experiment results")

    if len(df) > 0:
        tables = create_summary_tables(df)
        save_summary(df, results_dir)


if __name__ == "__main__":
    main()
