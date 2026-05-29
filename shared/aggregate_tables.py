"""
Aggregate per-trial JSON outputs into mean +/- std cells for Tables 2, 3, 4.

Table 2: glob exp_1_table2_relu_gelu_hopf/results/table2/d*_*_*/result.json
Table 3: glob exp_2_table3_plain_relu_vs_resnet_hopf/results/table3/d*_*_*/result.json
Table 4: glob exp_4_higher_dim_r5_multicopy/results/linking_seeds_v7/n2_d5_<act>_<skip>_c<k>_s*/results.json

For Tables 2/3 the cell statistic is best_val_acc (matches the original scripts).
For Table 4 the cell statistic is test_acc * 100 (v7 records fractions).

Prints LaTeX-ready rows for each table.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dir(pattern):
    rows = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"WARN: failed to read {path}: {e}")
    return rows


def aggregate(rows, key_fn, value_fn):
    cells = defaultdict(list)
    for r in rows:
        key = key_fn(r)
        if key is None:
            continue
        val = value_fn(r)
        if val is None or np.isnan(val):
            continue
        cells[key].append(val)
    return cells


def fmt_mean_std(vals, digits=1):
    if not vals:
        return "n/a"
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.{digits}f} $\\pm$ {arr.std(ddof=1):.{digits}f}"


def fmt_mean_std_max(vals, digits=1):
    if not vals:
        return "n/a"
    arr = np.array(vals, dtype=float)
    return (
        f"{arr.mean():.{digits}f} $\\pm$ {arr.std(ddof=1):.{digits}f} "
        f"({arr.max():.{digits}f})"
    )


def report_table2(base):
    rows = load_dir(os.path.join(base, 'd*_*_*', 'result.json'))
    print(f"\n[Table 2] loaded {len(rows)} trials from {base}")
    cells = aggregate(
        rows,
        key_fn=lambda r: (r['depth'], r['activation']),
        value_fn=lambda r: r['best_val_acc'],
    )
    depths = [3, 5, 8, 12, 16, 20]
    print("\nTable 2 (Mean +/- std (max) test/val accuracy, %, best across early-stopped epochs):")
    print(f"{'Depth':<8} " + " ".join(f"{d:>14}" for d in depths))
    for act in ['relu', 'gelu']:
        cell_strs = []
        for d in depths:
            vals = cells.get((d, act), [])
            cell_strs.append(f"{fmt_mean_std_max(vals)} (n={len(vals)})")
        print(f"{act:<8} " + " | ".join(cell_strs))

    print("\nLaTeX-ready table 2 row template:")
    for act in ['relu', 'gelu']:
        cells_s = []
        for d in depths:
            vals = cells.get((d, act), [])
            cells_s.append(fmt_mean_std_max(vals))
        print(f"  {act.upper():<5} & " + " & ".join(cells_s) + r" \\")


def report_table3(base):
    rows = load_dir(os.path.join(base, 'd*_*_*', 'result.json'))
    print(f"\n[Table 3] loaded {len(rows)} trials from {base}")
    cells = aggregate(
        rows,
        key_fn=lambda r: (r['depth'], r['arch']),
        value_fn=lambda r: r['best_val_acc'],
    )
    depths = [3, 4, 5, 6, 7, 8]
    print("\nTable 3 (Mean +/- std (max) val accuracy, %, best across early-stopped epochs):")
    print(f"{'Depth':<8} " + " ".join(f"{d:>14}" for d in depths))
    for arch in ['relu', 'resnet']:
        cell_strs = []
        for d in depths:
            vals = cells.get((d, arch), [])
            cell_strs.append(f"{fmt_mean_std_max(vals)} (n={len(vals)})")
        print(f"{arch:<8} " + " | ".join(cell_strs))

    print("\nLaTeX-ready table 3 row template:")
    label = {'relu': 'Plain ReLU', 'resnet': 'ResNet'}
    for arch in ['relu', 'resnet']:
        cells_s = []
        for d in depths:
            vals = cells.get((d, arch), [])
            cells_s.append(fmt_mean_std_max(vals))
        print(f"  {label[arch]:<10} & " + " & ".join(cells_s) + r" \\")


V7_PATTERN = re.compile(r'n2_d5_(?P<act>[a-z0-9_]+)_(?P<skip>skip|noskip)_c(?P<k>\d+)_s\d+')


def report_table4(base):
    print(f"\n[Table 4] scanning {base}")
    cells = defaultdict(list)
    n_seen = 0
    for d in os.listdir(base):
        m = V7_PATTERN.fullmatch(d)
        if not m:
            continue
        act = m.group('act')
        skip = m.group('skip') == 'skip'
        k = int(m.group('k'))
        res = os.path.join(base, d, 'results.json')
        if not os.path.isfile(res):
            continue
        try:
            with open(res) as f:
                obj = json.load(f)
        except Exception as e:
            print(f"WARN: bad json {res}: {e}")
            continue
        test = obj.get('test_acc')
        if test is None:
            continue
        cells[(act, skip, k)].append(float(test) * 100.0)
        n_seen += 1
    print(f"[Table 4] loaded {n_seen} trials")

    target_models = [
        ('relu', False, 'ReLU'),
        ('relu', True, 'ReLU+Skip'),
        ('gelu', False, 'GELU'),
        ('swish', False, 'Swish'),
    ]
    ks = [1, 2, 5, 10, 20, 50]

    print("\nTable 4 (Mean +/- std test accuracy, %, R^5 multi-copy linking):")
    print(f"{'Model':<12} " + " ".join(f"{k:>14}" for k in ks))
    for act, skip, label in target_models:
        cell_strs = []
        for k in ks:
            vals = cells.get((act, skip, k), [])
            cell_strs.append(f"{fmt_mean_std(vals)} (n={len(vals)})")
        print(f"{label:<12} " + " | ".join(cell_strs))

    print("\nLaTeX-ready table 4 row template:")
    for act, skip, label in target_models:
        cells_s = []
        for k in ks:
            vals = cells.get((act, skip, k), [])
            cells_s.append(fmt_mean_std(vals))
        print(f"  {label:<11} & " + " & ".join(cells_s) + r" \\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table2_dir', default=os.path.join(ROOT, 'exp_1_table2_relu_gelu_hopf', 'results', 'table2'))
    parser.add_argument('--table3_dir', default=os.path.join(ROOT, 'exp_2_table3_plain_relu_vs_resnet_hopf', 'results', 'table3'))
    parser.add_argument('--table4_dir', default=os.path.join(ROOT, 'exp_4_higher_dim_r5_multicopy', 'results', 'linking_seeds_v7'))
    parser.add_argument('--skip_t2', action='store_true')
    parser.add_argument('--skip_t3', action='store_true')
    parser.add_argument('--skip_t4', action='store_true')
    args = parser.parse_args()

    if not args.skip_t2 and os.path.isdir(args.table2_dir):
        report_table2(args.table2_dir)
    if not args.skip_t3 and os.path.isdir(args.table3_dir):
        report_table3(args.table3_dir)
    if not args.skip_t4 and os.path.isdir(args.table4_dir):
        report_table4(args.table4_dir)


if __name__ == '__main__':
    main()
