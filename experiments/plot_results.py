#!/usr/bin/env python3
"""
Plot experiment results for thesis figures.

Usage:
    python experiments/plot_results.py --input results/full.csv --output-dir figures/
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip3 install matplotlib")
    print("Falling back to text-only output.")


VARIANT_LABELS = {
    'disagg-static': 'Disagg-Static',
    'coloc-sarathi': 'Coloc-Sarathi',
    'cbs-nomig': 'CBS-NoMig',
    'cbs-norole': 'CBS-NoRole',
    'cbs-full': 'CBS-Full',
}
VARIANT_ORDER = ['disagg-static', 'coloc-sarathi', 'cbs-nomig', 'cbs-norole', 'cbs-full']
COLORS = {
    'disagg-static': '#1f77b4',
    'coloc-sarathi': '#ff7f0e',
    'cbs-nomig': '#2ca02c',
    'cbs-norole': '#9467bd',
    'cbs-full': '#d62728',
}
MARKERS = {
    'disagg-static': 'o',
    'coloc-sarathi': 's',
    'cbs-nomig': '^',
    'cbs-norole': 'D',
    'cbs-full': '*',
}

# PLACEHOLDER_PLOT_FUNCTIONS


def plot_goodput_vs_rate(df, workload, output_dir):
    """Figure: Goodput vs arrival rate for each variant."""
    wl_df = df[df.workload == workload]
    if wl_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for v in VARIANT_ORDER:
        vdf = wl_df[wl_df.variant == v].sort_values('rate')
        if vdf.empty:
            continue
        ax.plot(vdf.rate, vdf.goodput, label=VARIANT_LABELS[v],
                color=COLORS[v], marker=MARKERS[v], markersize=8, linewidth=2)

    ax.set_xlabel('Arrival Rate (req/s)', fontsize=13)
    ax.set_ylabel('Goodput (req/s)', fontsize=13)
    ax.set_title(f'Goodput vs Arrival Rate ({workload})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'goodput_{workload}.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, f'goodput_{workload}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved goodput_{workload}.pdf/png")


def plot_slo_vs_rate(df, workload, output_dir):
    """Figure: SLO attainment vs arrival rate."""
    wl_df = df[df.workload == workload]
    if wl_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for v in VARIANT_ORDER:
        vdf = wl_df[wl_df.variant == v].sort_values('rate')
        if vdf.empty:
            continue
        ax.plot(vdf.rate, vdf.slo_attainment, label=VARIANT_LABELS[v],
                color=COLORS[v], marker=MARKERS[v], markersize=8, linewidth=2)

    ax.set_xlabel('Arrival Rate (req/s)', fontsize=13)
    ax.set_ylabel('SLO Attainment (%)', fontsize=13)
    ax.set_title(f'SLO Attainment vs Arrival Rate ({workload})', fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'slo_{workload}.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, f'slo_{workload}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved slo_{workload}.pdf/png")


def plot_latency_bars(df, workload, rate, output_dir):
    """Figure: P99 TTFT and P99 TPOT bar chart at a specific rate."""
    wl_df = df[(df.workload == workload) & (df.rate == rate)]
    if wl_df.empty:
        return

    variants = [v for v in VARIANT_ORDER if v in wl_df.variant.values]
    labels = [VARIANT_LABELS[v] for v in variants]
    ttft_vals = [wl_df[wl_df.variant == v].p99_ttft.values[0] for v in variants]
    tpot_vals = [wl_df[wl_df.variant == v].p99_tpot.values[0] for v in variants]

    x = range(len(variants))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x, ttft_vals, width, color=[COLORS[v] for v in variants])
    ax1.set_ylabel('P99 TTFT (ms)', fontsize=13)
    ax1.set_title(f'P99 TTFT @ {rate} req/s ({workload})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    bars2 = ax2.bar(x, tpot_vals, width, color=[COLORS[v] for v in variants])
    ax2.set_ylabel('P99 TPOT (ms)', fontsize=13)
    ax2.set_title(f'P99 TPOT @ {rate} req/s ({workload})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'latency_{workload}_{int(rate)}.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, f'latency_{workload}_{int(rate)}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved latency_{workload}_{int(rate)}.pdf/png")


def print_text_summary(df):
    """Print text summary when matplotlib is not available."""
    for wl in df.workload.unique():
        wl_df = df[df.workload == wl]
        print(f"\n{'='*80}")
        print(f"Workload: {wl}")
        print(f"{'='*80}")
        for rate in sorted(wl_df.rate.unique()):
            print(f"\n  Rate: {rate} req/s")
            print(f"  {'Variant':<18s} {'Goodput':>8s} {'SLO%':>6s} {'P99TTFT':>8s} {'P99TPOT':>8s} {'Coloc':>6s} {'Migr':>5s}")
            print(f"  {'-'*60}")
            rdf = wl_df[wl_df.rate == rate]
            for v in VARIANT_ORDER:
                row = rdf[rdf.variant == v]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f"  {VARIANT_LABELS[v]:<18s} {r.goodput:8.1f} {r.slo_attainment:6.1f} {r.p99_ttft:8.0f} {r.p99_tpot:8.0f} {r.n_colocated:6.0f} {r.n_migrations:5.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='CSV from run_all.py')
    p.add_argument('--output-dir', default='figures/')
    p.add_argument('--rate-for-bars', type=float, default=12.0, help='Rate for bar chart')
    args = p.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    print_text_summary(df)

    if not HAS_MPL:
        return

    print("\nGenerating figures...")
    for wl in df.workload.unique():
        plot_goodput_vs_rate(df, wl, args.output_dir)
        plot_slo_vs_rate(df, wl, args.output_dir)
        plot_latency_bars(df, wl, args.rate_for_bars, args.output_dir)

    print("Done!")


if __name__ == '__main__':
    main()
