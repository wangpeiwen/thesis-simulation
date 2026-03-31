#!/usr/bin/env python3
"""
thesis-simulation: 毕设论文对比实验
对比 Disagg-Static / Coloc-Sarathi / CBS-NoMig / CBS-NoRole / CBS-Full

Usage:
    python experiments/run_all.py --workload uniform --rates 4,8,12 --output results/uniform.csv
    python experiments/run_all.py --workload bursty --rates 8 --output results/bursty.csv
    python experiments/run_all.py --workload all --rates 2,4,6,8,10,12,14,16 --output results/full.csv
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import simpy

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simdistserve.base.organize_data import (
    organize_request_df, organize_request_event_df, calculate_per_request_latency,
)
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import convert_pd_pair_to_request, get_gamma_interarrival
from simdistserve.clusters.cbs import CBSCluster
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens

# PLACEHOLDER_WORKLOAD_GEN


# ---- Workload generation ----

def generate_poisson_arrivals(N, rate, seed=42):
    np.random.seed(seed)
    intervals = np.random.exponential(1.0 / rate, N - 1) * 1000  # ms
    return [0] + list(intervals)


def generate_bursty_arrivals(N, base_rate, burst_period=30.0, burst_duration=5.0,
                              burst_multiplier=4.0, seed=42):
    """Poisson with periodic bursts: every burst_period seconds, rate *= burst_multiplier for burst_duration."""
    np.random.seed(seed)
    arrivals_abs = []
    t = 0.0
    for _ in range(N):
        cycle_pos = t % burst_period
        rate = base_rate * burst_multiplier if cycle_pos < burst_duration else base_rate
        t += np.random.exponential(1.0 / rate)
        arrivals_abs.append(t)
    # Convert to inter-arrival (ms)
    intervals = [0] + [max(0, (arrivals_abs[i] - arrivals_abs[i - 1]) * 1000) for i in range(1, N)]
    return intervals


def generate_workload(N, rate, config, seed=42):
    """Generate requests and arrival times from config."""
    random.seed(seed)
    pmin, pmax = config['prompt_range']
    omin, omax = config['output_range']
    pairs = [(random.randint(pmin, pmax), random.randint(omin, omax)) for _ in range(N)]
    requests = convert_pd_pair_to_request(pairs)

    arrival_type = config.get('arrival', 'poisson')
    if arrival_type == 'bursty':
        arrival = generate_bursty_arrivals(
            N, rate,
            burst_period=config.get('burst_period', 30.0),
            burst_duration=config.get('burst_duration', 5.0),
            burst_multiplier=config.get('burst_multiplier', 4.0),
            seed=seed,
        )
    else:
        arrival = generate_poisson_arrivals(N, rate, seed=seed)
    return requests, arrival


# ---- Cluster factories ----

VARIANTS = ['disagg-static', 'coloc-sarathi', 'cbs-nomig', 'cbs-norole', 'cbs-full']


def make_cluster(env, variant, args, model_type, pmt):
    """Create cluster for the given variant."""
    n_p, n_d = args.n_prefill, args.n_decode

    # Base worker config for distserve-style backends
    wc_dist = WorkerConfig(
        model_type=model_type, TP=args.tp, TP_Prefill=args.tp, TP_Decode=args.tp,
        prefill_max_batch_size=10**7, decode_max_batch_size=10**7,
        prefill_max_tokens=pmt, decode_max_tokens=pmt,
        enable_chunked_prefill=False, engine_type='distserve',
    )

    if variant == 'disagg-static':
        return DisaggCluster(
            env=env, N_prefill_instance=n_p, N_decode_instance=n_d,
            PP_prefill=1, PP_decode=1, worker_configs=wc_dist,
        )

    elif variant == 'coloc-sarathi':
        n_total = n_p + n_d  # All GPUs do mixed scheduling
        wc_coloc = WorkerConfig(
            model_type=model_type, TP=args.tp, TP_Prefill=args.tp, TP_Decode=args.tp,
            prefill_max_batch_size=10**7, decode_max_batch_size=10**7,
            prefill_max_tokens=512,  # Sarathi chunk size
            decode_max_tokens=pmt,
            enable_chunked_prefill=True,  # Key: chunked prefill
            engine_type='vllm',
        )
        return VLLMCluster(env=env, N_instance=n_total, PP=1, worker_configs=wc_coloc)

    elif variant == 'cbs-nomig':
        return CBSCluster(
            env=env, N_prefill_instance=n_p, N_decode_instance=n_d,
            PP_prefill=1, PP_decode=1, worker_configs=wc_dist,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.kv_latency,
        )

    elif variant == 'cbs-norole':
        return CBSCluster(
            env=env, N_prefill_instance=n_p, N_decode_instance=n_d,
            PP_prefill=1, PP_decode=1, worker_configs=wc_dist,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.kv_latency,
            enable_migration=True, slo_tpot=args.slo_tpot,
        )

    elif variant == 'cbs-full':
        return CBSCluster(
            env=env, N_prefill_instance=n_p, N_decode_instance=n_d,
            PP_prefill=1, PP_decode=1, worker_configs=wc_dist,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.kv_latency,
            enable_migration=True, enable_role_adaptation=True,
            slo_tpot=args.slo_tpot, slo_ttft=args.slo_ttft,
        )

    raise ValueError(f"Unknown variant: {variant}")


# PLACEHOLDER_RUN_AND_MAIN


def run_one(variant, rate, config, args, model_type, pmt):
    """Run a single experiment, return metrics dict."""
    requests, arrival = generate_workload(args.N, rate, config, seed=args.seed)

    env = simpy.Environment()
    cluster = make_cluster(env, variant, args, model_type, pmt)
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    env.run()

    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    latency_df = calculate_per_request_latency(request_event_df, request_df.output_lens)

    ttft = latency_df['first_token_latency']
    tpot = latency_df['tpot']
    slo_ok = ((ttft <= args.slo_ttft) & (tpot <= args.slo_tpot)).sum()

    # Goodput: SLO-meeting requests per second
    last_time = max(r.log[-1][0] for r in requests if r.log) / 1000.0  # seconds
    goodput = slo_ok / last_time if last_time > 0 else 0

    return {
        'workload': config['name'],
        'variant': variant,
        'rate': rate,
        'n_gpu': args.n_prefill + args.n_decode,
        'n_colocated': getattr(cluster.scheduler, 'n_colocated', 0),
        'n_migrations': getattr(cluster.scheduler, 'n_migrations', 0),
        'n_role_switches': getattr(cluster.scheduler, 'n_role_switches', 0),
        'p50_ttft': ttft.quantile(0.5),
        'p90_ttft': ttft.quantile(0.9),
        'p99_ttft': ttft.quantile(0.99),
        'p50_tpot': tpot.quantile(0.5),
        'p90_tpot': tpot.quantile(0.9),
        'p99_tpot': tpot.quantile(0.99),
        'slo_attainment': slo_ok / args.N * 100,
        'goodput': goodput,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Thesis simulation experiments')
    p.add_argument('--model', default='facebook/opt-13b')
    p.add_argument('--N', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rates', default='2,4,6,8,10,12', help='Comma-separated rates')
    p.add_argument('--workload', default='uniform', help='uniform/bursty/longctx/all')
    p.add_argument('--variants', default='all', help='Comma-separated or "all"')
    p.add_argument('--n-prefill', type=int, default=2)
    p.add_argument('--n-decode', type=int, default=6)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--slo-ttft', type=float, default=2000.0)
    p.add_argument('--slo-tpot', type=float, default=100.0)
    p.add_argument('--cbs-mu', type=float, default=0.5)
    p.add_argument('--cbs-lambda', type=float, default=0.1)
    p.add_argument('--cbs-kappa', type=float, default=0.1)
    p.add_argument('--kv-latency', type=float, default=5.0)
    p.add_argument('--output', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_type = ModelTypes.model_str_to_object(args.model)
    pmt = get_max_num_tokens(model_type, args.tp, 1)
    rates = [float(r) for r in args.rates.split(',')]

    # Load workload configs
    config_dir = Path(__file__).parent / 'configs'
    if args.workload == 'all':
        workloads = ['uniform', 'bursty', 'longctx']
    else:
        workloads = [w.strip() for w in args.workload.split(',')]

    configs = {}
    for wl in workloads:
        with open(config_dir / f'{wl}.json') as f:
            configs[wl] = json.load(f)

    if args.variants == 'all':
        variants = VARIANTS
    else:
        variants = [v.strip() for v in args.variants.split(',')]

    results = []
    total = len(workloads) * len(rates) * len(variants)
    done = 0

    for wl_name in workloads:
        config = configs[wl_name]
        for rate in rates:
            for variant in variants:
                done += 1
                sys.stdout.write(f"\r[{done}/{total}] {wl_name}/{variant} @ {rate} req/s")
                sys.stdout.flush()
                try:
                    row = run_one(variant, rate, config, args, model_type, pmt)
                    results.append(row)
                except Exception as e:
                    print(f"\n  ERROR: {e}")

    print()
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 100)
    for wl_name in workloads:
        wl_df = df[df.workload == wl_name]
        if wl_df.empty:
            continue
        print(f"\n--- {wl_name} ---")
        summary = wl_df.pivot_table(
            index='rate', columns='variant',
            values=['goodput', 'slo_attainment', 'p99_ttft', 'p99_tpot'],
            aggfunc='first',
        )
        print(summary.to_string(float_format='%.1f'))
    print("=" * 100)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
