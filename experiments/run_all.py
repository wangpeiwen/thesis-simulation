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


def generate_workload(N, rate, config, seed=42, duration_s=None):
    """
    Generate requests and arrival times from config.
    If duration_s is set, generate requests for that duration instead of fixed N.
    This matches real system behavior: requests arrive over a fixed time window.
    """
    random.seed(seed)
    pmin, pmax = config['prompt_range']
    omin, omax = config['output_range']

    arrival_type = config.get('arrival', 'poisson')

    if duration_s is not None:
        # Generate arrivals until we exceed the time window
        # This is more realistic: we don't know N in advance
        np.random.seed(seed)
        pairs = []
        intervals = [0]  # first request at t=0
        t = 0.0
        while True:
            if arrival_type == 'bursty':
                cycle_pos = t % config.get('burst_period', 30.0)
                cur_rate = rate * config.get('burst_multiplier', 4.0) \
                    if cycle_pos < config.get('burst_duration', 5.0) else rate
            else:
                cur_rate = rate
            gap = np.random.exponential(1.0 / cur_rate)
            t += gap
            if t > duration_s:
                break
            intervals.append(gap * 1000)  # ms
            pairs.append((random.randint(pmin, pmax), random.randint(omin, omax)))
        # First request
        if not pairs:
            pairs.append((random.randint(pmin, pmax), random.randint(omin, omax)))
        requests = convert_pd_pair_to_request(pairs)
        actual_n = len(requests)
        # Trim intervals to match
        intervals = intervals[:actual_n]
        return requests, intervals, actual_n

    # Fixed N mode (legacy)
    pairs = [(random.randint(pmin, pmax), random.randint(omin, omax)) for _ in range(N)]
    requests = convert_pd_pair_to_request(pairs)
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
    return requests, arrival, N


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
    duration_s = args.sim_time_limit  # e.g. 600s = 10 min

    # Generate requests that arrive within the time window
    requests, arrival, actual_n = generate_workload(
        args.N, rate, config, seed=args.seed, duration_s=duration_s,
    )

    env = simpy.Environment()
    cluster = make_cluster(env, variant, args, model_type, pmt)

    # Enable verbose logging if requested
    if hasattr(args, 'verbose') and args.verbose and hasattr(cluster.scheduler, 'verbose'):
        cluster.scheduler.verbose = True

    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)

    # Run for the injection window + a drain phase (2x injection time, capped)
    drain_time = min(duration_s, 120.0)  # up to 2 min drain
    sim_limit = (duration_s + drain_time) * 1000  # ms
    env.run(until=sim_limit)

    # Only count requests that finished
    finished = [r for r in requests if r._terminated]
    n_finished = len(finished)

    empty_row = {
        'workload': config['name'], 'variant': variant, 'rate': rate,
        'n_gpu': args.n_prefill + args.n_decode,
        'n_injected': actual_n, 'n_finished': 0,
        'n_colocated': getattr(cluster.scheduler, 'n_colocated', 0),
        'n_migrations': getattr(cluster.scheduler, 'n_migrations', 0),
        'n_role_switches': getattr(cluster.scheduler, 'n_role_switches', 0),
        'p50_ttft': 0, 'p90_ttft': 0, 'p99_ttft': 0,
        'p50_tpot': 0, 'p90_tpot': 0, 'p99_tpot': 0,
        'slo_attainment': 0, 'goodput': 0,
    }
    if not finished:
        return empty_row

    request_df = organize_request_df(finished)
    request_event_df = organize_request_event_df(finished)
    latency_df = calculate_per_request_latency(request_event_df, request_df.output_lens)

    ttft = latency_df['first_token_latency']
    tpot = latency_df['tpot']
    slo_ok = ((ttft <= args.slo_ttft) & (tpot <= args.slo_tpot)).sum()

    # Goodput = SLO-meeting requests / total simulation time
    goodput = slo_ok / duration_s

    # Save event log if log_dir is set
    event_log = getattr(cluster.scheduler, 'event_log', [])
    if hasattr(args, 'log_dir') and args.log_dir and event_log:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{config['name']}_{variant}_{rate}.json"
        import json as _json
        with open(log_file, 'w') as f:
            _json.dump(event_log, f, indent=1, default=str)

    return {
        'workload': config['name'],
        'variant': variant,
        'rate': rate,
        'n_gpu': args.n_prefill + args.n_decode,
        'n_injected': actual_n,
        'n_finished': n_finished,
        'n_colocated': getattr(cluster.scheduler, 'n_colocated', 0),
        'n_migrations': getattr(cluster.scheduler, 'n_migrations', 0),
        'n_role_switches': getattr(cluster.scheduler, 'n_role_switches', 0),
        'p50_ttft': ttft.quantile(0.5),
        'p90_ttft': ttft.quantile(0.9),
        'p99_ttft': ttft.quantile(0.99),
        'p50_tpot': tpot.quantile(0.5),
        'p90_tpot': tpot.quantile(0.9),
        'p99_tpot': tpot.quantile(0.99),
        'slo_attainment': slo_ok / n_finished * 100 if n_finished > 0 else 0,
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
    p.add_argument('--sim-time-limit', type=float, default=600.0,
                   help='Max simulation wall time in seconds (default 600 = 10 min)')
    p.add_argument('--output', default=None)
    p.add_argument('--verbose', action='store_true', help='Print CBS decisions to terminal')
    p.add_argument('--log-dir', default=None, help='Save event logs to this directory (JSON)')
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
        print(f"\n--- {wl_name} (sim_time={args.sim_time_limit}s) ---")
        for rate in sorted(wl_df.rate.unique()):
            rdf = wl_df[wl_df.rate == rate]
            print(f"\n  Rate: {rate} req/s")
            print(f"  {'Variant':<18s} {'Done':>8s} {'Goodput':>8s} {'SLO%':>6s} {'P99TTFT':>8s} {'P99TPOT':>8s} {'Coloc':>6s} {'Migr':>5s}")
            print(f"  {'-'*68}")
            for v in VARIANTS:
                row = rdf[rdf.variant == v]
                if row.empty:
                    continue
                r = row.iloc[0]
                done_str = f"{int(r.n_finished)}/{int(r.n_injected)}"
                print(f"  {v:<18s} {done_str:>8s} {r.goodput:8.1f} {r.slo_attainment:6.1f} {r.p99_ttft:8.0f} {r.p99_tpot:8.0f} {r.n_colocated:6.0f} {r.n_migrations:5.0f}")
    print("=" * 100)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
