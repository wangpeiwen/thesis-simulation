#!/usr/bin/env python3
"""
CBS experiment runner: compare Disagg-Static, CBS-NoMig, CBS-NoRole, CBS-Full
across multiple arrival rates.

Usage:
    python -m simdistserve.benchmarks.simulate_cbs [options]

Example:
    python -m simdistserve.benchmarks.simulate_cbs \
        --model facebook/opt-13b --N 500 \
        --rates 2,4,6,8,10,12,14,16 \
        --n-prefill 2 --n-decode 6 \
        --output results/cbs_comparison.csv
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

from simdistserve.base.organize_data import (
    organize_request_df, organize_request_event_df, calculate_per_request_latency,
)
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    convert_pd_pair_to_request, convert_absolutearrival_to_interarrival,
    get_gamma_interarrival,
)
from simdistserve.clusters.cbs import CBSCluster
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens


def parse_args():
    p = argparse.ArgumentParser(description='CBS comparison experiment')
    p.add_argument('--model', default='facebook/opt-13b')
    p.add_argument('--N', type=int, default=500, help='Requests per run')
    p.add_argument('--rates', default='2,4,6,8,10,12', help='Comma-separated arrival rates')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-prefill', type=int, default=2)
    p.add_argument('--n-decode', type=int, default=6)
    p.add_argument('--tp-prefill', type=int, default=1)
    p.add_argument('--pp-prefill', type=int, default=1)
    p.add_argument('--tp-decode', type=int, default=1)
    p.add_argument('--pp-decode', type=int, default=1)
    p.add_argument('--slo-ttft', type=float, default=2000.0, help='TTFT SLO (ms)')
    p.add_argument('--slo-tpot', type=float, default=100.0, help='TPOT SLO (ms)')
    p.add_argument('--cbs-mu', type=float, default=0.5)
    p.add_argument('--cbs-lambda-ext', type=float, default=0.1)
    p.add_argument('--cbs-kappa', type=float, default=0.1)
    p.add_argument('--cbs-kv-latency', type=float, default=5.0)
    p.add_argument('--interference-table', default=None)
    p.add_argument('--prompt-range', default='128,2048', help='min,max prompt length')
    p.add_argument('--output-range', default='32,512', help='min,max output length')
    p.add_argument('--output', default=None, help='Output CSV path')
    p.add_argument('--variants', default='all',
                   help='Comma-separated variants: disagg,vllm,cbs-nomig,cbs-norole,cbs-full,all')
    return p.parse_args()


# PLACEHOLDER_GENERATE_WORKLOAD


def generate_workload(N, rate, seed, prompt_range, output_range):
    """Generate synthetic workload with Poisson arrivals."""
    random.seed(seed)
    np.random.seed(seed)
    pmin, pmax = prompt_range
    omin, omax = output_range
    pairs = [(random.randint(pmin, pmax), random.randint(omin, omax)) for _ in range(N)]
    requests = convert_pd_pair_to_request(pairs)
    arrival = get_gamma_interarrival(N, rate, cv=1.0, seed=seed)
    return requests, arrival


def make_cluster(env, variant, args, wc, pmt):
    """Create cluster for the given variant."""
    if variant == 'disagg':
        return DisaggCluster(
            env=env, N_prefill_instance=args.n_prefill, N_decode_instance=args.n_decode,
            PP_prefill=args.pp_prefill, PP_decode=args.pp_decode, worker_configs=wc,
        )
    elif variant == 'vllm':
        vllm_wc = WorkerConfig(
            model_type=wc['model_type'], TP=args.tp_prefill,
            TP_Prefill=args.tp_prefill, TP_Decode=args.tp_prefill,
            prefill_max_batch_size=10**7, decode_max_batch_size=10**7,
            prefill_max_tokens=pmt, decode_max_tokens=pmt,
            enable_chunked_prefill=False, engine_type='vllm',
        )
        return VLLMCluster(env=env, PP=args.pp_prefill, worker_configs=vllm_wc)
    elif variant == 'cbs-nomig':
        return CBSCluster(
            env=env, N_prefill_instance=args.n_prefill, N_decode_instance=args.n_decode,
            PP_prefill=args.pp_prefill, PP_decode=args.pp_decode, worker_configs=wc,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda_ext, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.cbs_kv_latency,
            interference_table_path=args.interference_table,
        )
    elif variant == 'cbs-norole':
        return CBSCluster(
            env=env, N_prefill_instance=args.n_prefill, N_decode_instance=args.n_decode,
            PP_prefill=args.pp_prefill, PP_decode=args.pp_decode, worker_configs=wc,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda_ext, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.cbs_kv_latency,
            interference_table_path=args.interference_table,
            enable_migration=True, slo_tpot=args.slo_tpot,
        )
    elif variant == 'cbs-full':
        return CBSCluster(
            env=env, N_prefill_instance=args.n_prefill, N_decode_instance=args.n_decode,
            PP_prefill=args.pp_prefill, PP_decode=args.pp_decode, worker_configs=wc,
            mu=args.cbs_mu, lambda_ext=args.cbs_lambda_ext, kappa_dispatch=args.cbs_kappa,
            kv_transfer_latency=args.cbs_kv_latency,
            interference_table_path=args.interference_table,
            enable_migration=True, enable_role_adaptation=True,
            slo_tpot=args.slo_tpot, slo_ttft=args.slo_ttft,
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_one(variant, rate, args, wc, pmt):
    """Run a single experiment and return metrics."""
    prompt_range = tuple(int(x) for x in args.prompt_range.split(','))
    output_range = tuple(int(x) for x in args.output_range.split(','))
    requests, arrival = generate_workload(args.N, rate, args.seed, prompt_range, output_range)

    env = simpy.Environment()
    cluster = make_cluster(env, variant, args, wc, pmt)
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    env.run()

    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    latency_df = calculate_per_request_latency(request_event_df, request_df.output_lens)

    ttft = latency_df['first_token_latency']
    tpot = latency_df['tpot']
    slo_ok = ((ttft <= args.slo_ttft) & (tpot <= args.slo_tpot)).sum()
    goodput = slo_ok / (requests[-1].log[-1][0] / 1000) if requests[-1].log else 0

    return {
        'variant': variant,
        'rate': rate,
        'n_requests': args.N,
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


def main():
    args = parse_args()
    model_type = ModelTypes.model_str_to_object(args.model)
    pmt = get_max_num_tokens(model_type, args.tp_prefill, args.pp_prefill)

    wc = WorkerConfig(
        model_type=model_type, TP=args.tp_prefill,
        TP_Prefill=args.tp_prefill, TP_Decode=args.tp_decode,
        prefill_max_batch_size=10**7, decode_max_batch_size=10**7,
        prefill_max_tokens=pmt, decode_max_tokens=pmt,
        enable_chunked_prefill=False, engine_type='distserve',
    )

    rates = [float(r) for r in args.rates.split(',')]

    if args.variants == 'all':
        variants = ['disagg', 'cbs-nomig', 'cbs-norole', 'cbs-full']
    else:
        variants = [v.strip() for v in args.variants.split(',')]

    results = []
    total = len(variants) * len(rates)
    done = 0

    for rate in rates:
        for variant in variants:
            done += 1
            sys.stdout.write(f"\r[{done}/{total}] {variant} @ {rate} req/s ...")
            sys.stdout.flush()
            row = run_one(variant, rate, args, wc, pmt)
            results.append(row)

    print()
    df = pd.DataFrame(results)

    # Print summary table
    print("\n=== Results ===")
    pivot_cols = ['variant', 'rate', 'n_colocated', 'n_migrations', 'n_role_switches',
                  'p50_ttft', 'p90_ttft', 'p99_ttft', 'p90_tpot', 'slo_attainment', 'goodput']
    print(df[pivot_cols].to_string(index=False, float_format='%.1f'))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
