"""
CBS Scheduler: per-request dynamic routing between disaggregated and colocated paths.

For each arriving prefill request, computes:
  CBS(r, d) = C_disagg(r) - C_coloc(r, d) - mu * risk(r, d)

If max CBS > 0 across decode workers, route to colocation (best decode worker).
Otherwise, route to disaggregated path (least-loaded prefill worker).

Supports three variants:
  - CBS-NoMig: only per-request routing, no migration
  - CBS-NoRole: routing + bidirectional migration, no role adaptation
  - CBS-Full: routing + migration + role adaptation
"""
import math
from typing import List, TYPE_CHECKING

from simdistserve.base.scheduler import Scheduler
from simdistserve.estimators.interference_model import InterferenceModel
from simdistserve.estimators.time_estimator import get_prefill_time, get_decode_time

if TYPE_CHECKING:
    from simdistserve.base.cbs_worker import CBSWorker
    from simdistserve.base.request import Request


class CBSScheduler(Scheduler):
    def __init__(
        self,
        env,
        prefill_heads: 'List[CBSWorker]',
        decode_heads: 'List[CBSWorker]',
        cluster=None,
        kv_transfer_latency: float = 5.0,
        control_latency: float = 2.0,
        mu: float = 2.0,
        lambda_ext: float = 1.0,
        kappa_dispatch: float = 0.1,
        interference_model: InterferenceModel = None,
        # Migration parameters
        enable_migration: bool = False,
        enable_role_adaptation: bool = False,
        migration_interval: float = 1000.0,  # 1s scan period (chap4 sec 4.5.3)
        theta_ceil: float = 0.3,
        theta_floor: float = 0.4,
        theta_dispatch: float = 0.85,
        max_migrations_per_scan: int = 2,
        migration_cooldown: float = 10000.0,
        slo_tpot: float = 100.0,
        slo_ttft: float = 2000.0,
        # Memory / compute budget constraints
        rho_min: float = 0.05,
        max_gpu_mem_tokens: int = 66144,  # from profile data max_num_tokens
        # P_viol normal approximation
        sigma_alpha: float = 0.02,
    ):
        super().__init__(env, prefill_heads, decode_heads)
        self.cluster = cluster
        self.kv_transfer_latency = kv_transfer_latency
        self.control_latency = control_latency
        self.mu = mu
        self.lambda_ext = lambda_ext
        self.kappa_dispatch = kappa_dispatch
        self.interference_model = interference_model or InterferenceModel()
        # Migration config
        self.enable_migration = enable_migration
        self.enable_role_adaptation = enable_role_adaptation
        self.migration_interval = migration_interval
        self.theta_ceil = theta_ceil
        self.theta_floor = theta_floor
        self.theta_dispatch = theta_dispatch
        self.max_migrations_per_scan = max_migrations_per_scan
        self.migration_cooldown = migration_cooldown
        self.slo_tpot = slo_tpot
        self.slo_ttft = slo_ttft
        self.rho_min = rho_min
        self.max_gpu_mem_tokens = max_gpu_mem_tokens
        self.sigma_alpha = sigma_alpha
        # Track per-request cooldown (req_id -> earliest next migration time)
        self._cooldown_until: dict = {}
        # Track initial decode count for role adaptation guard
        self._initial_n_decode = len(decode_heads)
        # Track which workers were originally decode (for role switch-back)
        self._converted_workers: set = set()  # wid of workers converted from decode to prefill
        # Statistics
        self.n_colocated = 0
        self.n_disaggregated = 0
        self.n_migrations = 0
        self.n_role_switches = 0
        # Structured event log for detailed analysis
        self.event_log: list = []  # list of dicts
        self.verbose = False  # set True for terminal output

    def _log(self, event_type: str, **kwargs):
        entry = {'t': self.env.now, 'event': event_type, **kwargs}
        self.event_log.append(entry)
        if self.verbose:
            parts = [f"[{self.env.now:>8.0f}ms] {event_type:>12s}"]
            for k, v in kwargs.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.2f}")
                else:
                    parts.append(f"{k}={v}")
            print(" | ".join(parts))

    def schedule_new_req(self, req: 'Request'):
        if req.counter >= 0:
            return self.schedule_decode(req)

        c_disagg = self._compute_disagg_cost(req)

        best_score = float('-inf')
        best_worker = None
        for d_worker in self._decode_heads:
            # I_mem: memory feasibility check (chap4 eq 4.15)
            tokens_used = sum(r.current_context_len for r in d_worker.decode_queue)
            tokens_after = tokens_used + req.prefill_lens + req.output_lens
            if tokens_after > self.max_gpu_mem_tokens * (1 - self.rho_min):
                continue

            # B_token_max: compute budget check (chap4 eq 4.9)
            iter_tokens = sum(1 for _ in d_worker.decode_queue) + req.prefill_lens
            decode_bs = len(d_worker.decode_queue) + d_worker._decode_ips
            if decode_bs > 0 and self.slo_tpot > 0:
                t_iter_0 = self._estimate_tpot(d_worker)
                marginal = t_iter_0 / max(decode_bs, 1)
                b_token_max = self.slo_tpot / max(marginal, 0.001) if marginal > 0 else float('inf')
                if iter_tokens > b_token_max:
                    continue

            c_coloc = self._compute_coloc_cost(req, d_worker)
            risk = self._compute_risk(req, d_worker)
            score = c_disagg - c_coloc - self.mu * risk
            if score > best_score:
                best_score = score
                best_worker = d_worker

        if best_score > 0 and best_worker is not None:
            self._log('CBS_COLOC', req_id=req.req_id, score=best_score,
                      target_wid=best_worker.wid, prefill_len=req.prefill_lens,
                      output_len=req.output_lens, c_disagg=c_disagg)
            self._route_coloc(req, best_worker)
        else:
            self._log('CBS_DISAGG', req_id=req.req_id, score=best_score,
                      prefill_len=req.prefill_lens, output_len=req.output_lens,
                      c_disagg=c_disagg)
            self._route_disagg(req)

    def _route_coloc(self, req: 'Request', d_worker: 'CBSWorker'):
        req.is_colocated = True
        self.n_colocated += 1
        d_worker.prefill_queue.append(req)
        req.wait_prefill(wid=d_worker.wid)
        d_worker.wakeup()

    def _route_disagg(self, req: 'Request'):
        req.is_colocated = False
        self.n_disaggregated += 1
        self.schedule_prefill(req)

    # ---- Cost computation ----

    def _compute_disagg_cost(self, req: 'Request') -> float:
        """C_disagg = T_queue_p + T_prefill_0 + T_kv + T_ctrl + T_queue_d + T_decode_0 * o_remain"""
        # Prefill queue wait
        p_worker, _ = self._find_best_worker_and_queue(self._prefill_heads, self._prefill_queues)
        t_queue_p = self._estimate_prefill_queue_wait(p_worker)

        # Baseline prefill time (no interference)
        t_prefill_0 = get_prefill_time(
            num_tokens=req.prefill_lens, bs=1, decode_bs=0,
            pp=self.cluster.PP_prefill,
            model_type=p_worker.model_type, TP=p_worker.TP_Prefill,
            prefill_len_list=[req.prefill_lens],
            engine_type=p_worker.engine_type,
        )

        # KV transfer + control overhead
        t_kv = self.kv_transfer_latency
        t_ctrl = self.control_latency

        # Decode queue wait
        d_worker, _ = self._find_best_worker_and_queue(self._decode_heads, self._decode_queues)
        t_queue_d = self._estimate_decode_queue_wait(d_worker)

        # Baseline decode step time * remaining output tokens
        o_remain = req.output_lens
        t_decode_0 = get_decode_time(
            num_requests=max(len(d_worker.decode_queue), 1),
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Decode,
            token_generated_list=[req.prefill_lens + 1],
            engine_type=d_worker.engine_type,
        )

        return t_queue_p + t_prefill_0 + t_kv + t_ctrl + t_queue_d + t_decode_0 * o_remain

    def _compute_coloc_cost(self, req: 'Request', d_worker: 'CBSWorker') -> float:
        """
        C_coloc = T_queue_d + ΔT_prefill + λ·Δ_ext + Δ_dispatch
        where ΔT_prefill = α_p · T_prefill_0 (plus the baseline T_prefill_0 for symmetry with C_disagg)
        """
        decode_bs = len(d_worker.decode_queue) + d_worker._decode_ips

        # Queue wait at this decode worker
        t_queue_d = self._estimate_decode_queue_wait(d_worker)

        # Interference coefficients
        alpha_p = self.interference_model.get_alpha_p(
            decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
        )

        # Prefill time (baseline)
        t_prefill_0 = get_prefill_time(
            num_tokens=req.prefill_lens, bs=1, decode_bs=0,
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Prefill,
            prefill_len_list=[req.prefill_lens],
            engine_type=d_worker.engine_type,
        )

        # Decode step time (baseline)
        o_remain = req.output_lens
        t_decode_0 = get_decode_time(
            num_requests=max(decode_bs, 1),
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Decode,
            token_generated_list=[req.prefill_lens + 1],
            engine_type=d_worker.engine_type,
        )

        # Coloc prefill duration τ_j = T_prefill_0 * (1 + α_p)
        tau_j = t_prefill_0 * (1 + alpha_p)

        # Δ_ext: per-request externality with ω_u weighting (chap4 eq 4.11-4.12)
        delta_ext = 0.0
        if d_worker.decode_queue:
            avg_output = sum(r.output_lens for r in d_worker.decode_queue) / len(d_worker.decode_queue)
            for u in d_worker.decode_queue:
                alpha_d_u = self.interference_model.get_alpha_d(
                    decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
                )
                # ω_u = η1 + η2 * (o_remain_u / avg_o) + η3 * (TPOT_u / SLO_TPOT)
                o_remain_u = max(u.output_lens - max(0, u.counter), 1)
                tpot_u_ratio = self._estimate_single_tpot(u, d_worker) / self.slo_tpot if self.slo_tpot > 0 else 0
                omega_u = 0.5 + 0.3 * (o_remain_u / max(avg_output, 1)) + 0.2 * tpot_u_ratio
                delta_ext += omega_u * tau_j * alpha_d_u

        # Δ_dispatch: simplified (no g_launch in simulator)
        alpha_d = self.interference_model.get_alpha_d(
            decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
        )
        delta_dispatch = self.kappa_dispatch * tau_j

        return (
            t_queue_d
            + t_prefill_0 * (1 + alpha_p)
            + t_decode_0 * (1 + alpha_d) * o_remain
            + self.lambda_ext * delta_ext
            + delta_dispatch
        )

    def _compute_risk(self, req: 'Request', d_worker: 'CBSWorker') -> float:
        """
        Δ_risk = γ1 · max(0, TTFT_coloc - SLO_TTFT) + γ2 · Σ max(0, TPOT_u_coloc - SLO_TPOT)
        (chap4 eq 4.16)
        """
        decode_bs = len(d_worker.decode_queue) + d_worker._decode_ips
        alpha_p = self.interference_model.get_alpha_p(
            decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
        )

        # Predicted TTFT under colocation
        t_queue_d = self._estimate_decode_queue_wait(d_worker)
        t_prefill_0 = get_prefill_time(
            num_tokens=req.prefill_lens, bs=1, decode_bs=0,
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Prefill,
            prefill_len_list=[req.prefill_lens],
            engine_type=d_worker.engine_type,
        )
        ttft_coloc = t_queue_d + t_prefill_0 * (1 + alpha_p)
        gamma1 = 1.0
        risk_ttft = gamma1 * max(0.0, ttft_coloc - self.slo_ttft)

        # Predicted TPOT for each existing decode request after colocation
        gamma2 = 1.0
        risk_tpot = 0.0
        for u in d_worker.decode_queue:
            tpot_u = self._estimate_single_tpot_with_coloc(u, d_worker, req)
            risk_tpot += max(0.0, tpot_u - self.slo_tpot)
        risk_tpot *= gamma2

        return risk_ttft + risk_tpot

    def _estimate_single_tpot(self, u: 'Request', worker: 'CBSWorker') -> float:
        """Estimate TPOT for a single request u on worker (current state)."""
        decode_bs = len(worker.decode_queue) + worker._decode_ips
        if decode_bs == 0:
            return 0
        return get_decode_time(
            num_requests=decode_bs,
            pp=self.cluster.PP_decode,
            model_type=worker.model_type, TP=worker.TP_Decode,
            token_generated_list=[u.current_context_len + 1],
            engine_type=worker.engine_type,
        )

    def _estimate_single_tpot_with_coloc(self, u: 'Request', worker: 'CBSWorker', new_req: 'Request') -> float:
        """Estimate TPOT for request u if new_req is colocated on the same worker."""
        decode_bs = len(worker.decode_queue) + worker._decode_ips
        alpha_d = self.interference_model.get_alpha_d(
            decode_bs=decode_bs, prefill_len=new_req.prefill_lens, model_type=worker.model_type,
        )
        t_step_0 = get_decode_time(
            num_requests=max(decode_bs, 1),
            pp=self.cluster.PP_decode,
            model_type=worker.model_type, TP=worker.TP_Decode,
            token_generated_list=[u.current_context_len + 1],
            engine_type=worker.engine_type,
        )
        return t_step_0 * (1 + alpha_d)

    # ---- Queue wait estimation ----

    def _estimate_prefill_queue_wait(self, worker: 'CBSWorker') -> float:
        queue_len = len(worker.prefill_queue) + worker._prefill_ips
        if queue_len == 0:
            return 0
        avg_tokens = sum(r.remain_prefill_lens for r in worker.prefill_queue) / max(len(worker.prefill_queue), 1)
        return queue_len * get_prefill_time(
            num_tokens=max(int(avg_tokens), 1), bs=1, decode_bs=0,
            pp=self.cluster.PP_prefill,
            model_type=worker.model_type, TP=worker.TP_Prefill,
            prefill_len_list=[max(int(avg_tokens), 1)],
            engine_type=worker.engine_type,
        )

    def _estimate_decode_queue_wait(self, worker: 'CBSWorker') -> float:
        queue_len = len(worker.decode_queue) + worker._decode_ips
        if queue_len == 0:
            return 0
        # One decode step for the current batch
        return get_decode_time(
            num_requests=queue_len,
            pp=self.cluster.PP_decode,
            model_type=worker.model_type, TP=worker.TP_Decode,
            token_generated_list=[512] * queue_len,
            engine_type=worker.engine_type,
        )

    # ---- Migration (CBS-NoRole / CBS-Full) ----

    def migration_process(self):
        """SimPy process: periodic migration scan. Stops when all queues are empty."""
        while True:
            yield self.env.timeout(self.migration_interval)
            # Check if there's any work left in the system
            any_work = False
            for w in self._prefill_heads + self._decode_heads:
                if w.prefill_queue or w.decode_queue or w._prefill_ips > 0 or w._decode_ips > 0:
                    any_work = True
                    break
            if not any_work:
                return  # All done, stop migration process
            if not self.enable_migration:
                continue
            # Shared migration budget per scan (chap4: mitigation + consolidation share R_max)
            self._scan_migrations = 0
            # Log node snapshot before migration scan
            self._log_node_snapshot()
            self._do_mitigation()
            self._do_consolidation()
            # Switch-back check runs every scan (independent of consolidation)
            if self.enable_role_adaptation:
                self._try_recover_to_decode()

    def _estimate_tpot(self, worker: 'CBSWorker') -> float:
        """Estimate per-token output time for requests on this worker."""
        decode_bs = len(worker.decode_queue) + worker._decode_ips
        if decode_bs == 0:
            return 0
        avg_ctx = sum(r.current_context_len for r in worker.decode_queue) / max(len(worker.decode_queue), 1)
        t_step = get_decode_time(
            num_requests=decode_bs,
            pp=self.cluster.PP_decode,
            model_type=worker.model_type, TP=worker.TP_Decode,
            token_generated_list=[int(avg_ctx)] * decode_bs,
            engine_type=worker.engine_type,
        )
        return t_step

    def _log_node_snapshot(self):
        """Log state of all decode nodes at each migration scan."""
        nodes = []
        for w in self._decode_heads:
            bs = len(w.decode_queue) + w._decode_ips
            tpot = self._estimate_tpot(w) if bs > 0 else 0
            p_viol = self._violation_prob(w)
            nodes.append({
                'wid': w.wid, 'role': w.role, 'decode_bs': bs,
                'prefill_qs': len(w.prefill_queue),
                'tpot': round(tpot, 1), 'p_viol': round(p_viol, 3),
            })
        self._log('NODE_SNAP', n_prefill=len(self._prefill_heads),
                  n_decode=len(self._decode_heads), nodes=nodes)

    def _violation_prob(self, worker: 'CBSWorker') -> float:
        """
        P_viol ≈ 1 - Φ((SLO_TPOT / T_decode_step_0 - 1 - α_d) / σ_α)
        (chap4 eq 4.21)
        """
        tpot = self._estimate_tpot(worker)
        if self.slo_tpot <= 0 or tpot == 0:
            return 0
        # Approximate: use tpot as T_step_0 * (1 + α_d), so α_d ≈ tpot/T_step_0 - 1
        # For the normal CDF, compute z = (SLO/tpot_baseline - 1 - alpha_hat) / sigma
        # Simplified: z = (SLO - tpot) / (sigma * tpot)
        z = (self.slo_tpot - tpot) / max(self.sigma_alpha * tpot, 0.001)
        return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    def _do_mitigation(self):
        """
        Mitigation (chap4 Algorithm 2, Phase 1):
        Scan all decode nodes, find requests with P_viol > θ_ceil,
        sort by P_viol desc (tie-break by KV size asc),
        migrate to node with max G_mig > 0.
        """
        now = self.env.now

        # Collect candidates across all overloaded workers
        all_candidates = []
        for src_worker in self._decode_heads:
            p_viol = self._violation_prob(src_worker)
            if p_viol <= self.theta_ceil:
                continue
            for r in src_worker.decode_queue:
                if self._cooldown_until.get(r.req_id, 0) <= now:
                    kv_size = r.current_context_len
                    all_candidates.append((r, src_worker, p_viol, kv_size))

        if not all_candidates:
            return

        # Sort: P_viol desc; if diff < 0.05, by KV size asc
        all_candidates.sort(key=lambda x: (-x[2], x[3]))

        for req, src_worker, p_viol, kv_size in all_candidates:
            if self._scan_migrations >= self.max_migrations_per_scan:
                break
            if req not in src_worker.decode_queue:
                continue

            # Find target with max G_mig > 0
            best_dst = None
            best_g_mig = 0.0
            for dst_worker in self._decode_heads:
                if dst_worker is src_worker:
                    continue
                tpot_after = self._estimate_tpot_with_extra(dst_worker, 1)
                if tpot_after > self.theta_dispatch * self.slo_tpot:
                    continue

                # G_mig (chap4 eq 4.25)
                o_remain = max(req.output_lens - max(0, req.counter), 1)
                tpot_src_before = self._estimate_tpot(src_worker)
                tpot_src_after = self._estimate_tpot_with_extra(src_worker, -1)
                delta_src = max(tpot_src_before - tpot_src_after, 0)
                kv_delay = self.kv_transfer_latency * (req.current_context_len / 512.0)
                tpot_dst_before = self._estimate_tpot(dst_worker)
                delta_admit = max(tpot_after - tpot_dst_before, 0)
                g_mig = o_remain * delta_src - kv_delay - delta_admit * o_remain

                if g_mig > best_g_mig:
                    best_g_mig = g_mig
                    best_dst = dst_worker

            if best_dst is not None:
                kv_delay = self.kv_transfer_latency * (req.current_context_len / 512.0)
                self._log('MIGRATE_MIT', req_id=req.req_id,
                          src_wid=src_worker.wid, dst_wid=best_dst.wid,
                          g_mig=best_g_mig, kv_delay=kv_delay, p_viol=p_viol,
                          ctx_len=req.current_context_len)
                self.env.process(self._async_migrate(req, src_worker, best_dst, kv_delay))
                self._cooldown_until[req.req_id] = now + self.migration_cooldown
                self._scan_migrations += 1
                self.n_migrations += 1

    def _async_migrate(self, req, src_worker, dst_worker, delay):
        """SimPy process: async KV transfer, request stays on src during transfer."""
        yield self.env.timeout(delay)
        if req in src_worker.decode_queue:
            src_worker.decode_queue.remove(req)
            if not req._terminated:
                dst_worker.decode_queue.append(req)
                dst_worker.wakeup()

    def _do_consolidation(self):
        """
        Consolidation (chap4 Algorithm 2, Phase 2):
        Find low-load nodes (all requests TPOT < θ_floor * SLO),
        try to bin-pack their requests to other safe nodes (argmax TPOT_after),
        trigger role adaptation if node is fully cleared.
        """
        now = self.env.now

        # Find low-load workers (chap4: all requests TPOT < θ_floor * SLO and has active requests)
        # Additional stability check: only consider nodes with few requests (truly underutilized)
        low_load_workers = []
        avg_decode_bs = sum(
            len(w.decode_queue) + w._decode_ips for w in self._decode_heads
        ) / max(len(self._decode_heads), 1)

        for w in self._decode_heads:
            w_bs = len(w.decode_queue) + w._decode_ips
            if w_bs == 0:
                continue
            # Must be significantly below average load to be considered low-load
            if w_bs >= avg_decode_bs * 0.5 and avg_decode_bs > 2:
                continue
            all_below = True
            for u in w.decode_queue:
                tpot_u = self._estimate_single_tpot(u, w)
                if tpot_u >= self.theta_floor * self.slo_tpot:
                    all_below = False
                    break
            if all_below:
                low_load_workers.append(w)

        # Sort by |W_d| ascending (drain smallest first)
        low_load_workers.sort(key=lambda w: len(w.decode_queue) + w._decode_ips)

        for src_worker in low_load_workers:
            if len(self._decode_heads) <= 1:
                break
            if self._scan_migrations >= self.max_migrations_per_scan:
                break

            movable = [
                r for r in list(src_worker.decode_queue)
                if self._cooldown_until.get(r.req_id, 0) <= now
            ]
            if not movable:
                continue

            all_placed = True
            placements = []
            for req in movable:
                if self._scan_migrations + len(placements) >= self.max_migrations_per_scan:
                    all_placed = False
                    break

                # Find safe node with highest TPOT_after (bin-packing: fill fullest first)
                best_dst = None
                best_tpot = -1.0
                for dst_worker in self._decode_heads:
                    if dst_worker is src_worker:
                        continue
                    n_already = sum(1 for _, d in placements if d is dst_worker)
                    tpot_after = self._estimate_tpot_with_extra(dst_worker, 1 + n_already)
                    # Safety: TPOT_after ≤ θ_dispatch * SLO and memory check
                    if tpot_after > self.theta_dispatch * self.slo_tpot:
                        continue
                    if tpot_after > best_tpot:
                        best_tpot = tpot_after
                        best_dst = dst_worker

                if best_dst is None:
                    all_placed = False
                    break
                placements.append((req, best_dst))

            if all_placed and placements:
                self._log('MIGRATE_CON', src_wid=src_worker.wid,
                          n_reqs=len(placements),
                          dst_wids=[d.wid for _, d in placements])
                for req, dst_worker in placements:
                    kv_delay = self.kv_transfer_latency * (req.current_context_len / 512.0)
                    self.env.process(self._async_migrate(req, src_worker, dst_worker, kv_delay))
                    self._cooldown_until[req.req_id] = now + self.migration_cooldown
                    self._scan_migrations += 1
                    self.n_migrations += 1

                # Role adaptation after consolidation clears node (chap4 Algorithm 2 line 476)
                if self.enable_role_adaptation:
                    max_delay = max(
                        self.kv_transfer_latency * (r.current_context_len / 512.0) for r, _ in placements
                    )
                    self.env.process(self._delayed_role_check(src_worker, max_delay + 1))

    def _estimate_tpot_with_extra(self, worker: 'CBSWorker', extra_reqs: int) -> float:
        """Estimate TPOT if extra_reqs more (or fewer, if negative) requests on this worker."""
        decode_bs = len(worker.decode_queue) + worker._decode_ips + extra_reqs
        if decode_bs <= 0:
            return 0
        avg_ctx = 512  # Rough estimate
        if worker.decode_queue:
            avg_ctx = sum(r.current_context_len for r in worker.decode_queue) / len(worker.decode_queue)
        return get_decode_time(
            num_requests=decode_bs,
            pp=self.cluster.PP_decode,
            model_type=worker.model_type, TP=worker.TP_Decode,
            token_generated_list=[int(avg_ctx)] * decode_bs,
            engine_type=worker.engine_type,
        )

    # ---- Role Adaptation (CBS-Full) ----
    # Triggered only after consolidation clears a node (chap4 Algorithm 2 line 476)

    def _try_convert_to_prefill(self, empty_worker: 'CBSWorker'):
        """Convert a cleared decode worker to prefill if prefill queue pressure is high."""
        if empty_worker not in self._decode_heads:
            return  # Already converted by another async path
        n_decode_now = len(self._decode_heads)
        # Guard: |P|≥1 and |D|≥1
        if n_decode_now <= 1:
            return

        avg_queue_wait = 0
        for w in self._prefill_heads:
            avg_queue_wait += self._estimate_prefill_queue_wait(w)
        avg_queue_wait /= max(len(self._prefill_heads), 1)

        role_thresh = 0.5 * self.slo_ttft
        if avg_queue_wait <= role_thresh:
            return

        # Convert: decode -> prefill
        self._log('ROLE_D2P', wid=empty_worker.wid,
                  prefill_queue_wait=avg_queue_wait,
                  n_prefill_before=len(self._prefill_heads),
                  n_decode_before=n_decode_now)
        empty_worker.role = 'prefill'
        empty_worker.should_request_stay = False
        empty_worker.global_scheduler = self
        if empty_worker in self._decode_heads:
            self._decode_heads.remove(empty_worker)
        if empty_worker.decode_queue in self._decode_queues:
            self._decode_queues.remove(empty_worker.decode_queue)
        if empty_worker not in self._prefill_heads:
            self._prefill_heads.append(empty_worker)
        if empty_worker.prefill_queue not in self._prefill_queues:
            self._prefill_queues.append(empty_worker.prefill_queue)
        self._converted_workers.add(empty_worker.wid)
        self.n_role_switches += 1

    def _delayed_role_check(self, worker, delay):
        """SimPy process: check role conversion after migrations complete."""
        yield self.env.timeout(delay)
        if worker in self._decode_heads and len(worker.decode_queue) == 0 and worker._decode_ips == 0:
            self._try_convert_to_prefill(worker)

    def _try_recover_to_decode(self):
        """Switch back: converted prefill worker -> decode when decode pressure is high."""
        if not self._converted_workers:
            return

        avg_tpot = 0
        n_active = 0
        for w in self._decode_heads:
            if len(w.decode_queue) + w._decode_ips > 0:
                avg_tpot += self._estimate_tpot(w)
                n_active += 1
        if n_active > 0:
            avg_tpot /= n_active

        if avg_tpot <= self.theta_dispatch * self.slo_tpot:
            return

        for p_worker in list(self._prefill_heads):
            if p_worker.wid not in self._converted_workers:
                continue
            if p_worker not in self._prefill_heads:
                continue  # Already recovered by another path
            if len(p_worker.prefill_queue) + p_worker._prefill_ips > 0:
                continue
            if len(self._prefill_heads) <= 1:
                break
            # Switch back: prefill -> decode
            self._log('ROLE_P2D', wid=p_worker.wid, avg_tpot=avg_tpot,
                      n_prefill_before=len(self._prefill_heads),
                      n_decode_before=len(self._decode_heads))
            p_worker.role = 'decode'
            p_worker.should_request_stay = True
            p_worker.global_scheduler = None
            if p_worker in self._prefill_heads:
                self._prefill_heads.remove(p_worker)
            if p_worker.prefill_queue in self._prefill_queues:
                self._prefill_queues.remove(p_worker.prefill_queue)
            if p_worker not in self._decode_heads:
                self._decode_heads.append(p_worker)
            if p_worker.decode_queue not in self._decode_queues:
                self._decode_queues.append(p_worker.decode_queue)
            self._converted_workers.discard(p_worker.wid)
            self.n_role_switches += 1
            break
