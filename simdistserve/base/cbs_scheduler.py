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
        migration_interval: float = 500.0,
        theta_ceil: float = 0.3,
        theta_floor: float = 0.4,
        theta_dispatch: float = 0.85,
        max_migrations_per_scan: int = 2,
        migration_cooldown: float = 10000.0,
        slo_tpot: float = 100.0,
        slo_ttft: float = 2000.0,
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
        # Track per-request cooldown (req_id -> earliest next migration time)
        self._cooldown_until: dict = {}
        # Statistics
        self.n_colocated = 0
        self.n_disaggregated = 0
        self.n_migrations = 0
        self.n_role_switches = 0

    def schedule_new_req(self, req: 'Request'):
        if req.counter >= 0:
            return self.schedule_decode(req)

        c_disagg = self._compute_disagg_cost(req)

        best_score = float('-inf')
        best_worker = None
        for d_worker in self._decode_heads:
            c_coloc = self._compute_coloc_cost(req, d_worker)
            risk = self._compute_risk(req, d_worker)
            score = c_disagg - c_coloc - self.mu * risk
            if score > best_score:
                best_score = score
                best_worker = d_worker

        if best_score > 0 and best_worker is not None:
            self._route_coloc(req, best_worker)
        else:
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
        """C_coloc = T_queue_d + T_prefill_0*(1+a_p) + T_decode_0*(1+a_d)*o_remain + ext + dispatch"""
        decode_bs = len(d_worker.decode_queue) + d_worker._decode_ips

        # Queue wait at this decode worker
        t_queue_d = self._estimate_decode_queue_wait(d_worker)

        # Interference coefficients
        alpha_p = self.interference_model.get_alpha_p(
            decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
        )
        alpha_d = self.interference_model.get_alpha_d(
            decode_bs=decode_bs, prefill_len=req.prefill_lens, model_type=d_worker.model_type,
        )

        # Prefill time with interference
        t_prefill_0 = get_prefill_time(
            num_tokens=req.prefill_lens, bs=1, decode_bs=0,
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Prefill,
            prefill_len_list=[req.prefill_lens],
            engine_type=d_worker.engine_type,
        )

        # Decode step time with interference
        o_remain = req.output_lens
        t_decode_0 = get_decode_time(
            num_requests=max(decode_bs, 1),
            pp=self.cluster.PP_decode,
            model_type=d_worker.model_type, TP=d_worker.TP_Decode,
            token_generated_list=[req.prefill_lens + 1],
            engine_type=d_worker.engine_type,
        )

        # Externality: prefill blocks decode for existing requests
        delta_ext = t_prefill_0 * decode_bs if decode_bs > 0 else 0

        # Dispatch contention
        delta_dispatch = self.kappa_dispatch * (
            t_prefill_0 + t_decode_0 * o_remain
        )

        return (
            t_queue_d
            + t_prefill_0 * (1 + alpha_p)
            + t_decode_0 * (1 + alpha_d) * o_remain
            + self.lambda_ext * delta_ext
            + delta_dispatch
        )

    def _compute_risk(self, req: 'Request', d_worker: 'CBSWorker') -> float:
        """SLO violation risk: higher when decode worker is heavily loaded."""
        decode_bs = len(d_worker.decode_queue) + d_worker._decode_ips
        capacity = d_worker.decode_max_batch_size
        if capacity <= 0:
            return 0
        load_ratio = decode_bs / capacity
        return load_ratio * req.output_lens * 0.01

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
            self._do_mitigation()
            self._do_consolidation()
            if self.enable_role_adaptation:
                self._do_role_adaptation()

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

    def _violation_prob(self, worker: 'CBSWorker') -> float:
        """Approximate SLO violation probability based on current TPOT vs SLO."""
        tpot = self._estimate_tpot(worker)
        if self.slo_tpot <= 0:
            return 0
        ratio = tpot / self.slo_tpot
        # Simple sigmoid-like mapping: ratio > 1 means likely violation
        if ratio <= 0.5:
            return 0.0
        elif ratio >= 1.5:
            return 1.0
        else:
            return (ratio - 0.5) / 1.0  # Linear from 0 to 1 in [0.5, 1.5]

    def _do_mitigation(self):
        """Mitigation: migrate requests from overloaded decode workers to lighter ones."""
        now = self.env.now
        n_migrated = 0

        for src_worker in self._decode_heads:
            if n_migrated >= self.max_migrations_per_scan:
                break
            p_viol = self._violation_prob(src_worker)
            if p_viol <= self.theta_ceil:
                continue

            # Find candidates to migrate (from queue tail, not in cooldown)
            candidates = [
                r for r in src_worker.decode_queue
                if self._cooldown_until.get(r.req_id, 0) <= now
            ]
            if not candidates:
                continue

            # Sort by remaining output (migrate shortest first = least KV to transfer)
            candidates.sort(key=lambda r: r.output_lens - max(0, r.counter))

            for req in candidates:
                if n_migrated >= self.max_migrations_per_scan:
                    break

                # Find best destination
                best_dst = None
                best_tpot_after = float('inf')
                for dst_worker in self._decode_heads:
                    if dst_worker is src_worker:
                        continue
                    dst_bs = len(dst_worker.decode_queue) + dst_worker._decode_ips
                    # Check safety: TPOT after adding this request
                    tpot_after = self._estimate_tpot_with_extra(dst_worker, 1)
                    if tpot_after <= self.theta_dispatch * self.slo_tpot:
                        if tpot_after < best_tpot_after:
                            best_tpot_after = tpot_after
                            best_dst = dst_worker

                if best_dst is not None:
                    # Execute migration
                    src_worker.decode_queue.remove(req)
                    best_dst.decode_queue.append(req)
                    best_dst.wakeup()
                    self._cooldown_until[req.req_id] = now + self.migration_cooldown
                    n_migrated += 1
                    self.n_migrations += 1

    def _do_consolidation(self):
        """Consolidation: merge requests from lightly-loaded decode workers."""
        now = self.env.now
        n_migrated_total = self.n_migrations  # Track within this scan

        # Find low-load workers
        low_load_workers = []
        for w in self._decode_heads:
            if len(w.decode_queue) + w._decode_ips == 0:
                continue
            tpot = self._estimate_tpot(w)
            if tpot < self.theta_floor * self.slo_tpot:
                low_load_workers.append(w)

        # Sort by load ascending (merge smallest first)
        low_load_workers.sort(key=lambda w: len(w.decode_queue) + w._decode_ips)

        for src_worker in low_load_workers:
            if len(self._decode_heads) <= 1:
                break
            # Try to move all requests out
            movable = [
                r for r in list(src_worker.decode_queue)
                if self._cooldown_until.get(r.req_id, 0) <= now
            ]
            if not movable:
                continue

            all_placed = True
            placements = []  # (req, dst_worker)
            for req in movable:
                best_dst = None
                best_tpot = float('inf')
                for dst_worker in self._decode_heads:
                    if dst_worker is src_worker:
                        continue
                    tpot_after = self._estimate_tpot_with_extra(
                        dst_worker, 1 + sum(1 for _, d in placements if d is dst_worker)
                    )
                    if tpot_after <= self.theta_dispatch * self.slo_tpot and tpot_after < best_tpot:
                        best_tpot = tpot_after
                        best_dst = dst_worker
                if best_dst is None:
                    all_placed = False
                    break
                placements.append((req, best_dst))

            if all_placed and placements:
                for req, dst_worker in placements:
                    src_worker.decode_queue.remove(req)
                    dst_worker.decode_queue.append(req)
                    dst_worker.wakeup()
                    self._cooldown_until[req.req_id] = now + self.migration_cooldown
                    self.n_migrations += 1

    def _estimate_tpot_with_extra(self, worker: 'CBSWorker', extra_reqs: int) -> float:
        """Estimate TPOT if extra_reqs more requests were added to this worker."""
        decode_bs = len(worker.decode_queue) + worker._decode_ips + extra_reqs
        if decode_bs == 0:
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

    def _do_role_adaptation(self):
        """Dynamic role switching: idle decode -> prefill when prefill queue is long."""
        # Decode -> Prefill: when prefill queue pressure is high and decode worker is empty
        avg_prefill_queue = sum(
            len(w.prefill_queue) + w._prefill_ips for w in self._prefill_heads
        ) / max(len(self._prefill_heads), 1)

        # Estimate average prefill queue wait
        avg_queue_wait = 0
        for w in self._prefill_heads:
            avg_queue_wait += self._estimate_prefill_queue_wait(w)
        avg_queue_wait /= max(len(self._prefill_heads), 1)

        role_thresh = 0.5 * self.slo_ttft

        if avg_queue_wait > role_thresh:
            # Find an empty decode worker to convert
            for d_worker in list(self._decode_heads):
                if len(d_worker.decode_queue) + d_worker._decode_ips > 0:
                    continue
                if len(self._decode_heads) <= 1:
                    break  # Must keep at least 1 decode worker
                # Switch role: decode -> prefill
                d_worker.role = 'prefill'
                d_worker.should_request_stay = False
                d_worker.global_scheduler = self
                self._decode_heads.remove(d_worker)
                self._decode_queues.remove(d_worker.decode_queue)
                self._prefill_heads.append(d_worker)
                self._prefill_queues.append(d_worker.prefill_queue)
                self.n_role_switches += 1
                break  # One switch per scan

        # Prefill -> Decode: when decode pressure is high and prefill worker is idle
        avg_tpot = 0
        n_active = 0
        for w in self._decode_heads:
            if len(w.decode_queue) + w._decode_ips > 0:
                avg_tpot += self._estimate_tpot(w)
                n_active += 1
        if n_active > 0:
            avg_tpot /= n_active

        if avg_tpot > self.theta_dispatch * self.slo_tpot:
            # Find an idle prefill worker that was originally a decode worker
            for p_worker in list(self._prefill_heads):
                if p_worker.role != 'prefill':
                    # This was a converted decode worker
                    if len(p_worker.prefill_queue) + p_worker._prefill_ips > 0:
                        continue
                    if len(self._prefill_heads) <= 1:
                        break
                    # Switch back: prefill -> decode
                    p_worker.role = 'decode'
                    p_worker.should_request_stay = True
                    p_worker.global_scheduler = None
                    self._prefill_heads.remove(p_worker)
                    self._prefill_queues.remove(p_worker.prefill_queue)
                    self._decode_heads.append(p_worker)
                    self._decode_queues.append(p_worker.decode_queue)
                    self.n_role_switches += 1
                    break
