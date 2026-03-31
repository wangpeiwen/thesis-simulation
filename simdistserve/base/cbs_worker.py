"""
CBS Worker: extends Worker with interference simulation for colocated execution.

When a Decode-role worker executes prefill (colocation path), the prefill time
is inflated by alpha_p. When decode runs while prefill is queued, decode time
is inflated by alpha_d.
"""
from typing import List, TYPE_CHECKING, Literal

from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.estimators.interference_model import InterferenceModel
from simdistserve.estimators.time_estimator import get_prefill_time, get_decode_time

if TYPE_CHECKING:
    from simdistserve.base.request import Request


class CBSWorker(Worker):
    def __init__(
        self, env, wid,
        cluster=None,
        role: str = 'decode',
        interference_model: InterferenceModel = None,
        **kwargs,
    ):
        super().__init__(env, wid, cluster=cluster, **kwargs)
        self.role = role  # 'prefill', 'decode', or 'hybrid'
        self.interference_model = interference_model or InterferenceModel()

    def do_prefill(self):
        prefill_items: 'List[Request]' = self._enter_prefill()
        if self.enable_chunked_prefill:
            remaining_tok = self.prefill_max_tokens - sum(
                x.current_prefill_lens for x in prefill_items
            )
            decode_reqs = self._enter_decodes(remaining_tok)
        else:
            decode_reqs = []

        num_tokens = sum(x.current_prefill_lens for x in prefill_items) + len(decode_reqs)

        self._log_event(
            "do_prefill",
            num_tokens=num_tokens,
            prefill_bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )

        # Use PP_prefill for prefill-role workers, PP_decode for decode-role workers
        pp = self.cluster.PP_prefill if self.role == 'prefill' else self.cluster.PP_decode

        delay = get_prefill_time(
            num_tokens,
            bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            pp=pp,
            model_type=self.model_type,
            TP=self.TP_Prefill,
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            engine_type=self.engine_type,
        )

        # Interference: inflate prefill time on decode-role workers
        if self.role in ('decode', 'hybrid') and self.decode_queue:
            decode_bs = len(self.decode_queue)
            avg_prefill_len = num_tokens / max(len(prefill_items), 1)
            alpha_p = self.interference_model.get_alpha_p(
                decode_bs=decode_bs,
                prefill_len=int(avg_prefill_len),
                model_type=self.model_type,
            )
            delay *= (1 + alpha_p)

        num_tokens_ctx = sum(x.current_context_len for x in (prefill_items + decode_reqs))
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens_ctx)

        self._prefill_ips = len(prefill_items)
        yield self.env.timeout(delay)
        self._prefill_ips = 0

        # Colocation path: prefill exits locally on decode-role workers
        if self.role in ('decode', 'hybrid'):
            self._exit_prefill_coloc(prefill_items)
        else:
            self._exit_prefill(prefill_items)
        self._exit_decode(decode_reqs)

    def _exit_prefill_coloc(self, prefill_items: List['Request']):
        """Colocation exit: completed prefill stays on this decode worker."""
        for item in prefill_items:
            next_wid = self.next_worker.wid if self.next_worker else None
            item.finish_prefill(
                is_finished_one_round=self.is_last_in_pipeline,
                wid=self.wid, next_wid=next_wid,
            )
            if not self.is_last_in_pipeline or item.remain_prefill_lens > 0:
                self.forward_prefill(item)
                continue
            if item.should_finish():
                continue
            # Stay local: append to this worker's decode queue
            self.decode_queue.append(item)

    def do_decode(self):
        decode_reqs = self._enter_decodes(self.decode_max_tokens)
        batch_size = len(decode_reqs)
        self._log_event(
            "do_decode", num_tokens=batch_size, decode_bs=batch_size,
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )
        _token_generated_list = [x.current_context_len + 1 for x in decode_reqs]
        delay = get_decode_time(
            batch_size,
            pp=self.cluster.PP_decode,
            model_type=self.model_type,
            TP=self.TP_Decode,
            token_generated_list=_token_generated_list,
            engine_type=self.engine_type,
        )

        # Interference: inflate decode time when prefill is queued
        if self.role in ('decode', 'hybrid') and self.prefill_queue:
            avg_prefill_len = sum(
                r.remain_prefill_lens for r in self.prefill_queue
            ) / max(len(self.prefill_queue), 1)
            alpha_d = self.interference_model.get_alpha_d(
                decode_bs=batch_size,
                prefill_len=int(avg_prefill_len),
                model_type=self.model_type,
            )
            delay *= (1 + alpha_d)

        num_tokens = sum(x.current_context_len for x in decode_reqs)
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens)
        yield self.env.timeout(delay)
        self._exit_decode(decode_reqs)
