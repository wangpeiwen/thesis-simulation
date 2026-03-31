"""
CBS Cluster: hybrid cluster supporting both disaggregated and colocated execution paths.

Topology is similar to DisaggCluster (separate prefill and decode instances),
but CBSScheduler can route new requests directly to decode workers' prefill_queue
for colocation, bypassing the prefill instances when beneficial.
"""
from functools import reduce
from itertools import chain
from typing import List, Optional

from simdistserve.base.cbs_scheduler import CBSScheduler
from simdistserve.base.cbs_worker import CBSWorker
from simdistserve.base.worker import WorkerConfig
from simdistserve.estimators.interference_model import InterferenceModel
from simdistserve.utils import set_next_worker


class CBSCluster:
    def __init__(
        self,
        env,
        N_prefill_instance: int = 1,
        N_decode_instance: int = 1,
        PP_prefill: int = 1,
        PP_decode: int = 1,
        worker_configs: 'Optional[WorkerConfig]' = None,
        # CBS parameters
        kv_transfer_latency: float = 5.0,
        control_latency: float = 2.0,
        mu: float = 2.0,
        lambda_ext: float = 1.0,
        kappa_dispatch: float = 0.1,
        interference_table_path: str = None,
        # Migration parameters
        enable_migration: bool = False,
        enable_role_adaptation: bool = False,
        migration_interval: float = 500.0,
        theta_ceil: float = 0.3,
        theta_floor: float = 0.4,
        slo_tpot: float = 100.0,
        slo_ttft: float = 2000.0,
    ):
        self.env = env
        self.PP_prefill = PP_prefill
        self.PP_decode = PP_decode

        interference_model = InterferenceModel(table_path=interference_table_path)

        worker_kwargs = dict(
            global_scheduler=None,
            **(worker_configs or {}),
        )

        worker_id = 0
        prefill_instances: List[List[CBSWorker]] = []
        decode_instances: List[List[CBSWorker]] = []

        # Create prefill instances
        for inst_id in range(N_prefill_instance):
            instance = []
            for i in range(PP_prefill):
                w = CBSWorker(
                    env, worker_id, cluster=self, pipe_rank=i,
                    role='prefill', interference_model=interference_model,
                    **worker_kwargs,
                )
                instance.append(w)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            prefill_instances.append(instance)

        # Create decode instances
        for inst_id in range(N_decode_instance):
            instance = []
            for i in range(PP_decode):
                w = CBSWorker(
                    env, worker_id, cluster=self, pipe_rank=i,
                    role='decode', interference_model=interference_model,
                    **worker_kwargs,
                )
                instance.append(w)
                worker_id += 1
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            decode_instances.append(instance)

        # CBS Scheduler
        scheduler = CBSScheduler(
            env,
            prefill_heads=[inst[0] for inst in prefill_instances],
            decode_heads=[inst[0] for inst in decode_instances],
            cluster=self,
            kv_transfer_latency=kv_transfer_latency,
            control_latency=control_latency,
            mu=mu,
            lambda_ext=lambda_ext,
            kappa_dispatch=kappa_dispatch,
            interference_model=interference_model,
            enable_migration=enable_migration,
            enable_role_adaptation=enable_role_adaptation,
            migration_interval=migration_interval,
            theta_ceil=theta_ceil,
            theta_floor=theta_floor,
            slo_tpot=slo_tpot,
            slo_ttft=slo_ttft,
        )

        # Wire prefill pipeline tails to global scheduler (for disagg path)
        for inst in prefill_instances:
            inst[-1].global_scheduler = scheduler

        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.scheduler: CBSScheduler = scheduler

    def get_all_workers(self):
        return list(chain(
            chain(*self.prefill_instances),
            chain(*self.decode_instances),
        ))

    def run(self):
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
        # Start migration process if enabled
        if self.scheduler.enable_migration or self.scheduler.enable_role_adaptation:
            self.env.process(self.scheduler.migration_process())
        return self
