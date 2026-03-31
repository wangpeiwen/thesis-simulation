"""
Interference model for CBS scheduling simulation.

Computes interference coefficients alpha_p (prefill slowdown when colocated)
and alpha_d (decode slowdown when colocated) based on workload characteristics.

Supports two modes:
1. Formula-based: simple parametric model (default)
2. Table-based: load from JSON file (e.g., mlwd-collector output)
"""
import json
from pathlib import Path
from typing import Optional


class InterferenceModel:
    """
    Interference coefficient estimator.

    alpha_p: fractional slowdown of prefill when colocated with decode
    alpha_d: fractional slowdown of decode when colocated with prefill
    """

    def __init__(self, table_path: Optional[str] = None):
        self._table = None
        if table_path:
            self._table = self._load_table(table_path)

    @staticmethod
    def _load_table(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def get_alpha_p(self, decode_bs: int, prefill_len: int, model_type: str = None) -> float:
        """Prefill interference coefficient: how much prefill slows down due to colocated decode."""
        if self._table:
            return self._lookup(model_type, decode_bs, prefill_len, 'alpha_p')
        return self._formula_alpha_p(decode_bs, prefill_len)

    def get_alpha_d(self, decode_bs: int, prefill_len: int, model_type: str = None) -> float:
        """Decode interference coefficient: how much decode slows down due to colocated prefill."""
        if self._table:
            return self._lookup(model_type, decode_bs, prefill_len, 'alpha_d')
        return self._formula_alpha_d(decode_bs, prefill_len)

    @staticmethod
    def _formula_alpha_p(decode_bs: int, prefill_len: int) -> float:
        """
        Parametric model for prefill interference.
        Prefill is compute-bound, so interference from decode is moderate.
        Increases with decode batch size (more memory bandwidth contention).
        """
        # Base: ~5% at decode_bs=10, ~15% at decode_bs=50, ~25% at decode_bs=100
        return min(0.002 * decode_bs + 0.00005 * prefill_len, 0.4)

    @staticmethod
    def _formula_alpha_d(decode_bs: int, prefill_len: int) -> float:
        """
        Parametric model for decode interference.
        Decode is memory-bound, so interference from prefill is significant.
        Increases with prefill length (longer prefill = more compute contention).
        """
        # Base: ~8% at prefill_len=256, ~15% at prefill_len=1024, ~25% at prefill_len=2048
        return min(0.001 * decode_bs + 0.0001 * prefill_len, 0.5)

    def _lookup(self, model_type: str, decode_bs: int, prefill_len: int, key: str) -> float:
        """
        Lookup from loaded table with nearest-neighbor interpolation.

        Expected table format:
        {
            "model_name": {
                "entries": [
                    {"decode_bs": 1, "prefill_len": 128, "alpha_p": 0.03, "alpha_d": 0.05},
                    ...
                ]
            }
        }
        """
        model_data = self._table.get(str(model_type), self._table.get('default', {}))
        entries = model_data.get('entries', [])
        if not entries:
            # Fallback to formula
            if key == 'alpha_p':
                return self._formula_alpha_p(decode_bs, prefill_len)
            return self._formula_alpha_d(decode_bs, prefill_len)

        # Find nearest entry
        best = min(entries, key=lambda e: abs(e['decode_bs'] - decode_bs) + abs(e['prefill_len'] - prefill_len))
        return best.get(key, 0.1)
