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
        Based on empirical observations: alpha_p ranges 0-0.06.
        Prefill is compute-bound, interference from decode is moderate.
        """
        # ~2% at decode_bs=10, ~5% at decode_bs=50
        return min(0.001 * decode_bs + 0.000005 * prefill_len, 0.08)

    @staticmethod
    def _formula_alpha_d(decode_bs: int, prefill_len: int) -> float:
        """
        Parametric model for decode interference.
        Based on empirical observations: alpha_d ranges 0-0.09.
        Decode is memory-bound, interference from prefill is more significant.
        Key insight: larger decode batch DILUTES interference (shared across more steps).
        """
        # Base interference from prefill length, diluted by decode batch size
        base = 0.00003 * prefill_len  # ~3% at 1024, ~6% at 2048
        dilution = 1.0 / max(decode_bs, 1)  # Larger batch = less per-request impact
        return min(base * dilution + 0.0005 * decode_bs, 0.12)

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
