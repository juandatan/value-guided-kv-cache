"""Metrics collected per generation run: throughput, memory, cache utilization.
Accuracy is scored separately in vgkv.eval (needs the gold example, not just
generation-time signals).
"""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class StepTiming:
    step: int
    seq_len: int
    kv_cache_len: int
    wall_time: float


@dataclass
class RunMetrics:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    peak_kv_cache_bytes: int = 0
    step_timings: list[StepTiming] = field(default_factory=list)

    @property
    def decode_tokens_per_sec(self) -> float:
        if self.decode_time_s <= 0 or self.generated_tokens <= 1:
            return 0.0
        # first decode step produces the token right after prefill; throughput
        # is measured over steps 2..N to exclude prefill's own tail effects.
        return (self.generated_tokens - 1) / self.decode_time_s

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_time_s": self.prefill_time_s,
            "decode_time_s": self.decode_time_s,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "peak_kv_cache_bytes": self.peak_kv_cache_bytes,
            "peak_kv_cache_mb": self.peak_kv_cache_bytes / (1024**2),
        }


class Timer:
    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self._start


def kv_cache_bytes(cache, num_layers: int) -> int:
    """Sum of key+value tensor byte sizes across all layers of a DynamicCache."""
    total = 0
    for layer_idx in range(min(num_layers, len(cache.key_cache))):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        total += k.element_size() * k.nelement()
        total += v.element_size() * v.nelement()
    return total
