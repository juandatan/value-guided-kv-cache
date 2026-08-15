"""Common interface for KV cache eviction/value policies."""

from abc import ABC, abstractmethod

import torch

from vgkv.cache.managed_cache import CacheState


class EvictionPolicy(ABC):
    @abstractmethod
    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        """Return a [seq_len] tensor scoring each cached token at this layer.
        Higher score = more valuable = keep. ManagedKVCache.evict_to_budget
        keeps the top-`budget` scores outside the sink region.
        """
        raise NotImplementedError

    def sync_after_eviction(self, layer_idx: int, keep_idx: torch.Tensor) -> None:
        """Called by ManagedKVCache right after a layer's cache is sliced to
        `keep_idx`. Override for policies that keep their own per-token state
        (e.g. EntropySaliencePolicy's weighted attention accumulator) so it
        stays aligned with the cache. No-op by default.
        """
