"""Eviction policy baselines. See EvictionPolicy in base.py for the interface."""

import torch

from vgkv.cache.managed_cache import CacheState
from vgkv.value_models.base import EvictionPolicy


class NoEviction(EvictionPolicy):
    """Baseline: never evict (score is irrelevant since evict_to_budget is skipped
    when seq_len <= budget -- use this with budget=float('inf') / a very large budget)."""

    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        seq_len = state.seq_len(layer_idx)
        device = state.layers[layer_idx].attn_accum.device
        return torch.ones(seq_len, device=device)


class RandomPolicy(EvictionPolicy):
    """Sanity-floor baseline: uniform random score."""

    def __init__(self, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)

    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        seq_len = state.seq_len(layer_idx)
        device = state.layers[layer_idx].attn_accum.device
        scores = torch.rand(seq_len, generator=self.generator)
        return scores.to(device)


class RecencyPolicy(EvictionPolicy):
    """Sliding-window recency baseline: keep the newest cache positions.

    Cache slots remain in chronological order after eviction because
    ManagedKVCache sorts keep indices before applying them and each decoded
    token is appended at the end. Scoring by slot index therefore implements
    a true "keep the most recent tokens" baseline.

    The previous implementation used ``last_used_step``, but dense softmax
    attention gives nearly every cached token positive attention at every
    step. That made nearly all tokens tie as "recently used" and reduced the
    policy to arbitrary ``topk`` tie-breaking rather than recency.
    """

    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        seq_len = state.seq_len(layer_idx)
        device = state.layers[layer_idx].attn_accum.device
        return torch.arange(seq_len, device=device, dtype=torch.float32)


class H2OPolicy(EvictionPolicy):
    """H2O-style baseline: score = cumulative attention mass received, summed
    over kv heads. https://arxiv.org/abs/2306.14048
    """

    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        return state.layers[layer_idx].attn_accum.sum(dim=0)
