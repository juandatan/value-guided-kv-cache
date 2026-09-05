"""Custom KV cache wrapper supporting arbitrary per-layer token eviction.

Wraps transformers' DynamicCache and manipulates its internal key_cache /
value_cache tensor lists directly, since eviction (removing an arbitrary
slot, not just truncating from the front) isn't exposed by the public Cache
API in the transformers versions we target (see requirements.txt pin,
4.43-4.46 -- DynamicCache.key_cache/value_cache are plain per-layer tensor
lists in this range; 4.47+ began refactoring Cache internals).

Rotary position embeddings are already baked into cached K vectors at the
position they were computed, so removing a slot from the middle of the cache
does not require re-computing anything for the slots that remain -- this is
the same assumption StreamingLLM / H2O rely on.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import DynamicCache

if TYPE_CHECKING:
    from vgkv.value_models.base import EvictionPolicy


@dataclass
class LayerState:
    """Per-layer bookkeeping used by eviction policies."""

    attn_accum: torch.Tensor  # [num_kv_heads, seq_len] cumulative attention mass received
    last_used_step: torch.Tensor  # [seq_len] step index a token was last attended to
    position_ids: torch.Tensor  # [seq_len] original sequence position, for inspection/debug


@dataclass
class CacheState:
    cache: DynamicCache
    layers: list[LayerState] = field(default_factory=list)
    sink_size: int = 4
    step: int = 0

    def seq_len(self, layer_idx: int) -> int:
        return self.cache.key_cache[layer_idx].shape[-2]


class ManagedKVCache:
    """Owns a DynamicCache plus the bookkeeping eviction policies need.

    Usage:
        mkv = ManagedKVCache(num_layers, num_kv_heads, sink_size=4)
        # prefill
        out = model(**inputs, past_key_values=mkv.cache, use_cache=True,
                     output_attentions=True)
        mkv.record_step(out.attentions, step=0)
        mkv.evict_to_budget(policy, budget=budget)
        # decode loop: repeat model(...) with mkv.cache, mkv.record_step,
        # mkv.evict_to_budget each step
    """

    def __init__(self, num_layers: int, num_kv_heads: int, sink_size: int = 4):
        self.state = CacheState(cache=DynamicCache(), sink_size=sink_size)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

    @property
    def cache(self) -> DynamicCache:
        return self.state.cache

    def _ensure_layer_state(self, layer_idx: int, device: torch.device) -> None:
        while len(self.state.layers) <= layer_idx:
            self.state.layers.append(
                LayerState(
                    attn_accum=torch.zeros(self.num_kv_heads, 0, device=device),
                    last_used_step=torch.zeros(0, dtype=torch.long, device=device),
                    position_ids=torch.zeros(0, dtype=torch.long, device=device),
                )
            )

    def record_step(
        self,
        attentions: tuple[torch.Tensor, ...],
        step: int,
        cache_position: torch.Tensor | None = None,
    ) -> None:
        """Update per-token attention bookkeeping after a forward pass.

        attentions: tuple of length num_layers, each [batch, num_q_heads, q_len, kv_len].
        Assumes batch size 1 (single-sequence decode loop).

        cache_position contains the absolute position(s) of the newly appended
        token(s). It must be supplied by eviction-aware decode loops because
        the physical cache length stops increasing after eviction and no
        longer identifies the true sequence position.
        """
        self.state.step = step
        for layer_idx, attn in enumerate(attentions):
            device = attn.device
            self._ensure_layer_state(layer_idx, device)
            layer_state = self.state.layers[layer_idx]

            batch, num_q_heads, q_len, kv_len = attn.shape
            group_size = num_q_heads // self.num_kv_heads
            # Query heads are grouped: qh -> kv head qh // group_size (repeat_kv layout).
            attn_by_kv_head = attn.view(batch, self.num_kv_heads, group_size, q_len, kv_len)
            # mass received by each kv-cache slot, summed over query heads in the
            # group and over the query positions in this forward pass.
            mass = attn_by_kv_head.sum(dim=(0, 2, 3))  # [num_kv_heads, kv_len]

            prev_len = layer_state.attn_accum.shape[-1]
            if kv_len > prev_len:
                pad = torch.zeros(
                    self.num_kv_heads, kv_len - prev_len, device=device, dtype=layer_state.attn_accum.dtype
                )
                layer_state.attn_accum = torch.cat([layer_state.attn_accum, pad], dim=-1)
                pad_pos = torch.zeros(kv_len - prev_len, dtype=torch.long, device=device)
                layer_state.last_used_step = torch.cat([layer_state.last_used_step, pad_pos], dim=-1)
                new_count = kv_len - prev_len
                if cache_position is None:
                    new_positions = torch.arange(prev_len, kv_len, device=device, dtype=torch.long)
                else:
                    new_positions = cache_position.to(device=device, dtype=torch.long).reshape(-1)
                    if new_positions.numel() != new_count:
                        raise ValueError(
                            f"cache_position has {new_positions.numel()} entries but "
                            f"{new_count} new cache slots were appended"
                        )
                layer_state.position_ids = torch.cat([layer_state.position_ids, new_positions], dim=-1)

            layer_state.attn_accum += mass

            attended = mass.sum(dim=0) > 0  # [kv_len]
            layer_state.last_used_step = torch.where(
                attended, torch.full_like(layer_state.last_used_step, step), layer_state.last_used_step
            )

    def evict_to_budget(self, policy: "EvictionPolicy", budget: int) -> None:
        """Score each layer's cached tokens via `policy.score`, then drop the
        lowest-scoring tokens (outside the sink region) until seq_len <= budget.
        `policy.sync_after_eviction` is called after each layer's eviction so
        stateful policies (e.g. EntropySaliencePolicy) can keep their own
        accumulators aligned with the cache.
        """
        for layer_idx in range(len(self.state.layers)):
            seq_len = self.state.seq_len(layer_idx)
            if seq_len <= budget:
                continue

            sink = self.state.sink_size
            scores = policy.score(self.state, layer_idx)  # [seq_len]
            assert scores.shape[0] == seq_len

            evictable_scores = scores[sink:]
            n_keep_evictable = budget - sink
            if n_keep_evictable <= 0:
                keep_idx_evictable = torch.tensor([], dtype=torch.long, device=scores.device)
            else:
                topk = torch.topk(evictable_scores, k=n_keep_evictable)
                keep_idx_evictable = topk.indices.sort().values + sink

            keep_idx = torch.cat(
                [torch.arange(sink, device=scores.device, dtype=torch.long), keep_idx_evictable]
            )

            self._apply_keep_indices(layer_idx, keep_idx)
            sync = getattr(policy, "sync_after_eviction", None)
            if sync is not None:
                sync(layer_idx, keep_idx)

    def _apply_keep_indices(self, layer_idx: int, keep_idx: torch.Tensor) -> None:
        k = self.state.cache.key_cache[layer_idx]
        v = self.state.cache.value_cache[layer_idx]
        self.state.cache.key_cache[layer_idx] = k.index_select(-2, keep_idx)
        self.state.cache.value_cache[layer_idx] = v.index_select(-2, keep_idx)

        layer_state = self.state.layers[layer_idx]
        layer_state.attn_accum = layer_state.attn_accum.index_select(-1, keep_idx)
        layer_state.last_used_step = layer_state.last_used_step.index_select(-1, keep_idx)
        layer_state.position_ids = layer_state.position_ids.index_select(-1, keep_idx)

    def current_lengths(self) -> list[int]:
        return [self.state.seq_len(i) for i in range(len(self.state.layers))]
