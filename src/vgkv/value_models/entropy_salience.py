"""Forward-entropy salience: score a cached token by how much attending to it
reduces the model's predictive entropy, approximated cheaply via attention mass
weighted by (1 - normalized entropy of the attention distribution at the step
it was produced/queried).

This is an approximation, not the true counterfactual (removing token i and
re-running the forward pass to measure entropy delta) -- that's O(seq_len) forward
passes per eviction decision and not tractable per-step. Instead we use:

    salience(token_i) = sum_over_steps_t[ attn_mass(t -> i) * confidence(t) ]

where confidence(t) = 1 - H(attn_dist_t) / log(kv_len_t), i.e. steps whose attention
distribution was sharply peaked (low entropy, confident about what mattered) count
more than steps that attended diffusely (high entropy, attention wasn't
discriminating). This still needs the same attentions tensor H2O uses, so it's a
drop-in alternative scoring function on top of the same bookkeeping.

Approach log entry: see PROJECT.md section 6. First value-model iteration --
compare against H2OPolicy (raw attention-sum) to see whether entropy-weighting
the attention mass changes which tokens get evicted / whether it helps accuracy
at tight budgets.
"""

import torch

from vgkv.cache.managed_cache import CacheState
from vgkv.value_models.base import EvictionPolicy


class EntropySaliencePolicy(EvictionPolicy):
    def __init__(self):
        self._layer_weighted_accum: dict[int, torch.Tensor] = {}

    def accumulate_step(self, layer_idx: int, attn: torch.Tensor, num_kv_heads: int) -> None:
        """Call once per layer per forward pass, before/alongside
        ManagedKVCache.record_step, with the same raw attentions tensor
        (attn: [batch, num_q_heads, q_len, kv_len], batch=1).
        """
        batch, num_q_heads, q_len, kv_len = attn.shape
        group_size = num_q_heads // num_kv_heads
        attn_by_kv_head = attn.view(batch, num_kv_heads, group_size, q_len, kv_len)

        avg_attn = attn_by_kv_head.mean(dim=2)  # [batch, num_kv_heads, q_len, kv_len]
        eps = 1e-12
        entropy = -(avg_attn * (avg_attn + eps).log()).sum(dim=-1)  # [batch, num_kv_heads, q_len]
        max_entropy = torch.log(torch.tensor(float(kv_len), device=attn.device))
        confidence = 1.0 - (entropy / max_entropy.clamp(min=eps))  # [batch, num_kv_heads, q_len]

        mass = avg_attn * confidence.unsqueeze(-1)  # [batch, num_kv_heads, q_len, kv_len]
        weighted = mass.sum(dim=(0, 2))  # [num_kv_heads, kv_len]

        prev = self._layer_weighted_accum.get(layer_idx)
        if prev is None or prev.shape[-1] < kv_len:
            pad_len = kv_len - (prev.shape[-1] if prev is not None else 0)
            pad = torch.zeros(num_kv_heads, pad_len, device=attn.device, dtype=weighted.dtype)
            prev = pad if prev is None else torch.cat([prev, pad], dim=-1)
        prev = prev + weighted
        self._layer_weighted_accum[layer_idx] = prev

    def score(self, state: CacheState, layer_idx: int) -> torch.Tensor:
        weighted = self._layer_weighted_accum.get(layer_idx)
        seq_len = state.seq_len(layer_idx)
        if weighted is None:
            device = state.layers[layer_idx].attn_accum.device
            return torch.zeros(seq_len, device=device)
        return weighted.sum(dim=0)[:seq_len]

    def sync_after_eviction(self, layer_idx: int, keep_idx: torch.Tensor) -> None:
        """Re-index the weighted attention accumulator to match the cache after
        ManagedKVCache evicts a layer (it only knows how to re-index its own
        LayerState fields, not this policy's private state).
        """
        weighted = self._layer_weighted_accum.get(layer_idx)
        if weighted is not None:
            self._layer_weighted_accum[layer_idx] = weighted.index_select(-1, keep_idx.to(weighted.device))
