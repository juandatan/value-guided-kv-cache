"""Hand-rolled greedy decode loop with pluggable KV cache eviction.

Not using model.generate() because eviction requires slicing past_key_values
between steps -- generate()'s internals don't expose a hook for that. Requires
the model to have been loaded with attn_implementation="eager" so
output_attentions=True actually returns attention weights (SDPA/flash-attn
backends return None here).
"""

from dataclasses import dataclass

import torch

from vgkv.cache.managed_cache import ManagedKVCache
from vgkv.metrics.instrumentation import RunMetrics, Timer, kv_cache_bytes
from vgkv.value_models.entropy_salience import EntropySaliencePolicy


@dataclass
class DecodeConfig:
    max_new_tokens: int = 512
    budget: int = 256  # max KV cache length per layer; set >= prompt+max_new_tokens for "no eviction"
    sink_size: int = 4
    eos_token_id: int | None = None


def generate_with_policy(model, tokenizer, prompt: str, policy, config: DecodeConfig):
    """Runs prefill + greedy decode, applying `policy` for eviction each step
    once the cache exceeds config.budget. Returns (generated_text, RunMetrics).
    """
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    mkv = ManagedKVCache(num_layers=num_layers, num_kv_heads=num_kv_heads, sink_size=config.sink_size)
    metrics = RunMetrics(prompt_tokens=prompt_len)
    is_entropy_policy = isinstance(policy, EntropySaliencePolicy)

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id

    generated_ids: list[int] = []

    with Timer() as t_prefill:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                past_key_values=mkv.cache,
                use_cache=True,
                output_attentions=True,
            )
    metrics.prefill_time_s = t_prefill.elapsed

    if is_entropy_policy:
        for layer_idx, attn in enumerate(out.attentions):
            policy.accumulate_step(layer_idx, attn, num_kv_heads)
    mkv.record_step(out.attentions, step=0)
    mkv.evict_to_budget(policy, config.budget)

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_token.item())
    metrics.peak_kv_cache_bytes = max(metrics.peak_kv_cache_bytes, kv_cache_bytes(mkv.cache, num_layers))

    decode_elapsed = 0.0
    for step in range(1, config.max_new_tokens):
        if eos_token_id is not None and generated_ids[-1] == eos_token_id:
            break

        with Timer() as t_step:
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    past_key_values=mkv.cache,
                    use_cache=True,
                    output_attentions=True,
                )
        decode_elapsed += t_step.elapsed

        if is_entropy_policy:
            for layer_idx, attn in enumerate(out.attentions):
                policy.accumulate_step(layer_idx, attn, num_kv_heads)
        mkv.record_step(out.attentions, step=step)
        mkv.evict_to_budget(policy, config.budget)

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        metrics.peak_kv_cache_bytes = max(metrics.peak_kv_cache_bytes, kv_cache_bytes(mkv.cache, num_layers))

    metrics.decode_time_s = decode_elapsed
    metrics.generated_tokens = len(generated_ids)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, metrics
