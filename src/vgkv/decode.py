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
    budget: int = 256  # absolute max KV cache length per layer; ignored if generation_budget is set
    generation_budget: int | None = None  # if set, eviction budget = prompt_len + generation_budget
    sink_size: int = 4
    protect_prompt: bool = True  # if True, sink_size is ignored and the entire prompt is protected
    eos_token_id: int | None = None


def generate_with_policy(model, tokenizer, prompt: str, policy, config: DecodeConfig):
    """Runs prefill + greedy decode, applying `policy` for eviction each step
    once the cache exceeds the effective budget. Returns (generated_text, RunMetrics).

    With protect_prompt=True (default), the sink covers the entire prompt, so
    eviction only ever competes over which *generated* tokens to keep -- the
    problem statement itself (e.g. the numbers in a GSM8K word problem) is
    never evicted. Without this, a short fixed sink_size (StreamingLLM-style)
    leaves most of the prompt evictable, which for short-but-fact-dense
    prompts tends to destroy the model's ability to recall the problem at all
    regardless of policy quality -- a confound, not a real eviction-policy
    comparison.

    config.generation_budget, when set, expresses the budget as "how many
    generated tokens can survive eviction" rather than an absolute cache
    size -- this matters because GSM8K prompt lengths vary per example, so a
    fixed absolute config.budget gives different amounts of generation
    headroom to different examples (longer prompt -> less room to reason).
    generation_budget keeps that headroom constant across examples, which is
    what an accuracy-vs-budget sweep should be comparing. Effective absolute
    budget = prompt_len + generation_budget in that case.
    """
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    effective_sink_size = prompt_len if config.protect_prompt else config.sink_size
    effective_budget = (
        prompt_len + config.generation_budget if config.generation_budget is not None else config.budget
    )
    if config.protect_prompt and effective_budget < prompt_len:
        print(
            f"warning: effective_budget={effective_budget} < prompt_len={prompt_len} with "
            "protect_prompt=True -- no generated tokens can survive eviction, results won't "
            "reflect policy quality"
        )

    mkv = ManagedKVCache(num_layers=num_layers, num_kv_heads=num_kv_heads, sink_size=effective_sink_size)
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
    mkv.evict_to_budget(policy, effective_budget)

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
        mkv.evict_to_budget(policy, effective_budget)

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        metrics.peak_kv_cache_bytes = max(metrics.peak_kv_cache_bytes, kv_cache_bytes(mkv.cache, num_layers))

    metrics.decode_time_s = decode_elapsed
    metrics.generated_tokens = len(generated_ids)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, metrics
