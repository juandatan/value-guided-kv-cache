"""Multi-example, multi-policy evaluation runner.

Sweeps a set of policies across a GSM8K subset and returns per-example results
plus an aggregated summary (accuracy, throughput, peak KV cache memory).
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from vgkv.decode import DecodeConfig, generate_with_policy
from vgkv.eval.gsm8k import build_prompt, is_correct
from vgkv.value_models.base import EvictionPolicy


@dataclass
class PolicySpec:
    name: str
    factory: Callable[[], EvictionPolicy]
    generation_budget: int  # how many generated tokens may survive eviction, independent of prompt_len


def run_policy_eval(
    model,
    tokenizer,
    dataset,
    policy_specs: list[PolicySpec],
    max_new_tokens: int,
    protect_prompt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs every policy in `policy_specs` on every example in `dataset`.

    Each policy gets a *fresh* instance per example via `spec.factory()`.
    Stateful policies (e.g. EntropySaliencePolicy) accumulate per-sequence
    state on the policy object itself, not inside ManagedKVCache -- reusing
    one instance across examples would leak attention mass from one problem
    into the next example's scores.

    Uses generation_budget (not an absolute cache size) so every example gets
    the same amount of generation headroom regardless of its prompt length --
    GSM8K prompts vary in length, and an absolute budget would otherwise give
    longer-prompt examples less room to reason than shorter-prompt ones,
    confounding the accuracy-vs-budget comparison across examples.
    """
    rows = []
    for idx, example in enumerate(dataset):
        prompt = build_prompt(example)
        for spec in policy_specs:
            policy = spec.factory()
            cfg = DecodeConfig(
                max_new_tokens=max_new_tokens,
                generation_budget=spec.generation_budget,
                protect_prompt=protect_prompt,
            )
            text, metrics = generate_with_policy(model, tokenizer, prompt, policy, cfg)
            correct = is_correct(text, example)
            row = {
                "example_idx": idx,
                "policy": spec.name,
                "generation_budget": spec.generation_budget,
                "correct": correct,
                **metrics.as_dict(),
            }
            rows.append(row)
            if verbose:
                print(
                    f"[{idx}] {spec.name}: correct={correct} "
                    f"tok/s={metrics.decode_tokens_per_sec:.1f} "
                    f"peak_kv_mb={metrics.peak_kv_cache_bytes / (1024**2):.1f}"
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["policy", "generation_budget"])
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_tok_s=("decode_tokens_per_sec", "mean"),
            avg_peak_kv_mb=("peak_kv_cache_mb", "mean"),
            avg_prefill_s=("prefill_time_s", "mean"),
        )
        .reset_index()
    )
