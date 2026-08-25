"""Multi-example, multi-policy evaluation runner.

Sweeps a set of policies across a GSM8K subset and returns per-example results
plus an aggregated summary (accuracy, throughput, peak KV cache memory).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from vgkv.decode import DecodeConfig, generate_with_policy
from vgkv.eval.gsm8k import build_prompt, is_correct
from vgkv.value_models.base import EvictionPolicy

logger = logging.getLogger(__name__)


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
    log_every: int = 1,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    force: bool = False,
    transcripts_path: str | Path | None = None,
    kaggle_dataset_slug: str | None = None,
    kaggle_title: str = "vgkv GSM8K policy sweep results",
    kaggle_upload_every: int = 20,
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

    log_every controls how often (in completed example/policy runs) progress
    is logged via the `logging` module -- useful for tracking a long sweep
    without scrolling through per-run prints.

    checkpoint_path, if set, is overwritten with the full results-so-far
    DataFrame after every single (example, policy) run -- so a crash or
    interrupt partway through a long sweep loses at most one run's worth of
    progress, not the whole sweep.

    resume, if True (default) and checkpoint_path already exists, loads it
    and skips any (example_idx, policy, generation_budget) combination
    already present -- lets you re-run this function after a crash/interrupt
    without redoing completed work or re-uploading duplicate rows. Matching
    is keyed on those three columns, not row position, since policy_specs or
    dataset order might differ slightly between the interrupted and resumed
    calls.

    force, if True, ignores and deletes any existing checkpoint_path /
    transcripts_path before starting -- every run is redone from scratch and
    both files are overwritten rather than appended to. Use this when you
    want fresh transcripts for examples that are already checkpointed (the
    plain resume path skips them, so their transcripts would never get
    written), or when a checkpoint is suspected corrupt/stale. Takes
    precedence over resume.

    transcripts_path, if set, appends one JSON line per (example, policy) run
    containing the full generated text alongside the same identifying columns
    as checkpoint_path -- kept as a separate file rather than a column on
    checkpoint_path since full 1536-token transcripts would bloat every
    checkpoint write/upload; this file is meant for offline inspection (e.g.
    reading truncated generations to see why they didn't reach an answer),
    not for the accuracy/throughput summary.

    kaggle_dataset_slug, if set, pushes checkpoint_path to that Kaggle dataset
    (creating it on the first checkpoint, versioning it on every subsequent
    one) every kaggle_upload_every completed runs -- deliberately coarser
    than the local checkpoint_path write, since each upload is a network
    round-trip (zip + API call) and would otherwise dominate sweep runtime if
    fired after every single generation. Requires the `kaggle` CLI and
    credentials configured -- see vgkv.kaggle_utils.
    """
    if force:
        for path in (checkpoint_path, transcripts_path):
            if path is not None and Path(path).exists():
                logger.info("force=True: deleting existing %s", path)
                Path(path).unlink()

    rows = []
    done_keys: set[tuple[int, str, int]] = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        if resume:
            existing = pd.read_csv(checkpoint_path)
            rows = existing.to_dict("records")
            done_keys = set(
                zip(existing["example_idx"], existing["policy"], existing["generation_budget"])
            )
            logger.info(
                "RESUMED from checkpoint %s: %d runs already completed, skipping those",
                checkpoint_path,
                len(done_keys),
            )
        else:
            logger.info(
                "checkpoint %s exists but resume=False: overwriting from scratch", checkpoint_path
            )
    elif checkpoint_path is not None:
        logger.info("no checkpoint found at %s: starting fresh", checkpoint_path)

    resumed_count = len(done_keys)
    total_runs = len(dataset) * len(policy_specs)
    start = time.monotonic()
    for idx, example in enumerate(dataset):
        prompt = build_prompt(example)
        for spec in policy_specs:
            if (idx, spec.name, spec.generation_budget) in done_keys:
                continue
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

            if transcripts_path is not None:
                transcripts_path = Path(transcripts_path)
                transcripts_path.parent.mkdir(parents=True, exist_ok=True)
                with open(transcripts_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "example_idx": idx,
                                "policy": spec.name,
                                "generation_budget": spec.generation_budget,
                                "correct": correct,
                                "generated_tokens": metrics.generated_tokens,
                                "question": example["question"],
                                "gold_answer": example["answer"],
                                "generated_text": text,
                            }
                        )
                        + "\n"
                    )

            if verbose:
                print(
                    f"[{idx}] {spec.name}: correct={correct} "
                    f"tok/s={metrics.decode_tokens_per_sec:.1f} "
                    f"peak_kv_mb={metrics.peak_kv_cache_bytes / (1024**2):.1f}"
                )

            completed = len(rows)
            new_this_session = completed - resumed_count
            if log_every > 0 and new_this_session % log_every == 0:
                elapsed = time.monotonic() - start
                rate = new_this_session / elapsed if elapsed > 0 else 0.0
                eta_s = (total_runs - completed) / rate if rate > 0 else float("inf")
                logger.info(
                    "progress %d/%d (%.1f%%) [resumed=%d new=%d] elapsed=%.1fs eta=%.1fs last=%s/%s correct=%s",
                    completed,
                    total_runs,
                    100 * completed / total_runs,
                    resumed_count,
                    new_this_session,
                    elapsed,
                    eta_s,
                    spec.name,
                    idx,
                    correct,
                )

            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

                is_last_run = completed == total_runs
                if kaggle_dataset_slug is not None and (
                    completed % kaggle_upload_every == 0 or is_last_run
                ):
                    from vgkv.kaggle_utils import sync_file_to_dataset

                    try:
                        sync_file_to_dataset(
                            checkpoint_path,
                            kaggle_dataset_slug,
                            kaggle_title,
                            version_notes=f"checkpoint at {completed}/{total_runs} runs",
                        )
                    except Exception:
                        logger.exception("kaggle checkpoint upload failed, continuing sweep")

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
