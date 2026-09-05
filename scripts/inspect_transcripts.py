"""Inspect gsm8k_transcripts.jsonl for signs of degenerate generation under eviction.

Standalone CLI, not imported by anything else -- run after a sweep that set
transcripts_path, to look at *why* generations are truncating/failing rather
than just whether they're correct. Motivated by the first full sweep: 69% of
generations hit the max_new_tokens cap, and truncated generations dropped to
~5% accuracy vs ~70% for generations that stopped naturally -- summary stats
alone don't say whether that's the model looping/repeating under eviction, or
something else.

Usage:
    python scripts/inspect_transcripts.py ../results/gsm8k_transcripts.jsonl
    python scripts/inspect_transcripts.py ../results/gsm8k_transcripts.jsonl --policy entropy_salience --budget 64
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_transcripts(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ngram_repetition_ratio(text: str, n: int = 8) -> float:
    """Fraction of n-grams (by whitespace token) that are exact repeats of an
    earlier n-gram in the same text. High values indicate the model looping
    on the same phrase/sentence instead of making forward progress -- a
    plausible failure mode for eviction discarding the token(s) that would
    have signaled "I already covered this."
    """
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def has_answer_line(text: str) -> bool:
    return re.search(r"####\s*-?\d", text) is not None


def strict_correct(row: dict) -> bool:
    if "strict_correct" in row:
        return bool(row["strict_correct"])
    predicted = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", row["generated_text"])
    gold = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", row["gold_answer"])
    if predicted is None or gold is None:
        return False
    predicted_value = float(predicted.group(1).replace(",", ""))
    gold_value = float(gold.group(1).replace(",", ""))
    return abs(predicted_value - gold_value) < 1e-4


def summarize_group(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    truncated = [r for r in rows if not has_answer_line(r["generated_text"])]
    rep_ratios = [ngram_repetition_ratio(r["generated_text"]) for r in rows]
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "strict_accuracy": sum(strict_correct(r) for r in rows) / n,
        "pct_no_answer_line": len(truncated) / n,
        "avg_generated_tokens": sum(r["generated_tokens"] for r in rows) / n,
        "avg_8gram_repetition_ratio": sum(rep_ratios) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcripts_path", type=Path)
    parser.add_argument("--policy", default=None, help="filter to one policy name")
    parser.add_argument("--budget", type=int, default=None, help="filter to one generation_budget")
    parser.add_argument(
        "--show-worst", type=int, default=0, help="print the N most-repetitive transcripts in the filtered set"
    )
    args = parser.parse_args()

    rows = load_transcripts(args.transcripts_path)
    if args.policy:
        rows = [r for r in rows if r["policy"] == args.policy]
    if args.budget:
        rows = [r for r in rows if r["generation_budget"] == args.budget]

    print(f"loaded {len(rows)} transcripts from {args.transcripts_path}")
    print()

    by_policy_budget: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by_policy_budget.setdefault((r["policy"], r["generation_budget"]), []).append(r)

    print(
        f"{'policy':<20}{'budget':>8}{'n':>5}{'acc':>8}{'strict':>9}"
        f"{'no_answer%':>12}{'avg_gen_tok':>13}{'avg_8gram_rep':>15}"
    )
    for (policy, budget), group in sorted(by_policy_budget.items()):
        s = summarize_group(group)
        print(
            f"{policy:<20}{budget:>8}{s['n']:>5}{s['accuracy']:>8.2f}"
            f"{s['strict_accuracy']:>9.2f}"
            f"{100 * s['pct_no_answer_line']:>11.1f}%{s['avg_generated_tokens']:>13.1f}"
            f"{s['avg_8gram_repetition_ratio']:>15.3f}"
        )

    if args.show_worst > 0:
        ranked = sorted(rows, key=lambda r: ngram_repetition_ratio(r["generated_text"]), reverse=True)
        print(f"\n--- {args.show_worst} most repetitive transcripts in filtered set ---")
        for r in ranked[: args.show_worst]:
            rep = ngram_repetition_ratio(r["generated_text"])
            print(
                f"\n[{r['policy']} budget={r['generation_budget']} example_idx={r['example_idx']} "
                f"correct={r['correct']} gen_tokens={r['generated_tokens']} 8gram_rep={rep:.3f}]"
            )
            print(f"question: {r['question'][:200]}")
            text = r["generated_text"]
            print(f"generated_text (last 600 chars): ...{text[-600:]}")


if __name__ == "__main__":
    main()
