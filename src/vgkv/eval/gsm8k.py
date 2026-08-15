"""GSM8K loading, prompt formatting, and answer scoring."""

import re

from datasets import load_dataset

ANSWER_TAG_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
BOXED_OR_LAST_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_gsm8k(split: str = "test", limit: int | None = None):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(limit))
    return ds


def gold_answer(example: dict) -> float:
    match = ANSWER_TAG_RE.search(example["answer"])
    if not match:
        raise ValueError(f"Could not parse gold answer from: {example['answer']!r}")
    return float(match.group(1).replace(",", ""))


def build_prompt(example: dict) -> str:
    return (
        "Solve the following grade-school math problem. Think step by step, "
        "then give the final numeric answer on its own line after '#### '.\n\n"
        f"Problem: {example['question']}\n"
    )


def extract_predicted_answer(generated_text: str) -> float | None:
    match = ANSWER_TAG_RE.search(generated_text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = BOXED_OR_LAST_NUMBER_RE.findall(generated_text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def is_correct(generated_text: str, example: dict) -> bool:
    pred = extract_predicted_answer(generated_text)
    if pred is None:
        return False
    return abs(pred - gold_answer(example)) < 1e-4
