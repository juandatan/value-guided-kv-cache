import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from vgkv.eval.runner import run_policy_eval, summarize


class RunnerSummaryTests(unittest.TestCase):
    def test_summary_reports_permissive_and_strict_accuracy(self):
        results = pd.DataFrame(
            [
                {
                    "policy": "recency_lru",
                    "generation_budget": 512,
                    "correct": True,
                    "strict_correct": False,
                    "decode_tokens_per_sec": 10.0,
                    "peak_kv_cache_mb": 20.0,
                    "prefill_time_s": 0.1,
                },
                {
                    "policy": "recency_lru",
                    "generation_budget": 512,
                    "correct": True,
                    "strict_correct": True,
                    "decode_tokens_per_sec": 12.0,
                    "peak_kv_cache_mb": 22.0,
                    "prefill_time_s": 0.2,
                },
            ]
        )

        summary = summarize(results).iloc[0]

        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["strict_accuracy"], 0.5)

    def test_resume_rejects_checkpoint_without_strict_metric(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "old.csv"
            pd.DataFrame(
                [
                    {
                        "example_idx": 0,
                        "policy": "recency_lru",
                        "generation_budget": 512,
                        "correct": True,
                    }
                ]
            ).to_csv(checkpoint, index=False)

            with self.assertRaisesRegex(ValueError, "predates the strict accuracy metric"):
                run_policy_eval(
                    model=None,
                    tokenizer=None,
                    dataset=[],
                    policy_specs=[],
                    max_new_tokens=1,
                    checkpoint_path=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
