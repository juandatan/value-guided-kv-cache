import unittest

from vgkv.eval.gsm8k import (
    extract_predicted_answer,
    extract_strict_predicted_answer,
    is_correct,
    is_strict_correct,
)


EXAMPLE = {"answer": "The calculation gives the result.\n#### 18"}


class Gsm8kScoringTests(unittest.TestCase):
    def test_strict_answer_requires_explicit_tag(self):
        text = "The answer is 18."

        self.assertEqual(extract_predicted_answer(text), 18.0)
        self.assertIsNone(extract_strict_predicted_answer(text))
        self.assertTrue(is_correct(text, EXAMPLE))
        self.assertFalse(is_strict_correct(text, EXAMPLE))

    def test_strict_answer_accepts_tag_and_commas(self):
        example = {"answer": "#### 1200"}
        text = "Done.\n#### 1,200"

        self.assertEqual(extract_strict_predicted_answer(text), 1200.0)
        self.assertTrue(is_correct(text, example))
        self.assertTrue(is_strict_correct(text, example))

    def test_wrong_explicit_answer_is_wrong_for_both_metrics(self):
        text = "Done.\n#### 17"

        self.assertFalse(is_correct(text, EXAMPLE))
        self.assertFalse(is_strict_correct(text, EXAMPLE))


if __name__ == "__main__":
    unittest.main()
