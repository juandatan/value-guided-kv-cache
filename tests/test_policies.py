import unittest
from types import SimpleNamespace

import torch

from vgkv.value_models.policies import RecencyPolicy


class FakeState:
    def __init__(self, seq_len: int):
        self._seq_len = seq_len
        self.layers = [SimpleNamespace(attn_accum=torch.zeros(1, seq_len))]

    def seq_len(self, layer_idx: int) -> int:
        return self._seq_len


class RecencyPolicyTests(unittest.TestCase):
    def test_scores_newer_cache_slots_higher(self):
        scores = RecencyPolicy().score(FakeState(5), layer_idx=0)

        torch.testing.assert_close(scores, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
