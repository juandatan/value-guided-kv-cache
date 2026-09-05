import unittest
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from vgkv.decode import DecodeConfig, generate_with_policy
from vgkv.value_models.policies import RecencyPolicy


class FakeBatch(dict):
    def to(self, device):
        self["input_ids"] = self["input_ids"].to(device)
        return self


class FakeTokenizer:
    eos_token_id = None

    def __call__(self, prompt, return_tensors):
        return FakeBatch(input_ids=torch.tensor([[10, 11, 12]], dtype=torch.long))

    def decode(self, token_ids, skip_special_tokens):
        return " ".join(map(str, token_ids))


class PositionRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.calls = []

    def forward(
        self,
        input_ids,
        past_key_values,
        use_cache,
        output_attentions,
        cache_position,
        position_ids,
    ):
        self.assertIsDynamicCache(past_key_values)
        past_len = past_key_values.get_seq_length()
        self.calls.append(
            {
                "past_len": past_len,
                "cache_position": cache_position.detach().cpu().tolist(),
                "position_ids": position_ids.detach().cpu().tolist(),
            }
        )

        q_len = input_ids.shape[1]
        key = torch.zeros(1, 1, q_len, 1)
        value = torch.zeros_like(key)
        key, _ = past_key_values.update(key, value, layer_idx=0)
        kv_len = key.shape[-2]

        attentions = (torch.full((1, 1, q_len, kv_len), 1.0 / kv_len),)
        logits = torch.zeros(1, q_len, 3)
        logits[..., 1] = 1.0
        return SimpleNamespace(logits=logits, attentions=attentions)

    def assertIsDynamicCache(self, cache):
        if not isinstance(cache, DynamicCache):
            raise AssertionError(f"expected DynamicCache, got {type(cache)!r}")


class DecodePositionTests(unittest.TestCase):
    def test_absolute_positions_keep_increasing_after_cache_is_capped(self):
        model = PositionRecordingModel()
        _, metrics = generate_with_policy(
            model,
            FakeTokenizer(),
            prompt="ignored",
            policy=RecencyPolicy(),
            config=DecodeConfig(max_new_tokens=6, generation_budget=2),
        )

        self.assertEqual(metrics.generated_tokens, 6)
        self.assertEqual(
            [call["cache_position"] for call in model.calls],
            [[0, 1, 2], [3], [4], [5], [6], [7]],
        )
        self.assertEqual(
            [call["position_ids"] for call in model.calls],
            [[[0, 1, 2]], [[3]], [[4]], [[5]], [[6]], [[7]]],
        )

        # Eviction caps the physical cache at prompt_len + generation_budget
        # before the calls for positions 6 and 7, but their absolute positions
        # continue increasing rather than repeating physical cache length 5.
        self.assertEqual([call["past_len"] for call in model.calls], [0, 3, 4, 5, 5, 5])

    def test_tiny_qwen_accepts_absolute_positions_beyond_physical_cache_length(self):
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = Qwen2ForCausalLM(config).eval()

        text, metrics = generate_with_policy(
            model,
            FakeTokenizer(),
            prompt="ignored",
            policy=RecencyPolicy(),
            config=DecodeConfig(max_new_tokens=8, generation_budget=2),
        )

        self.assertEqual(metrics.generated_tokens, 8)
        self.assertEqual(len(text.split()), 8)


if __name__ == "__main__":
    unittest.main()
