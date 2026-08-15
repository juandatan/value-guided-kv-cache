"""Model loading for the value-guided KV cache engine.

Uses eager attention so attention weights are returned per step -- required
for attention-based value signals (entropy, centrality). SDPA/flash-attn
backends do not expose attention weights.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def load_model_and_tokenizer(model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer
