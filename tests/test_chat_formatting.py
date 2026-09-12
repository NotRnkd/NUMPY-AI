import numpy as np
import pytest
from tokenizer.bpe import ByteBPETokenizer
from training.dataset import ChatDataset


def test_chat_dataset_masking():
    tok = ByteBPETokenizer.train("System and User sample text assistant reply", vocab_size=300, min_frequency=1, verbose=False)

    samples = [
        {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello world!"},
                {"role": "assistant", "content": "Hi there, nice to meet you."},
            ]
        }
    ]

    dataset = ChatDataset(samples, tokenizer=tok, context_length=64, val_ratio=0.0)
    x, y, mask = dataset.get_batch(batch_size=1, split="train")

    assert x.shape == (1, 64)
    assert y.shape == (1, 64)
    assert mask.shape == (1, 64)

    # Mask should have both 0.0 (prompt/system/user/padding) and 1.0 (assistant tokens)
    assert np.any(mask == 1.0)
    assert np.any(mask == 0.0)


def test_sharegpt_and_alpaca_formatting(tmp_path):
    tok = ByteBPETokenizer.train("System and User sample text assistant reply human gpt", vocab_size=300, min_frequency=1, verbose=False)

    # ShareGPT format
    sharegpt_sample = {
        "conversations": [
            {"from": "human", "value": "How do I calculate 2+2?"},
            {"from": "gpt", "value": "2 + 2 equals 4."},
        ]
    }

    # Alpaca format with context
    alpaca_sample = {
        "instruction": "Summarize the following passage.",
        "input": "NumPy-GPT is an educational transformer written without PyTorch.",
        "output": "NumPy-GPT is a pure NumPy neural language model.",
    }

    dataset = ChatDataset([sharegpt_sample, alpaca_sample], tokenizer=tok, context_length=64, val_ratio=0.0)
    assert len(dataset.train_samples) == 2
    x, y, mask = dataset.get_batch(batch_size=2, split="train")
    assert x.shape == (2, 64)
    assert np.any(mask == 1.0)

    # Test JSONL file reading
    jsonl_path = tmp_path / "chat_data.jsonl"
    import json
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sharegpt_sample) + "\n")
        f.write(json.dumps(alpaca_sample) + "\n")

    loaded_dataset = ChatDataset.load_json(jsonl_path, tokenizer=tok, context_length=64, val_ratio=0.0)
    assert len(loaded_dataset.train_samples) == 2

