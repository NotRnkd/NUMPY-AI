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
