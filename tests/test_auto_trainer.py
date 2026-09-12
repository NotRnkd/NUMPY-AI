import time
import numpy as np
import pytest
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer
from training.auto_trainer import AutoTrainer
from training.dataset import ChatDataset


def test_auto_trainer_lifecycle(tmp_path):
    tok = ByteBPETokenizer.train("What is an auto train system in Python? Assistant reply.", vocab_size=300, min_frequency=1, verbose=False)
    cfg = GPTConfig(
        vocab_size=len(tok),
        context_length=32,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
    )
    model = GPT(cfg, seed=42)

    sample = {
        "messages": [
            {"role": "user", "content": "What is an auto train system?"},
            {"role": "assistant", "content": "An auto train system continuously optimizes weights."},
        ]
    }
    dataset = ChatDataset([sample], tokenizer=tok, context_length=32, val_ratio=0.0)

    auto_trainer = AutoTrainer(
        model=model,
        tokenizer=tok,
        dataset=dataset,
        base_lr=0.001,
        batch_size=1,
        checkpoint_dir=tmp_path,
    )

    assert not auto_trainer.is_running
    auto_trainer.start(mode="autonomous_loop")
    assert auto_trainer.is_running

    # Let it run 1-2 small cycles
    time.sleep(0.5)
    telemetry = auto_trainer.get_telemetry()
    assert telemetry["total_steps"] > 0

    auto_trainer.stop()
    assert not auto_trainer.is_running
