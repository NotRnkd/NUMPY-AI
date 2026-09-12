import numpy as np
import pytest
from export.onnx_export import export_gpt_to_onnx, run_onnx_inference
from model.config import GPTConfig
from model.transformer import GPT


def test_onnx_export_and_inference(tmp_path):
    config = GPTConfig(
        vocab_size=120,
        context_length=32,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
        norm_type="rmsnorm",
        activation="swiglu",
        weight_tying=True,
    )
    model = GPT(config, seed=42)
    onnx_file = tmp_path / "model.onnx"

    export_gpt_to_onnx(model, onnx_file)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 1000

    test_tokens = np.array([[5, 12, 23, 44]], dtype=np.int64)
    logits_onnx = run_onnx_inference(onnx_file, test_tokens)

    assert logits_onnx.shape == (1, 4, 120)
    assert not np.isnan(logits_onnx).any()
