import numpy as np
import pytest
from model.config import GPTConfig
from model.transformer import GPT


def test_model_forward_shape():
    config = GPTConfig(
        vocab_size=300,
        context_length=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        norm_type="rmsnorm",
        pos_emb_type="rope",
        activation="swiglu",
        weight_tying=True,
    )
    model = GPT(config, seed=42)
    x = np.random.randint(0, 300, size=(2, 16), dtype=np.int64)
    logits, cache, kv_cache = model.forward(x)

    assert logits.shape == (2, 16, 300)
    assert not np.isnan(logits).any()


def test_model_loss_and_gradients():
    config = GPTConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
        norm_type="rmsnorm",
        pos_emb_type="rope",
        activation="swiglu",
        weight_tying=True,
    )
    model = GPT(config, seed=42)
    x = np.random.randint(0, 100, size=(2, 8), dtype=np.int64)
    y = np.random.randint(0, 100, size=(2, 8), dtype=np.int64)

    loss, grads = model.loss_and_gradients(x, y)

    assert loss > 0.0
    assert not np.isnan(loss)
    for name, g in grads.items():
        assert not np.isnan(g).any(), f"NaN in gradient for {name}"
        assert g.shape == model.params[name].shape, f"Shape mismatch for {name}"


def test_kv_cache_equivalence():
    """Verify that KV-cached autoregressive generation matches full-context generation."""
    config = GPTConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
        norm_type="rmsnorm",
        pos_emb_type="rope",
        activation="swiglu",
    )
    model = GPT(config, seed=42)

    prompt = [5, 12, 18]
    # Full forward pass on prompt
    x_prompt = np.array([prompt], dtype=np.int64)
    logits_full, _, kv_cache = model.forward(x_prompt, start_pos=0, kv_caches=[None] * config.num_layers)

    # Next token prediction
    next_token = int(np.argmax(logits_full[0, -1]))

    # Step forward with single token using KV cache
    x_next = np.array([[next_token]], dtype=np.int64)
    logits_step, _, _ = model.forward(x_next, start_pos=len(prompt), kv_caches=kv_cache)

    # Also compute via standard forward on combined prompt + next_token
    combined = prompt + [next_token]
    logits_ground_truth, _, _ = model.forward(np.array([combined], dtype=np.int64), start_pos=0)

    # Step prediction should match ground truth up to numerical tolerance
    diff = np.max(np.abs(logits_step[0, 0] - logits_ground_truth[0, -1]))
    assert diff < 1e-4, f"KV cache discrepancy: max diff {diff}"


def test_checkpoint_save_and_load(tmp_path):
    config = GPTConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=32,
        num_heads=2,
        num_layers=2,
    )
    model = GPT(config, seed=42)
    ckpt_file = tmp_path / "test.npz"
    cfg_file = tmp_path / "test.json"

    model.save(ckpt_file, cfg_file, optimizer_state={"step": 10})
    loaded_model, opt_state = GPT.load(ckpt_file, cfg_file)

    assert loaded_model.config.vocab_size == config.vocab_size
    assert opt_state is not None
    assert int(opt_state["step"]) == 10

    # Ensure weights match exactly
    for k in model.params:
        np.testing.assert_allclose(model.params[k], loaded_model.params[k])
