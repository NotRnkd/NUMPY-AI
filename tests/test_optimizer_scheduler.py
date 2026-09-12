import numpy as np
import pytest
from training.optimizer import AdamW
from training.scheduler import CosineWarmupScheduler


def test_scheduler():
    scheduler = CosineWarmupScheduler(base_lr=1e-3, warmup_steps=100, max_steps=1000, min_lr=1e-5)
    # Warmup start
    assert scheduler.get_lr(0) == 1e-5
    # Warmup midpoint
    assert 1e-5 < scheduler.get_lr(50) < 1e-3
    # Peak at warmup end
    assert np.isclose(scheduler.get_lr(100), 1e-3)
    # Decay midpoint
    assert 1e-5 < scheduler.get_lr(550) < 1e-3
    # Max steps reached
    assert np.isclose(scheduler.get_lr(1000), 1e-5)
    # Post max steps remains min_lr
    assert scheduler.get_lr(1200) == 1e-5


def test_adamw_weight_decay_exclusion():
    params = {
        "wte": np.ones((10, 8), dtype=np.float32),
        "ln1_g": np.ones((8,), dtype=np.float32),   # 1D norm gain
        "bias": np.zeros((8,), dtype=np.float32),   # 1D bias
    }
    opt = AdamW(params, lr=1e-2, weight_decay=0.1)

    # Gradient is zero
    zero_grads = {k: np.zeros_like(v) for k, v in params.items()}
    opt.step(zero_grads)

    # 2D weight matrix should decay
    assert float(np.mean(params["wte"])) < 1.0
    # 1D normalization and bias MUST NOT decay
    assert float(np.mean(params["ln1_g"])) == 1.0
    assert float(np.mean(params["bias"])) == 0.0


def test_optimizer_state_dict_roundtrip():
    params = {"w": np.ones((4, 4), dtype=np.float32)}
    opt = AdamW(params, lr=1e-3)
    grads = {"w": np.random.randn(4, 4).astype(np.float32)}
    opt.step(grads)

    state = opt.state_dict()
    assert state["step"] == 1

    opt2 = AdamW(params, lr=1e-3)
    opt2.load_state_dict(state)
    assert opt2.step_count == 1
    np.testing.assert_allclose(opt.exp_avg["w"], opt2.exp_avg["w"])
