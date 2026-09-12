"""Model configuration presets and dataclasses for NumPy GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.0
    norm_type: str = "rmsnorm"     # 'rmsnorm' or 'layernorm'
    pos_emb_type: str = "rope"      # 'rope' or 'learned'
    activation: str = "swiglu"      # 'swiglu', 'gelu', or 'relu'
    weight_tying: bool = True       # tie lm_head weights to wte
    bias: bool = False              # include additive biases in linear projections
    eps: float = 1e-5

    # Backwards-compatibility aliases with earlier NumPy-GPT releases
    @property
    def block_size(self) -> int:
        return self.context_length

    @property
    def n_embd(self) -> int:
        return self.embedding_dim

    @property
    def n_head(self) -> int:
        return self.num_heads

    @property
    def n_layer(self) -> int:
        return self.num_layers

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        if self.pos_emb_type not in ("rope", "learned"):
            raise ValueError(f"Unsupported pos_emb_type: {self.pos_emb_type}")
        if self.activation not in ("swiglu", "gelu", "relu"):
            raise ValueError(f"Unsupported activation: {self.activation}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Add legacy keys so older loaders can read the config file
        d["block_size"] = self.context_length
        d["n_embd"] = self.embedding_dim
        d["n_head"] = self.num_heads
        d["n_layer"] = self.num_layers
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTConfig":
        clean = dict(data)
        # Remap legacy keys if modern keys are not present
        if "context_length" not in clean and "block_size" in clean:
            clean["context_length"] = clean["block_size"]
        if "embedding_dim" not in clean and "n_embd" in clean:
            clean["embedding_dim"] = clean["n_embd"]
        if "num_heads" not in clean and "n_head" in clean:
            clean["num_heads"] = clean["n_head"]
        if "num_layers" not in clean and "n_layer" in clean:
            clean["num_layers"] = clean["n_layer"]

        # Default legacy configs to layernorm, learned pos_emb, relu, separate head for 100% exact compatibility
        is_legacy = "norm_type" not in clean and "pos_emb_type" not in clean
        if is_legacy:
            clean.setdefault("norm_type", "layernorm")
            clean.setdefault("pos_emb_type", "learned")
            clean.setdefault("activation", "relu")
            clean.setdefault("weight_tying", False)
            clean.setdefault("bias", True)

        # Filter to dataclass fields
        valid_keys = {
            "vocab_size", "context_length", "embedding_dim", "num_heads",
            "num_layers", "dropout", "norm_type", "pos_emb_type",
            "activation", "weight_tying", "bias", "eps"
        }
        filtered = {k: v for k, v in clean.items() if k in valid_keys}
        return cls(**filtered)

    def estimate_params(self) -> int:
        c = self.embedding_dim
        v = self.vocab_size
        l = self.num_layers

        # Token embedding
        params = v * c

        # Position embedding
        if self.pos_emb_type == "learned":
            params += self.context_length * c

        # Layers
        for _ in range(l):
            # Attention (QKV + Proj)
            params += c * (3 * c) + (3 * c if self.bias else 0)
            params += c * c + (c if self.bias else 0)
            # Norm 1
            params += c + (c if self.norm_type == "layernorm" else 0)

            # MLP
            if self.activation == "swiglu":
                hidden = int(8 * c / 3)
                hidden = ((hidden + 31) // 32) * 32  # align to 32
                # Gate + Up + Down
                params += 2 * (c * hidden) + (2 * hidden if self.bias else 0)
                params += hidden * c + (c if self.bias else 0)
            else:
                hidden = 4 * c
                params += c * hidden + (hidden if self.bias else 0)
                params += hidden * c + (c if self.bias else 0)
            # Norm 2
            params += c + (c if self.norm_type == "layernorm" else 0)

        # Final norm
        params += c + (c if self.norm_type == "layernorm" else 0)

        # LM Head
        if not self.weight_tying:
            params += c * v + (v if self.bias else 0)
        elif self.bias:
            params += v

        return params


# Predefined architectural configurations tailored to local consumer hardware
CONFIG_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "context_length": 256,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "norm_type": "rmsnorm",
        "pos_emb_type": "rope",
        "activation": "swiglu",
        "weight_tying": True,
        "bias": False,
    },
    "small": {
        "context_length": 256,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "norm_type": "rmsnorm",
        "pos_emb_type": "rope",
        "activation": "swiglu",
        "weight_tying": True,
        "bias": False,
    },
    "medium": {
        "context_length": 512,
        "embedding_dim": 384,
        "num_heads": 12,
        "num_layers": 8,
        "norm_type": "rmsnorm",
        "pos_emb_type": "rope",
        "activation": "swiglu",
        "weight_tying": True,
        "bias": False,
    },
    "large": {
        "context_length": 512,
        "embedding_dim": 512,
        "num_heads": 16,
        "num_layers": 12,
        "norm_type": "rmsnorm",
        "pos_emb_type": "rope",
        "activation": "swiglu",
        "weight_tying": True,
        "bias": False,
    },
}


def get_preset_config(name: str, vocab_size: int) -> GPTConfig:
    preset = CONFIG_PRESETS.get(name.lower())
    if preset is None:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(CONFIG_PRESETS.keys())}")
    return GPTConfig(vocab_size=vocab_size, **preset)
