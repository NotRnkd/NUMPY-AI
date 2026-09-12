"""Modern Transformer and complete GPT decoder-only language model in NumPy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np_cpu
from hardware.backend import get_backend, indexed_add, to_cpu, to_device
from model.attention import CausalSelfAttention
from model.config import GPTConfig
from model.layers import (
    cross_entropy_loss_and_grad,
    gelu_backward,
    gelu_forward,
    layer_norm_backward,
    layer_norm_forward,
    relu_backward,
    relu_forward,
    rms_norm_backward,
    rms_norm_forward,
    silu,
    silu_backward,
)


class TransformerBlock:
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            context_length=config.context_length,
            use_rope=(config.pos_emb_type == "rope"),
            has_bias=config.bias,
        )

        if config.activation == "swiglu":
            # SwiGLU hidden dimension: 8/3 * d, aligned to 32
            hidden = int(8 * config.embedding_dim / 3)
            self.hidden_dim = ((hidden + 31) // 32) * 32
        else:
            self.hidden_dim = 4 * config.embedding_dim

    def forward(
        self,
        h: Any,
        params: Dict[str, Any],
        start_pos: int = 0,
        kv_cache: Optional[Tuple[Any, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[Any, dict, Optional[Tuple[Any, Any]]]:
        p = params
        prefix = f"blocks.{self.layer_idx}."
        cfg = self.config

        # 1. Pre-Attention Normalization
        if cfg.norm_type == "rmsnorm":
            norm1, norm1_cache = rms_norm_forward(h, p[prefix + "ln1_g"], eps=cfg.eps)
        else:
            norm1, norm1_cache = layer_norm_forward(h, p[prefix + "ln1_g"], p[prefix + "ln1_b"], eps=cfg.eps)

        # 2. Causal Self-Attention
        attn_out, attn_cache, new_kv = self.attn.forward(
            norm1,
            w_qkv=p[prefix + "qkv_w"],
            b_qkv=p.get(prefix + "qkv_b"),
            w_proj=p[prefix + "proj_w"],
            b_proj=p.get(prefix + "proj_b"),
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        h1 = h + attn_out

        # 3. Pre-FFN Normalization
        if cfg.norm_type == "rmsnorm":
            norm2, norm2_cache = rms_norm_forward(h1, p[prefix + "ln2_g"], eps=cfg.eps)
        else:
            norm2, norm2_cache = layer_norm_forward(h1, p[prefix + "ln2_g"], p[prefix + "ln2_b"], eps=cfg.eps)

        # 4. Feed-Forward Network
        mlp_cache = {}
        if cfg.activation == "swiglu":
            # SwiGLU: (silu(x @ W_gate) * (x @ W_up)) @ W_down
            gate_proj = norm2 @ p[prefix + "ffn_gate_w"]
            up_proj = norm2 @ p[prefix + "ffn_up_w"]
            if cfg.bias:
                gate_proj = gate_proj + p.get(prefix + "ffn_gate_b")
                up_proj = up_proj + p.get(prefix + "ffn_up_b")

            act_gate, silu_s = silu(gate_proj)
            ffn_hidden = act_gate * up_proj
            ffn_out = ffn_hidden @ p[prefix + "ffn_down_w"]
            if cfg.bias:
                ffn_out = ffn_out + p.get(prefix + "ffn_down_b")

            mlp_cache = {
                "gate_proj": gate_proj,
                "up_proj": up_proj,
                "act_gate": act_gate,
                "silu_s": silu_s,
                "ffn_hidden": ffn_hidden,
            }
        elif cfg.activation == "gelu":
            ff1 = norm2 @ p[prefix + "ff1_w"]
            if cfg.bias:
                ff1 = ff1 + p.get(prefix + "ff1_b")
            act, gelu_cache = gelu_forward(ff1)
            ffn_out = act @ p[prefix + "ff2_w"]
            if cfg.bias:
                ffn_out = ffn_out + p.get(prefix + "ff2_b")
            mlp_cache = {"ff1": ff1, "act": act, "gelu_cache": gelu_cache}
        else:  # relu (legacy)
            ff1 = norm2 @ p[prefix + "ff1_w"]
            if cfg.bias:
                ff1 = ff1 + p.get(prefix + "ff1_b")
            act, relu_cache = relu_forward(ff1)
            ffn_out = act @ p[prefix + "ff2_w"]
            if cfg.bias:
                ffn_out = ffn_out + p.get(prefix + "ff2_b")
            mlp_cache = {"ff1": ff1, "act": act, "relu_cache": relu_cache}

        h2 = h1 + ffn_out

        cache = {
            "h": h,
            "norm1": norm1,
            "norm1_cache": norm1_cache,
            "attn_cache": attn_cache,
            "h1": h1,
            "norm2": norm2,
            "norm2_cache": norm2_cache,
            "mlp_cache": mlp_cache,
        }
        return h2, cache, new_kv

    def backward(
        self,
        dh2: Any,
        cache: dict,
        params: Dict[str, Any],
        grads: Dict[str, Any],
    ) -> Any:
        prefix = f"blocks.{self.layer_idx}."
        cfg = self.config
        p = params

        # Backward through residual: h2 = h1 + ffn_out
        dh1 = dh2.copy()
        dffn_out = dh2

        # Backward through MLP
        norm2 = cache["norm2"]
        mlp_c = cache["mlp_cache"]

        if cfg.activation == "swiglu":
            w_gate = p[prefix + "ffn_gate_w"]
            w_up = p[prefix + "ffn_up_w"]
            w_down = p[prefix + "ffn_down_w"]

            # dffn_down_w
            hidden = mlp_c["ffn_hidden"]
            grads[prefix + "ffn_down_w"] = hidden.reshape(-1, hidden.shape[-1]).T @ dffn_out.reshape(-1, cfg.embedding_dim)
            if cfg.bias:
                grads[prefix + "ffn_down_b"] = dffn_out.sum(axis=(0, 1))

            dhidden = dffn_out @ w_down.T

            # hidden = act_gate * up_proj
            d_act_gate = dhidden * mlp_c["up_proj"]
            d_up_proj = dhidden * mlp_c["act_gate"]

            # d_gate_proj from silu
            d_gate_proj = silu_backward(d_act_gate, mlp_c["gate_proj"], mlp_c["silu_s"])

            # grads for gate and up weights
            grads[prefix + "ffn_gate_w"] = norm2.reshape(-1, cfg.embedding_dim).T @ d_gate_proj.reshape(-1, self.hidden_dim)
            grads[prefix + "ffn_up_w"] = norm2.reshape(-1, cfg.embedding_dim).T @ d_up_proj.reshape(-1, self.hidden_dim)
            if cfg.bias:
                grads[prefix + "ffn_gate_b"] = d_gate_proj.sum(axis=(0, 1))
                grads[prefix + "ffn_up_b"] = d_up_proj.sum(axis=(0, 1))

            dnorm2 = d_gate_proj @ w_gate.T + d_up_proj @ w_up.T

        elif cfg.activation == "gelu":
            w1 = p[prefix + "ff1_w"]
            w2 = p[prefix + "ff2_w"]
            grads[prefix + "ff2_w"] = mlp_c["act"].reshape(-1, 4 * cfg.embedding_dim).T @ dffn_out.reshape(-1, cfg.embedding_dim)
            if cfg.bias:
                grads[prefix + "ff2_b"] = dffn_out.sum(axis=(0, 1))

            dact = dffn_out @ w2.T
            dff1 = gelu_backward(dact, mlp_c["gelu_cache"])
            grads[prefix + "ff1_w"] = norm2.reshape(-1, cfg.embedding_dim).T @ dff1.reshape(-1, 4 * cfg.embedding_dim)
            if cfg.bias:
                grads[prefix + "ff1_b"] = dff1.sum(axis=(0, 1))
            dnorm2 = dff1 @ w1.T

        else:  # relu
            w1 = p[prefix + "ff1_w"]
            w2 = p[prefix + "ff2_w"]
            grads[prefix + "ff2_w"] = mlp_c["act"].reshape(-1, 4 * cfg.embedding_dim).T @ dffn_out.reshape(-1, cfg.embedding_dim)
            if cfg.bias:
                grads[prefix + "ff2_b"] = dffn_out.sum(axis=(0, 1))

            dact = dffn_out @ w2.T
            dff1 = relu_backward(dact, mlp_c["relu_cache"])
            grads[prefix + "ff1_w"] = norm2.reshape(-1, cfg.embedding_dim).T @ dff1.reshape(-1, 4 * cfg.embedding_dim)
            if cfg.bias:
                grads[prefix + "ff1_b"] = dff1.sum(axis=(0, 1))
            dnorm2 = dff1 @ w1.T

        # Backward through norm2
        if cfg.norm_type == "rmsnorm":
            dh1_from_norm, dgamma2 = rms_norm_backward(dnorm2, cache["norm2_cache"])
            grads[prefix + "ln2_g"] = dgamma2
        else:
            dh1_from_norm, dgamma2, dbeta2 = layer_norm_backward(dnorm2, cache["norm2_cache"])
            grads[prefix + "ln2_g"] = dgamma2
            grads[prefix + "ln2_b"] = dbeta2
        dh1 += dh1_from_norm

        # Backward through residual: h1 = h + attn_out
        dh = dh1.copy()
        dattn_out = dh1

        # Backward through attention
        dnorm1, dw_qkv, db_qkv, dw_proj, db_proj = self.attn.backward(
            dattn_out,
            cache["attn_cache"],
            w_qkv=p[prefix + "qkv_w"],
            w_proj=p[prefix + "proj_w"],
        )
        grads[prefix + "qkv_w"] = dw_qkv
        grads[prefix + "proj_w"] = dw_proj
        if cfg.bias and db_qkv is not None:
            grads[prefix + "qkv_b"] = db_qkv
        if cfg.bias and db_proj is not None:
            grads[prefix + "proj_b"] = db_proj

        # Backward through norm1
        if cfg.norm_type == "rmsnorm":
            dh_from_norm, dgamma1 = rms_norm_backward(dnorm1, cache["norm1_cache"])
            grads[prefix + "ln1_g"] = dgamma1
        else:
            dh_from_norm, dgamma1, dbeta1 = layer_norm_backward(dnorm1, cache["norm1_cache"])
            grads[prefix + "ln1_g"] = dgamma1
            grads[prefix + "ln1_b"] = dbeta1
        dh += dh_from_norm

        return dh


class GPT:
    """Decoder-only Transformer Language Model."""

    def __init__(self, config: GPTConfig, seed: int = 1337) -> None:
        self.config = config
        self.rng = np_cpu.random.default_rng(seed)
        self.params: Dict[str, Any] = {}
        self.blocks = [TransformerBlock(config, i) for i in range(config.num_layers)]
        self._initialize_parameters()

    def _random_matrix(self, shape: Tuple[int, ...], std: float = 0.02) -> Any:
        xp = get_backend()
        return xp.asarray(self.rng.normal(0.0, std, size=shape).astype(np_cpu.float32))

    def _initialize_parameters(self) -> None:
        cfg = self.config
        xp = get_backend()
        c = cfg.embedding_dim
        v = cfg.vocab_size

        # Residual scale factor: 1 / sqrt(2 * num_layers) for projection stability
        proj_std = 0.02 / math.sqrt(2.0 * cfg.num_layers)

        # 1. Token Embeddings
        self.params["wte"] = self._random_matrix((v, c), std=0.02)

        # 2. Learned Positional Embeddings (if enabled)
        if cfg.pos_emb_type == "learned":
            self.params["wpe"] = self._random_matrix((cfg.context_length, c), std=0.01)

        # 3. Transformer Blocks
        for i in range(cfg.num_layers):
            prefix = f"blocks.{i}."
            # Norm 1
            self.params[prefix + "ln1_g"] = xp.ones(c, dtype=xp.float32)
            if cfg.norm_type == "layernorm":
                self.params[prefix + "ln1_b"] = xp.zeros(c, dtype=xp.float32)

            # Attention
            self.params[prefix + "qkv_w"] = self._random_matrix((c, 3 * c), std=0.02)
            if cfg.bias:
                self.params[prefix + "qkv_b"] = xp.zeros(3 * c, dtype=xp.float32)

            self.params[prefix + "proj_w"] = self._random_matrix((c, c), std=proj_std)
            if cfg.bias:
                self.params[prefix + "proj_b"] = xp.zeros(c, dtype=xp.float32)

            # Norm 2
            self.params[prefix + "ln2_g"] = xp.ones(c, dtype=xp.float32)
            if cfg.norm_type == "layernorm":
                self.params[prefix + "ln2_b"] = xp.zeros(c, dtype=xp.float32)

            # MLP
            if cfg.activation == "swiglu":
                h_dim = self.blocks[i].hidden_dim
                self.params[prefix + "ffn_gate_w"] = self._random_matrix((c, h_dim), std=0.02)
                self.params[prefix + "ffn_up_w"] = self._random_matrix((c, h_dim), std=0.02)
                self.params[prefix + "ffn_down_w"] = self._random_matrix((h_dim, c), std=proj_std)
                if cfg.bias:
                    self.params[prefix + "ffn_gate_b"] = xp.zeros(h_dim, dtype=xp.float32)
                    self.params[prefix + "ffn_up_b"] = xp.zeros(h_dim, dtype=xp.float32)
                    self.params[prefix + "ffn_down_b"] = xp.zeros(c, dtype=xp.float32)
            else:
                hidden = 4 * c
                self.params[prefix + "ff1_w"] = self._random_matrix((c, hidden), std=0.02)
                self.params[prefix + "ff2_w"] = self._random_matrix((hidden, c), std=proj_std)
                if cfg.bias:
                    self.params[prefix + "ff1_b"] = xp.zeros(hidden, dtype=xp.float32)
                    self.params[prefix + "ff2_b"] = xp.zeros(c, dtype=xp.float32)

        # 4. Final Normalization
        self.params["ln_f_g"] = xp.ones(c, dtype=xp.float32)
        if cfg.norm_type == "layernorm":
            self.params["ln_f_b"] = xp.zeros(c, dtype=xp.float32)

        # 5. Language Model Head
        if not cfg.weight_tying:
            self.params["lm_head_w"] = self._random_matrix((c, v), std=0.02)
        if cfg.bias:
            self.params["lm_head_b"] = xp.zeros(v, dtype=xp.float32)

    def parameter_count(self) -> int:
        return sum(int(val.size) for val in self.params.values())

    def to_device(self) -> None:
        """Ensure all parameters reside on active device (NumPy CPU or CuPy GPU)."""
        for name in list(self.params.keys()):
            self.params[name] = to_device(self.params[name])

    def forward(
        self,
        token_ids: Any,
        start_pos: int = 0,
        kv_caches: Optional[List[Optional[Tuple[Any, Any]]]] = None,
    ) -> Tuple[Any, dict, Optional[List[Tuple[Any, Any]]]]:
        """Forward pass of GPT model."""
        xp = get_backend()
        cfg = self.config
        token_ids = to_device(token_ids)

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape (batch, time), got {token_ids.shape}")
        b, t = token_ids.shape

        if start_pos + t > cfg.context_length:
            raise ValueError(f"Sequence length {start_pos + t} exceeds context_length {cfg.context_length}")

        # 1. Embeddings
        h = self.params["wte"][token_ids]  # (b, t, c)
        if cfg.pos_emb_type == "learned":
            pos = xp.arange(start_pos, start_pos + t)
            h = h + self.params["wpe"][pos][None, :, :]
        else:
            pos = None

        # 2. Transformer Blocks
        block_caches = []
        use_cache = kv_caches is not None
        new_kv_caches = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_cache_in = kv_caches[i] if (use_cache and i < len(kv_caches)) else None
            h, b_cache, layer_cache_out = block.forward(
                h,
                self.params,
                start_pos=start_pos,
                kv_cache=layer_cache_in,
                use_cache=use_cache,
            )
            block_caches.append(b_cache)
            if new_kv_caches is not None and layer_cache_out is not None:
                new_kv_caches.append(layer_cache_out)

        # 3. Final Normalization
        if cfg.norm_type == "rmsnorm":
            h_norm, ln_f_cache = rms_norm_forward(h, self.params["ln_f_g"], eps=cfg.eps)
        else:
            h_norm, ln_f_cache = layer_norm_forward(h, self.params["ln_f_g"], self.params["ln_f_b"], eps=cfg.eps)

        # 4. Projection to Vocab Logits
        if cfg.weight_tying:
            # Shared weights: lm_head_w = wte.T
            logits = h_norm @ self.params["wte"].T
        else:
            logits = h_norm @ self.params["lm_head_w"]

        if cfg.bias and "lm_head_b" in self.params:
            logits = logits + self.params["lm_head_b"]

        cache = {
            "token_ids": token_ids,
            "start_pos": start_pos,
            "h_norm": h_norm,
            "ln_f_cache": ln_f_cache,
            "blocks": block_caches,
        }
        return logits, cache, new_kv_caches

    def loss_and_gradients(
        self,
        token_ids: Any,
        targets: Any,
        target_mask: Optional[Any] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute cross-entropy loss and exact analytical gradients via backprop."""
        xp = get_backend()
        cfg = self.config
        token_ids = to_device(token_ids)
        targets = to_device(targets)
        if target_mask is not None:
            target_mask = to_device(target_mask)

        # Forward pass
        logits, cache, _ = self.forward(token_ids, start_pos=0)

        # Compute loss and dlogits
        loss, dlogits = cross_entropy_loss_and_grad(logits, targets, mask=target_mask)

        # Initialize gradients dict
        grads: Dict[str, Any] = {name: xp.zeros_like(param) for name, param in self.params.items()}

        b, t, v = logits.shape
        c = cfg.embedding_dim
        h_norm = cache["h_norm"]

        # 1. Backward through LM Head
        if cfg.weight_tying:
            # logits = h_norm @ wte.T
            # dwte = dlogits.T @ h_norm (which is added to wte grads)
            dwte_from_head = dlogits.reshape(-1, v).T @ h_norm.reshape(-1, c)
            grads["wte"] += dwte_from_head
            dh_norm = dlogits @ self.params["wte"]
        else:
            grads["lm_head_w"] = h_norm.reshape(-1, c).T @ dlogits.reshape(-1, v)
            dh_norm = dlogits @ self.params["lm_head_w"].T

        if cfg.bias and "lm_head_b" in self.params:
            grads["lm_head_b"] = dlogits.sum(axis=(0, 1))

        # 2. Backward through Final Norm
        if cfg.norm_type == "rmsnorm":
            dh, dgamma_f = rms_norm_backward(dh_norm, cache["ln_f_cache"])
            grads["ln_f_g"] = dgamma_f
        else:
            dh, dgamma_f, dbeta_f = layer_norm_backward(dh_norm, cache["ln_f_cache"])
            grads["ln_f_g"] = dgamma_f
            grads["ln_f_b"] = dbeta_f

        # 3. Backward through Transformer Blocks in reverse order
        for i in reversed(range(cfg.num_layers)):
            dh = self.blocks[i].backward(
                dh,
                cache["blocks"][i],
                self.params,
                grads,
            )

        # 4. Backward through Token & Positional Embeddings
        indexed_add(grads["wte"], token_ids, dh)
        if cfg.pos_emb_type == "learned":
            pos = xp.arange(t)
            indexed_add(grads["wpe"], pos, dh.sum(axis=0))

        return loss, grads

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 25,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_tokens: Optional[List[int]] = None,
        use_cache: bool = True,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Autoregressive text generation with KV caching, top-k, top-p, and repetition penalty."""
        xp = get_backend()
        rng = np_cpu.random.default_rng(seed)

        if not prompt_ids:
            prompt_ids = [0]

        generated = list(prompt_ids)
        stop_set = set(stop_tokens or [])

        # Initialize KV caches for each layer if caching enabled
        kv_caches: Optional[List[Optional[Tuple[Any, Any]]]] = (
            [None] * self.config.num_layers if use_cache else None
        )

        # 1. Prefill stage: Process initial prompt
        context_ids = xp.asarray([prompt_ids], dtype=xp.int64)
        logits, _, kv_caches = self.forward(context_ids, start_pos=0, kv_caches=kv_caches)
        next_logits = to_cpu(logits[0, -1]).astype(np_cpu.float64)

        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                recent_tokens = set(generated[-64:])
                for tok_id in recent_tokens:
                    if tok_id < len(next_logits):
                        if next_logits[tok_id] > 0:
                            next_logits[tok_id] /= repetition_penalty
                        else:
                            next_logits[tok_id] *= repetition_penalty

            # Apply temperature
            temp = max(temperature, 1e-5)
            next_logits = next_logits / temp

            # Apply Top-K filtering
            if 0 < top_k < len(next_logits):
                cutoff = np_cpu.partition(next_logits, -top_k)[-top_k]
                next_logits[next_logits < cutoff] = -np_cpu.inf

            # Softmax to probabilities
            shifted = next_logits - np_cpu.max(next_logits)
            exp_logits = np_cpu.exp(shifted)
            probs = exp_logits / np_cpu.sum(exp_logits)

            # Apply Top-P (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_indices = np_cpu.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumulative_probs = np_cpu.cumsum(sorted_probs)
                # Cut off tokens with cumulative probability above top_p
                cutoff_idx = np_cpu.searchsorted(cumulative_probs, top_p)
                valid_indices = sorted_indices[: max(1, cutoff_idx + 1)]
                filtered_probs = np_cpu.zeros_like(probs)
                filtered_probs[valid_indices] = probs[valid_indices]
                probs = filtered_probs / np_cpu.sum(filtered_probs)

            # Sample next token
            next_id = int(rng.choice(len(probs), p=probs))
            generated.append(next_id)

            if next_id in stop_set:
                break

            # Check context length limit
            if len(generated) >= self.config.context_length:
                break

            # Decode stage: Step forward with only 1 new token using KV Cache
            if use_cache and kv_caches is not None:
                step_ids = xp.asarray([[next_id]], dtype=xp.int64)
                start_pos = len(generated) - 1
                logits, _, kv_caches = self.forward(step_ids, start_pos=start_pos, kv_caches=kv_caches)
                next_logits = to_cpu(logits[0, -1]).astype(np_cpu.float64)
            else:
                context_ids = xp.asarray([generated[-self.config.context_length :]], dtype=xp.int64)
                logits, _, _ = self.forward(context_ids, start_pos=0)
                next_logits = to_cpu(logits[0, -1]).astype(np_cpu.float64)

        return generated

    def save(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights, config, and optimizer checkpoint."""
        ckpt = Path(checkpoint_path)
        cfg_p = Path(config_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {name: to_cpu(val) for name, val in self.params.items()}
        if optimizer_state is not None:
            for k, v in optimizer_state.items():
                save_dict[f"__opt_{k}__"] = to_cpu(v) if hasattr(v, "shape") else np_cpu.asarray(v)

        np_cpu.savez_compressed(ckpt, **save_dict)

        cfg_dict = self.config.to_dict()
        if metadata:
            cfg_dict["metadata"] = metadata
        cfg_p.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
    ) -> Tuple["GPT", Optional[Dict[str, Any]]]:
        """Load GPT model and optional optimizer state from disk."""
        ckpt = Path(checkpoint_path)
        cfg_p = Path(config_path)

        if not cfg_p.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_p}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint weights not found: {ckpt}")

        raw_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        config = GPTConfig.from_dict(raw_cfg)

        model = cls(config)
        loaded = np_cpu.load(ckpt)
        opt_state = {}

        for key in loaded.files:
            if key.startswith("__opt_"):
                opt_key = key[6:-2]
                opt_state[opt_key] = loaded[key]
            elif key in model.params:
                model.params[key][...] = to_device(np_cpu.asarray(loaded[key]))

        return model, (opt_state if opt_state else None)
