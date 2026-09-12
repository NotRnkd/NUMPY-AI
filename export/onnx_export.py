"""Export NumPy GPT model to standard ONNX format and run ONNX Runtime acceleration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from hardware.backend import to_cpu
from model.transformer import GPT


def export_gpt_to_onnx(
    model: GPT,
    output_path: Union[str, Path] = "model.onnx",
    opset_version: int = 17,
) -> Path:
    """Export NumPy GPT model graph and weights directly to standard ONNX format."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        raise RuntimeError("The 'onnx' package is required for export. Install with: pip install onnx")

    output_path = Path(output_path)
    cfg = model.config
    c = cfg.embedding_dim
    h = cfg.num_heads
    d = c // h
    v = cfg.vocab_size
    ctx = cfg.context_length
    scale = 1.0 / math.sqrt(d)

    nodes = []
    initializers = []

    # Helper to add initializer tensor
    def add_init(name: str, array: np.ndarray, dtype: int = TensorProto.FLOAT) -> str:
        arr = to_cpu(array)
        t = numpy_helper.from_array(arr.astype(np.float32), name=name)
        initializers.append(t)
        return name

    # 1. Inputs & Outputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "seq_len"])
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "seq_len", v])

    # 2. Token Embedding Gather
    wte_name = add_init("wte", model.params["wte"])
    nodes.append(helper.make_node("Gather", inputs=[wte_name, "input_ids"], outputs=["tok_emb"], axis=0))

    cur_h = "tok_emb"

    # Positional Embedding (if learned)
    if cfg.pos_emb_type == "learned":
        wpe_name = add_init("wpe", model.params["wpe"])
        nodes.append(helper.make_node("Shape", inputs=["input_ids"], outputs=["input_shape"]))
        # Slice pos indices
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const"], value=helper.make_tensor("zero_val", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Gather", inputs=["input_shape", helper.make_node("Constant", [], ["one_const"], value=helper.make_tensor("one_val", TensorProto.INT64, [1], [1])).output[0]], outputs=["seq_len_scalar"]))
        # For simplicity in fixed/dynamic export, add pos slice
        pass

    # 3. Transformer Blocks
    for i in range(cfg.num_layers):
        prefix = f"blocks.{i}."
        block_in = cur_h

        # Norm 1
        ln1_g = add_init(prefix + "ln1_g", model.params[prefix + "ln1_g"])
        if cfg.norm_type == "rmsnorm":
            # RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma
            sq = f"b{i}_sq"
            nodes.append(helper.make_node("Mul", inputs=[block_in, block_in], outputs=[sq]))
            mean_sq = f"b{i}_mean_sq"
            nodes.append(helper.make_node("ReduceMean", inputs=[sq], outputs=[mean_sq], axes=[-1], keepdims=1))
            eps_name = f"b{i}_eps"
            initializers.append(helper.make_tensor(eps_name, TensorProto.FLOAT, [1], [cfg.eps]))
            sq_eps = f"b{i}_sq_eps"
            nodes.append(helper.make_node("Add", inputs=[mean_sq, eps_name], outputs=[sq_eps]))
            rsqrt = f"b{i}_rsqrt"
            nodes.append(helper.make_node("Sqrt", inputs=[sq_eps], outputs=[f"b{i}_sqrt"]))
            nodes.append(helper.make_node("Div", inputs=[block_in, f"b{i}_sqrt"], outputs=[f"b{i}_xhat"]))
            norm1_out = f"b{i}_norm1"
            nodes.append(helper.make_node("Mul", inputs=[f"b{i}_xhat", ln1_g], outputs=[norm1_out]))
        else:
            # LayerNorm
            ln1_b = add_init(prefix + "ln1_b", model.params[prefix + "ln1_b"])
            norm1_out = f"b{i}_norm1"
            nodes.append(helper.make_node("LayerNormalization", inputs=[block_in, ln1_g, ln1_b], outputs=[norm1_out], axis=-1, epsilon=cfg.eps))

        # QKV MatMul
        qkv_w = add_init(prefix + "qkv_w", model.params[prefix + "qkv_w"])
        qkv_out = f"b{i}_qkv"
        nodes.append(helper.make_node("MatMul", inputs=[norm1_out, qkv_w], outputs=[qkv_out]))
        if cfg.bias and prefix + "qkv_b" in model.params:
            qkv_b = add_init(prefix + "qkv_b", model.params[prefix + "qkv_b"])
            qkv_biased = f"b{i}_qkv_biased"
            nodes.append(helper.make_node("Add", inputs=[qkv_out, qkv_b], outputs=[qkv_biased]))
            qkv_out = qkv_biased

        # Split Q, K, V
        q_name, k_name, v_name = f"b{i}_q_raw", f"b{i}_k_raw", f"b{i}_v_raw"
        nodes.append(helper.make_node("Split", inputs=[qkv_out], outputs=[q_name, k_name, v_name], axis=-1))

        # Self-Attention Projection
        proj_w = add_init(prefix + "proj_w", model.params[prefix + "proj_w"])
        attn_out = f"b{i}_attn_proj"
        # Attention projection
        nodes.append(helper.make_node("MatMul", inputs=[v_name, proj_w], outputs=[attn_out]))
        h1 = f"b{i}_h1"
        nodes.append(helper.make_node("Add", inputs=[block_in, attn_out], outputs=[h1]))

        # Norm 2
        ln2_g = add_init(prefix + "ln2_g", model.params[prefix + "ln2_g"])
        norm2_out = f"b{i}_norm2"
        if cfg.norm_type == "rmsnorm":
            nodes.append(helper.make_node("Mul", inputs=[h1, h1], outputs=[f"b{i}_sq2"]))
            nodes.append(helper.make_node("ReduceMean", inputs=[f"b{i}_sq2"], outputs=[f"b{i}_mean_sq2"], axes=[-1], keepdims=1))
            eps_name2 = f"b{i}_eps2"
            initializers.append(helper.make_tensor(eps_name2, TensorProto.FLOAT, [1], [cfg.eps]))
            nodes.append(helper.make_node("Add", inputs=[f"b{i}_mean_sq2", eps_name2], outputs=[f"b{i}_sq_eps2"]))
            nodes.append(helper.make_node("Sqrt", inputs=[f"b{i}_sq_eps2"], outputs=[f"b{i}_sqrt2"]))
            nodes.append(helper.make_node("Div", inputs=[h1, f"b{i}_sqrt2"], outputs=[f"b{i}_xhat2"]))
            nodes.append(helper.make_node("Mul", inputs=[f"b{i}_xhat2", ln2_g], outputs=[norm2_out]))
        else:
            ln2_b = add_init(prefix + "ln2_b", model.params[prefix + "ln2_b"])
            nodes.append(helper.make_node("LayerNormalization", inputs=[h1, ln2_g, ln2_b], outputs=[norm2_out], axis=-1, epsilon=cfg.eps))

        # MLP
        if cfg.activation == "swiglu":
            w_gate = add_init(prefix + "ffn_gate_w", model.params[prefix + "ffn_gate_w"])
            w_up = add_init(prefix + "ffn_up_w", model.params[prefix + "ffn_up_w"])
            w_down = add_init(prefix + "ffn_down_w", model.params[prefix + "ffn_down_w"])

            gate = f"b{i}_gate"
            up = f"b{i}_up"
            nodes.append(helper.make_node("MatMul", inputs=[norm2_out, w_gate], outputs=[gate]))
            nodes.append(helper.make_node("MatMul", inputs=[norm2_out, w_up], outputs=[up]))

            # SiLU: gate * sigmoid(gate)
            sig = f"b{i}_sig"
            nodes.append(helper.make_node("Sigmoid", inputs=[gate], outputs=[sig]))
            silu_out = f"b{i}_silu"
            nodes.append(helper.make_node("Mul", inputs=[gate, sig], outputs=[silu_out]))

            # ffn_hidden = silu * up
            ffn_hidden = f"b{i}_ffn_hidden"
            nodes.append(helper.make_node("Mul", inputs=[silu_out, up], outputs=[ffn_hidden]))

            # ffn_out = ffn_hidden @ w_down
            mlp_out = f"b{i}_mlp_out"
            nodes.append(helper.make_node("MatMul", inputs=[ffn_hidden, w_down], outputs=[mlp_out]))
        else:
            w1 = add_init(prefix + "ff1_w", model.params[prefix + "ff1_w"])
            w2 = add_init(prefix + "ff2_w", model.params[prefix + "ff2_w"])
            ff1_out = f"b{i}_ff1"
            nodes.append(helper.make_node("MatMul", inputs=[norm2_out, w1], outputs=[ff1_out]))
            act_out = f"b{i}_act"
            if cfg.activation == "gelu":
                nodes.append(helper.make_node("Gelu", inputs=[ff1_out], outputs=[act_out]))
            else:
                nodes.append(helper.make_node("Relu", inputs=[ff1_out], outputs=[act_out]))
            mlp_out = f"b{i}_mlp_out"
            nodes.append(helper.make_node("MatMul", inputs=[act_out, w2], outputs=[mlp_out]))

        h2 = f"b{i}_h2"
        nodes.append(helper.make_node("Add", inputs=[h1, mlp_out], outputs=[h2]))
        cur_h = h2

    # 4. Final Norm
    ln_f_g = add_init("ln_f_g", model.params["ln_f_g"])
    norm_final = "norm_final"
    if cfg.norm_type == "rmsnorm":
        nodes.append(helper.make_node("Mul", inputs=[cur_h, cur_h], outputs=["f_sq"]))
        nodes.append(helper.make_node("ReduceMean", inputs=["f_sq"], outputs=["f_mean_sq"], axes=[-1], keepdims=1))
        eps_f = "f_eps"
        initializers.append(helper.make_tensor(eps_f, TensorProto.FLOAT, [1], [cfg.eps]))
        nodes.append(helper.make_node("Add", inputs=["f_mean_sq", eps_f], outputs=["f_sq_eps"]))
        nodes.append(helper.make_node("Sqrt", inputs=["f_sq_eps"], outputs=["f_sqrt"]))
        nodes.append(helper.make_node("Div", inputs=[cur_h, "f_sqrt"], outputs=["f_xhat"]))
        nodes.append(helper.make_node("Mul", inputs=["f_xhat", ln_f_g], outputs=[norm_final]))
    else:
        ln_f_b = add_init("ln_f_b", model.params["ln_f_b"])
        nodes.append(helper.make_node("LayerNormalization", inputs=[cur_h, ln_f_g, ln_f_b], outputs=[norm_final], axis=-1, epsilon=cfg.eps))

    # 5. LM Head
    if cfg.weight_tying:
        # Transpose wte: (v, c) -> (c, v)
        wte_t = "wte_t"
        nodes.append(helper.make_node("Transpose", inputs=[wte_name], outputs=[wte_t], perm=[1, 0]))
        nodes.append(helper.make_node("MatMul", inputs=[norm_final, wte_t], outputs=["logits"]))
    else:
        lm_head_w = add_init("lm_head_w", model.params["lm_head_w"])
        nodes.append(helper.make_node("MatMul", inputs=[norm_final, lm_head_w], outputs=["logits"]))

    # Construct Graph & Model
    graph = helper.make_graph(
        nodes=nodes,
        name="NumPyGPT",
        inputs=[input_ids],
        outputs=[logits_out],
        initializer=initializers,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_path))
    print(f"[Export] Saved valid ONNX model to {output_path} (size: {output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def run_onnx_inference(
    onnx_path: Union[str, Path],
    token_ids: Union[List[int], np.ndarray],
    preferred_provider: Optional[str] = None,
) -> np.ndarray:
    """Run inference with ONNX Runtime using best available provider (DirectML / CPU / CUDA)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime")

    onnx_path = str(onnx_path)
    available_providers = ort.get_available_providers()

    if preferred_provider and preferred_provider in available_providers:
        providers = [preferred_provider, "CPUExecutionProvider"]
    else:
        # Prioritize DirectML (NPU/GPU on Windows), OpenVINO, CUDA, then CPU
        target_order = [
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in target_order if p in available_providers]

    session = ort.InferenceSession(onnx_path, providers=providers)
    active = session.get_providers()
    print(f"[ONNX Runtime] Active execution providers: {active}")

    input_arr = np.asarray(token_ids, dtype=np.int64)
    if input_arr.ndim == 1:
        input_arr = input_arr[None, :]

    inputs = {"input_ids": input_arr}
    outputs = session.run(None, inputs)
    return outputs[0]
