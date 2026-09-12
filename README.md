# NumPy-GPT: Modern, Offline Neural Language Model in Pure NumPy / CuPy

NumPy-GPT is a production-grade, decoder-only Transformer built from scratch using pure NumPy (for universal CPU execution) and CuPy (for optional CUDA GPU acceleration), without PyTorch, TensorFlow, JAX, or any external cloud APIs.

It runs 100% locally and offline on your computer.

---

## What's New & Architecture Upgrades

The neural network "brain" has been redesigned with modern LLM architectural standards:

1. **Rotary Positional Embeddings (RoPE):**
   - Relative positional encoding injected directly into Query and Key states at every layer with inverse-frequency rotators, replacing rigid absolute learned embeddings.
2. **RMSNorm (Root Mean Square Normalization):**
   - Shift-invariant normalization replacing LayerNorm for faster execution and numerically stable training without mean-centering overhead.
3. **SwiGLU Feed-Forward Network:**
   - Swish-Gated Linear Units ($\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$) yielding higher sample efficiency and representation capacity than traditional single-matrix ReLU/GELU layers.
4. **Weight Tying:**
   - Sharing input embedding `wte` and output projection `lm_head` weights, cutting memory footprint by 30–45% while regularizing language representations.
5. **Linear-Time KV-Caching ($O(T)$ Decoding):**
   - Layer-wise Key-Value state cache during autoregressive generation, eliminating redundant recalculation of past tokens and accelerating generation by 2.5x–5x.
6. **Decoupled AdamW & Gradient Clipping:**
   - Proper weight decay applied exclusively to 2D weight matrices (excluding 1D biases and normalization gains), paired with global L2 gradient clipping and cosine learning-rate scheduling with linear warmup.
7. **Instruction Tuning with Loss Masking:**
   - Supervised conversation dataset training where gradients only backpropagate through assistant responses, masking system prompts and user inputs.
8. **Hardware Diagnostics & DirectML / ONNX Export:**
   - Native export to `.onnx` for local execution via ONNX Runtime using DirectML (NPU/AMD/Intel/NVIDIA GPU on Windows) or OpenVINO.

---

## Modular Architecture Overview

```
├── hardware/              # Hardware detection and NumPy / CuPy abstraction
│   ├── backend.py         # Dynamic CPU/GPU device selection and array helpers
│   └── detection.py       # SIMD, AVX, CUDA, VRAM, and NPU inspection
├── tokenizer/             # Enhanced Byte-Pair Encoding (BPE)
│   ├── bpe.py             # Byte-level BPE with special tokens, caching, regex pre-tokenization
│   └── train_tokenizer.py # Standalone corpus tokenizer trainer
├── model/                 # Neural network architecture & autograd
│   ├── config.py          # GPTConfig dataclass with presets (tiny, small, medium, large)
│   ├── layers.py          # RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, and CrossEntropy
│   ├── attention.py       # Causal Self-Attention with KV cache and RoPE
│   └── transformer.py     # Decoder-only Transformer, manual backprop, forward, generation
├── training/              # Optimization and training orchestration
│   ├── optimizer.py       # Decoupled AdamW with 1D parameter exclusion and gradient clipping
│   ├── scheduler.py       # Linear warmup and cosine decay scheduler
│   ├── dataset.py         # Text chunking and instruction-masked chat datasets
│   └── trainer.py         # Training loop, gradient accumulation, validation, checkpoints
├── inference/             # Inference and chat runtime
│   ├── cache.py           # KV Cache structure
│   ├── generate.py        # Streaming generator with temperature, top-k, top-p, repetition penalty
│   └── chat.py            # Multi-turn conversational REPL with system prompt preservation
├── evaluation/            # Benchmarks and validation
│   └── evaluate.py        # Perplexity evaluation, generation speed benchmarks, repetition metrics
├── export/                # Hardware acceleration export
│   └── onnx_export.py     # Pure ONNX exporter and ONNX Runtime runner
├── configs/               # Model configuration presets
│   ├── tiny.json          # Fast CPU / laptop (<1.5M params)
│   ├── small.json         # Balanced desktop CPU / entry GPU (~6M params)
│   └── medium.json        # High-capacity GPU (~18M params)
└── numpy_gpt.py           # Unified CLI
```

---

## Quick Start: One-Click Launcher

### Windows (.bat)
Simply double-click **`launch.bat`** in the project folder (or run it from cmd / PowerShell):
```cmd
launch.bat
```
This automatically:
1. Verifies Python 3.9+ and installs `numpy` if not present
2. Verifies Node.js and installs frontend dependencies on first run
3. Starts the background Python inference & training daemon (port 5005)
4. Starts the web studio on `http://localhost:3000`
5. Opens your default web browser directly to the studio

### Linux / macOS (.sh)
```bash
./launch.sh
```

---

## Installation & Setup

Requirements: Python 3.9+ with `numpy`.

```powershell
pip install numpy
```

### Optional GPU Acceleration (NVIDIA CUDA)
For NVIDIA GPU acceleration via CuPy:
```powershell
pip install cupy-cuda12x  # For CUDA 12
# or
pip install "cupy-cuda13x[ctk]"  # For CUDA 13
```

### Optional NPU / DirectML Acceleration
For ONNX Runtime execution:
```powershell
pip install onnx onnxruntime
# On Windows with DirectML (NPU / GPU):
pip install onnxruntime-directml
```

---

## CLI Usage Guide

### 1. Hardware Inspection
Inspect your system's CPU cores, SIMD vector instructions, CUDA GPU, and NPU availability:
```bash
python numpy_gpt.py hardware
```

### 2. Pre-Training a Model
Train a model from scratch on any text file:
```bash
python numpy_gpt.py train --data data.txt --preset tiny --steps 1000 --batch-size 4 --grad-accum 2 --lr 5e-4
```
For custom model configurations:
```bash
python numpy_gpt.py train --data english.txt --dim 256 --heads 8 --layers 6 --context-length 256 --steps 5000 --device auto
```

### 3. Instruction Tuning (Chat Training)
Train on conversation datasets in JSON format with automatic prompt loss masking:
```bash
python numpy_gpt.py train --data english_large_gpu_chat_datasets.json --checkpoint chat_model.npz --steps 2000
```

### 4. Interactive Terminal Chat
Launch a multi-turn conversation in your terminal:
```bash
python numpy_gpt.py chat --checkpoint checkpoint.npz
```
*Special chat commands:*
- `/reset` - Clear conversation history while preserving the system prompt
- `/system <text>` - Set a new system prompt
- `/history` - View previous conversation turns
- `/exit` - Exit chat

### 5. Autoregressive Text Generation
Generate text from a prompt with real-time token streaming:
```bash
python numpy_gpt.py generate --checkpoint checkpoint.npz --prompt "Machine learning is" --temperature 0.7 --top-k 40 --top-p 0.9
```

### 6. Perplexity & Instruction Evaluation
Evaluate cross-entropy loss, perplexity, and KV cache speedup:
```bash
python numpy_gpt.py evaluate --checkpoint checkpoint.npz --data data.txt
```

### 7. Hardware Benchmark
Benchmark your machine's matrix multiplication throughput and token processing speed:
```bash
python numpy_gpt.py benchmark --dim 128 --heads 4 --layers 4 --batch-size 4
```

### 8. ONNX Export for NPU / DirectML
Export your trained weights to standard ONNX:
```bash
python numpy_gpt.py export --checkpoint checkpoint.npz --output model.onnx --test
```

---

## Running Unit Tests

Run all unit tests verifying the tokenizer, RoPE, RMSNorm, SwiGLU, KV caching, AdamW, and ONNX export:
```bash
python -m pytest tests/
```
