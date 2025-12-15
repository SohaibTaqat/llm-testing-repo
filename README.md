# LLM CLI Tools

A Python CLI toolkit for downloading, running, and benchmarking LLM models from HuggingFace. Optimized for multi-GPU setups with RTX PRO 6000 Blackwell GPUs on RunPod.

## Features

- **Download models** from HuggingFace with progress tracking
- **Interactive chat** with downloaded models
- **GPU benchmarking** with detailed performance metrics
- **Multi-GPU support** via accelerate's automatic device mapping
- **Quantization** support (4-bit and 8-bit) via bitsandbytes
- **Pre-quantized models** support (GPTQ/AWQ) - auto-detected
- **Safetensors** preference for faster model loading

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU(s)
- PyTorch with CUDA support (pre-installed on RunPod)
- HuggingFace account (for gated models like Llama)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-testing-repo

# create a venv and use existing downloaded dependencies
python -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set your HuggingFace token for gated models
export HF_TOKEN=your_token_here
```

## Quick Start

```bash
# 1. Download a model
python download.py mistralai/Mistral-7B-Instruct-v0.3

# 2. Chat with it
python chat.py mistralai/Mistral-7B-Instruct-v0.3

# 3. Benchmark performance
python benchmark.py mistralai/Mistral-7B-Instruct-v0.3
```

## Usage

### download.py

Download models from HuggingFace Hub.

```bash
# Download a model
python download.py mistralai/Mistral-7B-Instruct-v0.3

# Download a gated model (requires HF_TOKEN)
python download.py meta-llama/Llama-2-7b-chat-hf

# List downloaded models
python download.py --list

# Force redownload
python download.py mistralai/Mistral-7B-Instruct-v0.3 --force

# Use custom model directory
python download.py mistralai/Mistral-7B-Instruct-v0.3 --model-dir /custom/path
```

### chat.py

Interactive chat with downloaded models.

```bash
# Basic chat
python chat.py mistralai/Mistral-7B-Instruct-v0.3

# With 4-bit quantization (saves VRAM)
python chat.py mistralai/Mistral-7B-Instruct-v0.3 --4bit

# With 8-bit quantization
python chat.py mistralai/Mistral-7B-Instruct-v0.3 --8bit
```

**Chat commands:**
- `/clear` - Reset conversation history
- `/quit` - Exit chat

### benchmark.py

Benchmark GPU performance on models.

```bash
# Run benchmark
python benchmark.py mistralai/Mistral-7B-Instruct-v0.3

# Benchmark with quantization
python benchmark.py mistralai/Mistral-7B-Instruct-v0.3 --4bit

# Save results to JSON
python benchmark.py mistralai/Mistral-7B-Instruct-v0.3 --output results.json
```

**Metrics measured:**
- Tokens per second (generation throughput)
- Time to first token (TTFT / latency)
- Total inference time
- VRAM usage per GPU (current and peak)

**Results are auto-saved** to the `results/` folder:
- Filename format: `{org}_{model}_{quantization}_{timestamp}.txt`
- Example: `results/mistralai_Mistral-7B-Instruct-v0.3_fp16_20241215_143022.txt`

## Configuration

### Model Storage

Default model directory: `/workspace/models/`

Override with:
1. `--model-dir` flag (highest priority)
2. `LLM_MODEL_DIR` environment variable
3. Default path

```bash
# Using environment variable
export LLM_MODEL_DIR=/my/models
python chat.py mistralai/Mistral-7B-Instruct-v0.3

# Using flag
python chat.py mistralai/Mistral-7B-Instruct-v0.3 --model-dir /my/models
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_MODEL_DIR` | Override default model storage directory |
| `HF_TOKEN` | HuggingFace token for gated models |

## Recommended Models

| Model | Size | Notes |
|-------|------|-------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Fast, good quality |
| `meta-llama/Llama-2-7b-chat-hf` | 7B | Requires HF token |
| `meta-llama/Llama-2-13b-chat-hf` | 13B | Better quality, needs more VRAM |
| `google/gemma-7b-it` | 7B | Good instruction following |
| `Qwen/Qwen2-7B-Instruct` | 7B | Multilingual |

### Pre-quantized Models (GPTQ/AWQ)

Pre-quantized models are already compressed and load faster with lower VRAM usage. No `--4bit` or `--8bit` flags needed - transformers auto-detects the format.

```bash
# Download and use AWQ model
python download.py TheBloke/Mistral-7B-Instruct-v0.2-AWQ
python chat.py TheBloke/Mistral-7B-Instruct-v0.2-AWQ

# Download and use GPTQ model
python download.py TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
python chat.py TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
```

Popular pre-quantized models from TheBloke:
| Model | Format | Notes |
|-------|--------|-------|
| `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | AWQ | Fast inference |
| `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` | GPTQ | Good quality |
| `TheBloke/Llama-2-13B-chat-AWQ` | AWQ | Larger model, pre-quantized |
| `TheBloke/CodeLlama-34B-Instruct-AWQ` | AWQ | Code generation |

## Example Workflow

```bash
# Start fresh on RunPod
cd /workspace
git clone <repo-url> llm-tools
cd llm-tools
pip install -r requirements.txt

# Set HF token if using gated models
export HF_TOKEN=hf_xxxxx

# Download models
python download.py mistralai/Mistral-7B-Instruct-v0.3
python download.py meta-llama/Llama-2-13b-chat-hf

# Check what's downloaded
python download.py --list

# Benchmark both
python benchmark.py mistralai/Mistral-7B-Instruct-v0.3 --output mistral-7b.json
python benchmark.py meta-llama/Llama-2-13b-chat-hf --4bit --output llama-13b-4bit.json

# Chat with a model
python chat.py mistralai/Mistral-7B-Instruct-v0.3
```

## Troubleshooting

### "No CUDA GPUs available"

This tool requires GPU acceleration. Ensure:
- You're running on a GPU pod
- CUDA drivers are installed
- PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### "Model not found"

Run `download.py` first:
```bash
python download.py <model-name>
```

### "GatedRepoError"

For gated models (Llama, etc.):
1. Accept the license on HuggingFace model page
2. Create a token at https://huggingface.co/settings/tokens
3. Set it: `export HF_TOKEN=your_token`

### Out of VRAM

Try quantization:
```bash
python chat.py <model> --4bit  # Most VRAM savings
python chat.py <model> --8bit  # Moderate savings
```

## File Structure

```
llm-testing-repo/
├── config.py          # Shared configuration utilities
├── download.py        # Model download CLI
├── chat.py            # Interactive chat CLI
├── benchmark.py       # GPU benchmarking CLI
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── results/           # Auto-saved benchmark results (created on first run)
```
