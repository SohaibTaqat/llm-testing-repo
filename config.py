"""
Shared configuration for LLM CLI tools.
"""

import os
from pathlib import Path

# Default model storage directory (RunPod workspace)
DEFAULT_MODEL_DIR = "/workspace/models"


def get_model_dir(cli_arg: str | None = None) -> Path:
    """
    Get model directory with priority:
    1. CLI argument (--model-dir)
    2. Environment variable (LLM_MODEL_DIR)
    3. Default path
    """
    if cli_arg:
        return Path(cli_arg)

    env_dir = os.environ.get("LLM_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    return Path(DEFAULT_MODEL_DIR)


def get_model_path(model_name: str, model_dir: Path) -> Path:
    """
    Build full path for a model: {model_dir}/{org}/{model_name}/

    Args:
        model_name: HuggingFace model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        model_dir: Base directory for models

    Returns:
        Full path to model directory
    """
    return model_dir / model_name


def model_exists(model_name: str, model_dir: Path) -> bool:
    """Check if a model has been downloaded."""
    model_path = get_model_path(model_name, model_dir)
    if not model_path.exists():
        return False

    # Check for essential files (config.json or model files)
    has_config = (model_path / "config.json").exists()
    has_safetensors = any(model_path.glob("*.safetensors"))
    has_bin = any(model_path.glob("*.bin"))

    return has_config and (has_safetensors or has_bin)


def get_gpu_info() -> list[dict]:
    """
    Get information about available GPUs.

    Returns:
        List of dicts with 'index', 'name', 'total_memory_gb' keys
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3),
            })
        return gpus
    except ImportError:
        return []


def get_vram_usage() -> list[dict]:
    """
    Get current VRAM usage per GPU.

    Returns:
        List of dicts with 'index', 'allocated_gb', 'total_gb', 'percent' keys
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        usage = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            usage.append({
                "index": i,
                "allocated_gb": allocated,
                "total_gb": total,
                "percent": (allocated / total) * 100 if total > 0 else 0,
            })
        return usage
    except ImportError:
        return []


def print_gpu_info():
    """Print formatted GPU information."""
    gpus = get_gpu_info()

    if not gpus:
        print("No CUDA GPUs detected")
        return

    print(f"\n{'='*60}")
    print("GPU Information")
    print(f"{'='*60}")
    for gpu in gpus:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"          VRAM: {gpu['total_memory_gb']:.1f} GB")
    print(f"{'='*60}\n")


def print_vram_usage():
    """Print formatted VRAM usage."""
    usage = get_vram_usage()

    if not usage:
        print("No CUDA GPUs detected")
        return

    print(f"\n{'='*60}")
    print("VRAM Usage")
    print(f"{'='*60}")
    for gpu in usage:
        bar_width = 30
        filled = int(bar_width * gpu['percent'] / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"  GPU {gpu['index']}: [{bar}] {gpu['allocated_gb']:.2f}/{gpu['total_gb']:.1f} GB ({gpu['percent']:.1f}%)")
    print(f"{'='*60}\n")
