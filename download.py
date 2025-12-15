#!/usr/bin/env python3
"""
Download LLM models from HuggingFace with progress tracking.

Usage:
    python download.py mistralai/Mistral-7B-Instruct-v0.3
    python download.py --list
    python download.py meta-llama/Llama-2-7b-chat-hf --force
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from config import get_model_dir, get_model_path, model_exists


def list_models(model_dir: Path) -> None:
    """List all downloaded models."""
    if not model_dir.exists():
        print(f"Model directory does not exist: {model_dir}")
        return

    print(f"\nDownloaded models in {model_dir}:")
    print("=" * 60)

    found_any = False

    # Walk through org directories
    for org_path in sorted(model_dir.iterdir()):
        if not org_path.is_dir():
            continue

        # Walk through model directories
        for model_path in sorted(org_path.iterdir()):
            if not model_path.is_dir():
                continue

            # Check if it looks like a valid model
            has_config = (model_path / "config.json").exists()
            has_safetensors = any(model_path.glob("*.safetensors"))
            has_bin = any(model_path.glob("*.bin"))

            if has_config and (has_safetensors or has_bin):
                model_name = f"{org_path.name}/{model_path.name}"

                # Calculate size
                total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                size_gb = total_size / (1024 ** 3)

                format_type = "safetensors" if has_safetensors else "bin"
                print(f"  {model_name}")
                print(f"      Size: {size_gb:.2f} GB | Format: {format_type}")
                found_any = True

    if not found_any:
        print("  No models found")

    print("=" * 60)


def download_model(model_name: str, model_dir: Path, force: bool = False) -> bool:
    """
    Download a model from HuggingFace.

    Args:
        model_name: HuggingFace model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        model_dir: Directory to save models
        force: Force redownload even if exists

    Returns:
        True if successful, False otherwise
    """
    local_path = get_model_path(model_name, model_dir)

    # Check if already exists
    if model_exists(model_name, model_dir) and not force:
        print(f"\nModel already downloaded: {local_path}")
        print("Use --force to redownload")
        return True

    print(f"\nDownloading: {model_name}")
    print(f"Destination: {local_path}")
    print("-" * 60)

    # Check for HF token for gated models
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("Using HuggingFace token from environment")

    try:
        # Try to download safetensors first, fall back to bin if not available
        # First, check what formats are available
        api = HfApi()
        try:
            files = api.list_repo_files(model_name, token=hf_token)
            has_safetensors = any(f.endswith(".safetensors") for f in files)
        except Exception:
            has_safetensors = False

        # Set ignore patterns based on available formats
        ignore_patterns = []
        if has_safetensors:
            # Prefer safetensors, ignore bin files
            ignore_patterns = ["*.bin", "*.msgpack", "*.h5"]
            print("Downloading safetensors format (faster loading)")
        else:
            ignore_patterns = ["*.msgpack", "*.h5"]
            print("Downloading bin format (safetensors not available)")

        # Download with progress
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns=ignore_patterns,
        )

        print("-" * 60)
        print(f"Download complete: {local_path}")
        return True

    except GatedRepoError:
        print("\nError: This is a gated model requiring authentication.")
        print("Please:")
        print("  1. Accept the model license on HuggingFace")
        print("  2. Set your token: export HF_TOKEN=your_token_here")
        print("  3. Get token from: https://huggingface.co/settings/tokens")
        return False

    except RepositoryNotFoundError:
        print(f"\nError: Model not found: {model_name}")
        print("Please check the model name and try again.")
        print("Browse models at: https://huggingface.co/models")
        return False

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        return False

    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download LLM models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download.py mistralai/Mistral-7B-Instruct-v0.3
  python download.py meta-llama/Llama-2-7b-chat-hf
  python download.py --list
  python download.py google/gemma-7b --force

Environment variables:
  LLM_MODEL_DIR   Override default model directory
  HF_TOKEN        HuggingFace token for gated models
        """,
    )

    parser.add_argument(
        "model_name",
        nargs="?",
        help="HuggingFace model ID (e.g., mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--model-dir",
        help="Override model storage directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List downloaded models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if model exists",
    )

    args = parser.parse_args()
    model_dir = get_model_dir(args.model_dir)

    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models(model_dir)
        return

    if not args.model_name:
        parser.error("Model name required (or use --list)")

    # Validate model name format
    if "/" not in args.model_name:
        print(f"Error: Invalid model name format: {args.model_name}")
        print("Expected format: org/model (e.g., mistralai/Mistral-7B-Instruct-v0.3)")
        sys.exit(1)

    success = download_model(args.model_name, model_dir, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
