#!/usr/bin/env python3
"""
Benchmark GPU performance on LLM models.

Usage:
    python benchmark.py mistralai/Mistral-7B-Instruct-v0.3
    python benchmark.py meta-llama/Llama-2-7b-chat-hf --4bit
    python benchmark.py google/gemma-7b --output results.json
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    get_model_dir,
    get_model_path,
    model_exists,
    get_gpu_info,
    get_vram_usage,
    print_gpu_info,
    print_vram_usage,
)


# Test prompts of varying lengths
TEST_PROMPTS = [
    {
        "name": "short",
        "prompt": "Explain what Python is in one sentence.",
        "max_tokens": 64,
    },
    {
        "name": "medium",
        "prompt": "Write a detailed explanation of how neural networks learn, including the concepts of forward propagation, backpropagation, and gradient descent. Explain how weights are updated during training.",
        "max_tokens": 256,
    },
    {
        "name": "long",
        "prompt": "Write a comprehensive guide to building REST APIs with Python. Cover the following topics: choosing a framework (Flask vs FastAPI), designing endpoints, handling authentication, input validation, error handling, database integration, and testing. Provide code examples for each section.",
        "max_tokens": 512,
    },
]

# Number of runs per prompt for averaging
NUM_RUNS = 3


def get_quantization_config(use_4bit: bool, use_8bit: bool) -> BitsAndBytesConfig | None:
    """Get quantization config based on flags."""
    if use_4bit and use_8bit:
        print("Error: Cannot use both --4bit and --8bit")
        sys.exit(1)

    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def load_model(model_path: str, quantization_config: BitsAndBytesConfig | None):
    """
    Load model and tokenizer.

    Raises:
        RuntimeError: If no CUDA GPUs are available
    """
    # Ensure GPU is available - never fall back to CPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPUs available. This tool requires GPU acceleration.\n"
            "CPU fallback is disabled to prevent pod freezes with large models."
        )

    print(f"Loading model from: {model_path}")

    if quantization_config:
        if quantization_config.load_in_4bit:
            print("Using 4-bit quantization")
        elif quantization_config.load_in_8bit:
            print("Using 8-bit quantization")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    return model, tokenizer


def measure_generation(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """
    Measure generation metrics for a single prompt.

    Returns:
        Dict with timing and token metrics
    """
    # Prepare input
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = f"User: {prompt}\nAssistant: "

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # Warmup CUDA
    torch.cuda.synchronize()

    # Measure time to first token using a streaming approach
    first_token_time = None
    start_time = time.perf_counter()

    with torch.no_grad():
        # Generate with timing
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Calculate metrics
    output_length = outputs.shape[1] - input_length
    total_time = end_time - start_time

    # Estimate TTFT (approximation - first token usually takes longest due to prefill)
    # In practice, TTFT = prefill_time, which we estimate as proportional to input length
    estimated_ttft = total_time * (input_length / (input_length + output_length))

    tokens_per_second = output_length / total_time if total_time > 0 else 0

    return {
        "input_tokens": input_length,
        "output_tokens": output_length,
        "total_time_s": total_time,
        "ttft_s": estimated_ttft,
        "tokens_per_second": tokens_per_second,
    }


def get_peak_vram() -> list[dict]:
    """Get peak VRAM usage per GPU."""
    usage = []
    for i in range(torch.cuda.device_count()):
        peak = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usage.append({
            "index": i,
            "peak_gb": peak,
            "total_gb": total,
            "percent": (peak / total) * 100 if total > 0 else 0,
        })
    return usage


def benchmark_model(model, tokenizer, model_name: str, quantization: str) -> dict:
    """
    Run full benchmark suite on a model.

    Returns:
        Dict with all benchmark results
    """
    results = {
        "model": model_name,
        "quantization": quantization,
        "timestamp": datetime.now().isoformat(),
        "gpu_info": get_gpu_info(),
        "prompts": [],
    }

    print("\n" + "=" * 70)
    print("Running Benchmarks")
    print("=" * 70)

    for prompt_info in TEST_PROMPTS:
        prompt_name = prompt_info["name"]
        prompt_text = prompt_info["prompt"]
        max_tokens = prompt_info["max_tokens"]

        print(f"\nBenchmarking: {prompt_name} prompt ({NUM_RUNS} runs)")
        print("-" * 50)

        runs = []
        for run_idx in range(NUM_RUNS):
            # Reset peak memory tracking
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            metrics = measure_generation(model, tokenizer, prompt_text, max_tokens)
            runs.append(metrics)
            print(f"  Run {run_idx + 1}: {metrics['tokens_per_second']:.2f} tok/s, "
                  f"{metrics['total_time_s']:.2f}s total")

        # Calculate averages
        avg_metrics = {
            "prompt_name": prompt_name,
            "prompt_length": len(prompt_text),
            "max_tokens": max_tokens,
            "avg_input_tokens": sum(r["input_tokens"] for r in runs) / len(runs),
            "avg_output_tokens": sum(r["output_tokens"] for r in runs) / len(runs),
            "avg_total_time_s": sum(r["total_time_s"] for r in runs) / len(runs),
            "avg_ttft_s": sum(r["ttft_s"] for r in runs) / len(runs),
            "avg_tokens_per_second": sum(r["tokens_per_second"] for r in runs) / len(runs),
            "runs": runs,
        }

        results["prompts"].append(avg_metrics)

    # Get final VRAM stats
    results["vram_usage"] = get_vram_usage()
    results["peak_vram"] = get_peak_vram()

    return results


def print_results_table(results: dict):
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nModel: {results['model']}")
    print(f"Quantization: {results['quantization']}")
    print(f"Timestamp: {results['timestamp']}")

    # GPU info
    print("\nGPUs:")
    for gpu in results["gpu_info"]:
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")

    # Performance table
    print("\n" + "-" * 70)
    print(f"{'Prompt':<10} {'Output Tok':<12} {'Total Time':<12} {'TTFT':<10} {'Tok/s':<10}")
    print("-" * 70)

    for prompt in results["prompts"]:
        print(f"{prompt['prompt_name']:<10} "
              f"{prompt['avg_output_tokens']:<12.0f} "
              f"{prompt['avg_total_time_s']:<12.2f}s "
              f"{prompt['avg_ttft_s']:<10.3f}s "
              f"{prompt['avg_tokens_per_second']:<10.2f}")

    print("-" * 70)

    # Calculate overall average
    all_tps = [p["avg_tokens_per_second"] for p in results["prompts"]]
    avg_tps = sum(all_tps) / len(all_tps)
    print(f"{'AVERAGE':<10} {'':<12} {'':<12} {'':<10} {avg_tps:<10.2f}")

    # VRAM usage
    print("\nVRAM Usage (Current / Peak):")
    for i, (current, peak) in enumerate(zip(results["vram_usage"], results["peak_vram"])):
        print(f"  GPU {i}: {current['allocated_gb']:.2f} GB / {peak['peak_gb']:.2f} GB "
              f"(of {current['total_gb']:.1f} GB)")

    print("=" * 70)


def save_results_json(results: dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nBenchmark interrupted. Exiting...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPU performance on LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py mistralai/Mistral-7B-Instruct-v0.3
  python benchmark.py meta-llama/Llama-2-7b-chat-hf --4bit
  python benchmark.py google/gemma-7b --output results.json

Metrics measured:
  - Tokens per second (generation throughput)
  - Time to first token (TTFT / latency)
  - Total inference time
  - VRAM usage per GPU

Environment variables:
  LLM_MODEL_DIR   Override default model directory
        """,
    )

    parser.add_argument(
        "model_name",
        help="HuggingFace model ID (e.g., mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--model-dir",
        help="Override model storage directory",
    )
    parser.add_argument(
        "--4bit",
        dest="use_4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    model_dir = get_model_dir(args.model_dir)

    # Check if model exists
    if not model_exists(args.model_name, model_dir):
        print(f"\nError: Model not found: {args.model_name}")
        print(f"Expected location: {get_model_path(args.model_name, model_dir)}")
        print(f"\nPlease download the model first:")
        print(f"  python download.py {args.model_name}")
        sys.exit(1)

    # Print GPU info
    print_gpu_info()

    # Load model
    model_path = get_model_path(args.model_name, model_dir)
    quant_config = get_quantization_config(args.use_4bit, args.use_8bit)

    # Determine quantization string for results
    if args.use_4bit:
        quant_str = "4-bit"
    elif args.use_8bit:
        quant_str = "8-bit"
    else:
        quant_str = "none (fp16)"

    try:
        model, tokenizer = load_model(str(model_path), quant_config)
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    # Print VRAM usage after loading
    print_vram_usage()

    # Run benchmarks
    results = benchmark_model(model, tokenizer, args.model_name, quant_str)

    # Print results
    print_results_table(results)

    # Save to JSON if requested
    if args.output:
        save_results_json(results, args.output)


if __name__ == "__main__":
    main()
