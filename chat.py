#!/usr/bin/env python3
"""
Interactive CLI chat with downloaded LLM models.

Usage:
    python chat.py mistralai/Mistral-7B-Instruct-v0.3
    python chat.py meta-llama/Llama-2-7b-chat-hf --4bit
    python chat.py google/gemma-7b --8bit
"""

import argparse
import signal
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    get_model_dir,
    get_model_path,
    model_exists,
    print_gpu_info,
    print_vram_usage,
)


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

    Args:
        model_path: Path to local model directory
        quantization_config: Optional quantization settings

    Returns:
        Tuple of (model, tokenizer)

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

    # Ensure pad token is set
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


def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """
    Generate a response given conversation history.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Generated response text
    """
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback for models without chat template
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def chat_loop(model, tokenizer):
    """Run the interactive chat loop."""
    messages = []

    print("\n" + "=" * 60)
    print("Interactive Chat")
    print("=" * 60)
    print("Commands:")
    print("  /clear - Clear conversation history")
    print("  /quit  - Exit chat")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "/clear":
                messages = []
                print("\nConversation cleared.\n")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages)
            print(response + "\n")

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupted. Exiting...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py mistralai/Mistral-7B-Instruct-v0.3
  python chat.py meta-llama/Llama-2-7b-chat-hf --4bit
  python chat.py google/gemma-7b --8bit --model-dir /custom/path

Commands during chat:
  /clear  Clear conversation history
  /quit   Exit the chat

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
        help="Use 4-bit quantization (reduces VRAM, slight quality loss)",
    )
    parser.add_argument(
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Use 8-bit quantization (reduces VRAM, minimal quality loss)",
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

    try:
        model, tokenizer = load_model(str(model_path), quant_config)
    except Exception as e:
        print(f"\nError loading model: {e}")
        sys.exit(1)

    # Print VRAM usage after loading
    print_vram_usage()

    # Start chat
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
