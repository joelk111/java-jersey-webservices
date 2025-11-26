#!/usr/bin/env python3
"""
Export fine-tuned LoRA model to Ollama format.

This script:
1. Merges LoRA adapters with base model
2. Converts to GGUF format (Ollama-compatible)
3. Creates Modelfile for Ollama

Usage:
    python export_to_ollama.py --model-dir ./output --output-name rules-llama

Then import to Ollama:
    ollama create rules-llama -f Modelfile
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def check_dependencies():
    """Check required tools are installed."""
    # Check for llama.cpp quantize tool
    if not shutil.which('llama-quantize'):
        print("WARNING: llama-quantize not found.")
        print("You may need to build llama.cpp for GGUF conversion:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make")
        return False
    return True


def merge_lora_with_base(model_dir: Path, output_dir: Path, base_model: str):
    """Merge LoRA adapters with base model."""
    print("\n" + "=" * 60)
    print("Merging LoRA adapters with base model")
    print("=" * 60)

    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install peft transformers torch")
        sys.exit(1)

    # Load config to get base model
    config_path = model_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path", base_model)

    print(f"Base model: {base_model}")
    print(f"LoRA adapter: {model_dir}")

    # Load base model
    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(model_dir))

    print("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


def convert_to_gguf(model_dir: Path, output_file: Path, quantization: str = "q4_k_m"):
    """Convert HuggingFace model to GGUF format."""
    print("\n" + "=" * 60)
    print("Converting to GGUF format")
    print("=" * 60)

    # Check if llama.cpp conversion script exists
    llama_cpp_dir = Path.home() / "llama.cpp"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("llama.cpp not found. Attempting to clone...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp",
            str(llama_cpp_dir)
        ], check=True)

        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r",
            str(llama_cpp_dir / "requirements.txt")
        ])

    # Convert to GGUF (fp16 first)
    fp16_output = output_file.with_suffix('.fp16.gguf')
    print(f"\nConverting to FP16 GGUF: {fp16_output}")

    subprocess.run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(fp16_output),
        "--outtype", "f16"
    ], check=True)

    # Quantize
    print(f"\nQuantizing to {quantization}: {output_file}")
    quantize_bin = llama_cpp_dir / "llama-quantize"

    if quantize_bin.exists():
        subprocess.run([
            str(quantize_bin),
            str(fp16_output),
            str(output_file),
            quantization
        ], check=True)

        # Clean up fp16 file
        fp16_output.unlink()
    else:
        print("WARNING: llama-quantize not found, using FP16 model")
        shutil.move(fp16_output, output_file)

    return output_file


def create_modelfile(output_dir: Path, model_name: str, gguf_file: Path):
    """Create Ollama Modelfile."""
    print("\n" + "=" * 60)
    print("Creating Ollama Modelfile")
    print("=" * 60)

    # System prompt for the fine-tuned model
    system_prompt = """You are a data quality rules assistant for the ORBIS data dictionary. You help users create validation rules using natural language and answer questions about rules.

You can create these rule types:
- REGEX: Pattern matching
- NOT_NULL: Required field
- LENGTH: String length validation
- RANGE: Numeric range
- IN_LIST: Allowed values
- CUSTOM_FUNCTION: Python validation functions

For complex rules, use JSON format:
- Composite: Multiple conditions with AND/OR
- Conditional: If X then validate Y
- Cross-Field: Compare two fields

Always ask for clarification when required information is missing."""

    modelfile_content = f'''# Modelfile for NLP Rules Engine
# Generated by export_to_ollama.py

FROM {gguf_file.name}

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# System prompt
SYSTEM """{system_prompt}"""

# Template for chat
TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""

# License
LICENSE """MIT License - NLP Rules Engine Fine-tuned Model"""
'''

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"Modelfile created: {modelfile_path}")

    # Print instructions
    print("\n" + "=" * 60)
    print("To import into Ollama:")
    print("=" * 60)
    print(f"  cd {output_dir}")
    print(f"  ollama create {model_name} -f Modelfile")
    print(f"\nThen use with:")
    print(f"  ollama run {model_name}")

    return modelfile_path


def export_for_ollama_direct(model_dir: Path, output_dir: Path, model_name: str):
    """
    Export using Unsloth's direct GGUF export (if available).
    This is faster and doesn't require llama.cpp.
    """
    print("\n" + "=" * 60)
    print("Attempting direct GGUF export with Unsloth")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Unsloth not available, falling back to standard export")
        return None

    try:
        # Load the saved model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Export to GGUF
        gguf_path = output_dir / f"{model_name}.gguf"
        print(f"Exporting to: {gguf_path}")

        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method="q4_k_m"
        )

        return gguf_path

    except Exception as e:
        print(f"Direct export failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export LoRA model to Ollama format')

    parser.add_argument('--model-dir', type=Path,
                       default=Path(__file__).parent / 'output',
                       help='Directory containing fine-tuned model')
    parser.add_argument('--output-dir', type=Path,
                       default=Path(__file__).parent / 'ollama_export',
                       help='Output directory for Ollama files')
    parser.add_argument('--output-name', default='rules-llama',
                       help='Name for the Ollama model')
    parser.add_argument('--base-model', default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Base model (if not in adapter config)')
    parser.add_argument('--quantization', default='q4_k_m',
                       choices=['q4_0', 'q4_k_m', 'q5_k_m', 'q8_0', 'f16'],
                       help='Quantization method')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip merge step (if already merged)')

    args = parser.parse_args()

    # Check model exists
    if not args.model_dir.exists():
        print(f"ERROR: Model directory not found: {args.model_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("NLP Rules Engine - Export to Ollama")
    print("=" * 60)
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model name: {args.output_name}")
    print(f"Quantization: {args.quantization}")

    # Try direct Unsloth export first
    gguf_path = export_for_ollama_direct(args.model_dir, args.output_dir, args.output_name)

    if gguf_path is None:
        # Fall back to standard export
        print("\nUsing standard export pipeline...")

        # Step 1: Merge LoRA with base
        if not args.skip_merge:
            merged_dir = args.output_dir / "merged_model"
            merge_lora_with_base(args.model_dir, merged_dir, args.base_model)
        else:
            merged_dir = args.model_dir

        # Step 2: Convert to GGUF
        gguf_path = args.output_dir / f"{args.output_name}.gguf"
        convert_to_gguf(merged_dir, gguf_path, args.quantization)

    # Step 3: Create Modelfile
    create_modelfile(args.output_dir, args.output_name, gguf_path)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nFiles created in: {args.output_dir}")
    print(f"  - {gguf_path.name}")
    print(f"  - Modelfile")


if __name__ == '__main__':
    main()
