#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for NLP Rules Engine

This script fine-tunes a Llama model using LoRA (Low-Rank Adaptation)
on the generated training data.

Supports:
- Unsloth (fastest, recommended)
- HuggingFace PEFT (fallback)

Usage:
    # With Unsloth (recommended)
    python train_lora.py --backend unsloth --epochs 3

    # With HuggingFace PEFT
    python train_lora.py --backend peft --epochs 3

Requirements:
    pip install unsloth peft transformers datasets accelerate bitsandbytes
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_training_data(filepath: Path):
    """Load training data from JSONL file."""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_for_training(examples, tokenizer):
    """Format examples for training with chat template."""
    formatted = []
    for ex in examples:
        messages = ex.get('messages', [])
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted.append({"text": text})
    return formatted


def train_with_unsloth(args):
    """Train using Unsloth (4x faster than standard)."""
    print("=" * 60)
    print("Training with Unsloth (optimized)")
    print("=" * 60)

    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
    except ImportError:
        print("ERROR: Unsloth not installed. Install with:")
        print("  pip install unsloth")
        sys.exit(1)

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load model
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format training data
    print(f"\nLoading training data from: {args.data}")
    examples = load_training_data(args.data)
    formatted = format_for_training(examples, tokenizer)
    dataset = Dataset.from_list(formatted)

    print(f"Training examples: {len(dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save LoRA only (smaller file)
    lora_dir = args.output_dir / "lora_adapter"
    print(f"Saving LoRA adapter to: {lora_dir}")
    model.save_pretrained_merged(
        str(lora_dir),
        tokenizer,
        save_method="lora",
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return model, tokenizer


def train_with_peft(args):
    """Train using HuggingFace PEFT (standard, more compatible)."""
    print("=" * 60)
    print("Training with HuggingFace PEFT")
    print("=" * 60)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install peft transformers datasets trl bitsandbytes")
        sys.exit(1)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Add LoRA
    print("\nAdding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format training data
    print(f"\nLoading training data from: {args.data}")
    examples = load_training_data(args.data)
    formatted = format_for_training(examples, tokenizer)
    dataset = Dataset.from_list(formatted)

    print(f"Training examples: {len(dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning for NLP Rules Engine')

    # Model settings
    parser.add_argument('--model', default='unsloth/llama-3-8b-Instruct-bnb-4bit',
                       help='Base model to fine-tune')
    parser.add_argument('--backend', choices=['unsloth', 'peft'], default='unsloth',
                       help='Training backend (unsloth is 4x faster)')

    # Data
    parser.add_argument('--data', type=Path,
                       default=Path(__file__).parent / 'training_data.jsonl',
                       help='Training data JSONL file')

    # Output
    parser.add_argument('--output-dir', type=Path,
                       default=Path(__file__).parent / 'output',
                       help='Output directory for model')

    # LoRA settings
    parser.add_argument('--lora-r', type=int, default=16,
                       help='LoRA rank (higher = more capacity, more memory)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout')

    # Training settings
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size per device')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=10,
                       help='Warmup steps')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    if not args.data.exists():
        print(f"ERROR: Training data not found: {args.data}")
        print("Run generate_training_data.py first")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("NLP Rules Engine - LoRA Fine-tuning")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} (effective: {args.batch_size * args.gradient_accumulation})")
    print("=" * 60)

    # Train
    if args.backend == 'unsloth':
        model, tokenizer = train_with_unsloth(args)
    else:
        model, tokenizer = train_with_peft(args)

    # Save training info
    info = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.model,
        "backend": args.backend,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "training_examples": len(load_training_data(args.data)),
    }

    with open(args.output_dir / "training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining info saved to: {args.output_dir / 'training_info.json'}")


if __name__ == '__main__':
    main()
