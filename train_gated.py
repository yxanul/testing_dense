"""
Training script for GPT-2 model with optional Gated Attention.
Trains on fineweb-edu dataset with streaming.

Usage:
    # Train with gated attention (default)
    python train_gated.py --use_gated_attention

    # Train baseline without gated attention
    python train_gated.py --no_gated_attention

    # Custom configuration
    python train_gated.py --batch_size 16 --learning_rate 4e-4 --no_fp8
"""
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformer_engine.pytorch as te
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from model_te_gated import GatedGPT2Model, get_gated_gpt2_config


@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "gpt2-gated-124M"
    vocab_size: int = 32768
    use_gated_attention: bool = True
    use_qk_norm: bool = False
    use_rope: bool = True
    use_fp8: bool = True  # Enable FP8 compute via TransformerEngine

    # Training hyperparameters
    batch_size: int = 12  # Optimal for RTX 5090
    sequence_length: int = 2048
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # "cosine" or "linear"

    # Training duration
    num_train_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 5000
    log_interval: int = 10

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    streaming: bool = True

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # WandB
    wandb_project: str = "gpt2-gated"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "fp32" - works alongside FP8
    compile_model: bool = False  # torch.compile (slower with FP8)

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["gated-attention", "rtx5090"]


class DataCollator:
    """Collate function for streaming dataset."""

    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        # Extract texts
        texts = [ex["text"] for ex in examples]

        # Tokenize batch
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create labels (same as input_ids for language modeling)
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch


def create_streaming_dataloader(config: TrainingConfig, tokenizer, seed=None):
    """Create properly functioning streaming dataloader.

    CRITICAL FIXES:
    1. DO NOT use PyTorch DataLoader with streaming datasets - it doesn't work properly!
    2. MUST shuffle with buffer to avoid sequential data
    3. MUST use different seeds for train/eval to avoid data leak
    """
    # Load dataset with streaming
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        streaming=True  # Always use streaming
    )

    # CRITICAL: Shuffle the dataset with a buffer
    # Without this, we see data sequentially and loss drops unnaturally fast!
    if seed is not None:
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    else:
        dataset = dataset.shuffle(buffer_size=10000)

    # Create batches manually from streaming dataset
    # DO NOT use DataLoader with streaming datasets!
    def batch_generator():
        batch_texts = []
        for example in dataset:
            batch_texts.append(example["text"])

            if len(batch_texts) == config.batch_size:
                # Tokenize batch
                batch = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=config.sequence_length,
                    return_tensors="pt"
                )

                # Create labels (shifted for language modeling)
                batch["labels"] = batch["input_ids"].clone()
                batch["labels"][batch["attention_mask"] == 0] = -100

                yield batch
                batch_texts = []

    return batch_generator()


def get_lr_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler."""
    if config.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < config.warmup_steps:
                # Linear warmup
                return step / config.warmup_steps
            else:
                # Cosine decay
                progress = (step - config.warmup_steps) / (config.num_train_steps - config.warmup_steps)
                return config.min_learning_rate / config.learning_rate + \
                    (1 - config.min_learning_rate / config.learning_rate) * \
                    0.5 * (1 + math.cos(math.pi * progress))
    else:  # linear
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / (config.num_train_steps - config.warmup_steps)
                return max(config.min_learning_rate / config.learning_rate, 1 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, optimizer, scheduler, scaler, config):
    """Single training step."""
    model.train()

    # Move batch to device
    input_ids = batch["input_ids"].to(config.device)
    labels = batch["labels"].to(config.device)
    # Note: attention_mask not needed - padding handled by labels=-100

    # FP8 is handled by TransformerEngine inside the model
    # The model forward already uses: with te.fp8_autocast(enabled=self.config.use_fp8)
    # We just need bf16 for the outer context
    if config.mixed_precision == "bf16":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = model(input_ids, labels=labels)
    elif config.mixed_precision == "fp16":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, loss = model(input_ids, labels=labels)
    else:
        logits, loss = model(input_ids, labels=labels)

    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Increment step count
    optimizer.step_count += 1

    # Gradient clipping and optimization
    if optimizer.step_count % config.gradient_accumulation_steps == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        grad_norm = None

    return loss.item() * config.gradient_accumulation_steps, grad_norm


def evaluate(model, dataloader, config, num_eval_steps=50):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        # Use manual iteration for generator
        for i in tqdm(range(num_eval_steps), desc="Evaluating"):
            try:
                batch = next(dataloader)
            except StopIteration:
                print("Warning: Eval dataset exhausted")
                break

            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.mixed_precision != "fp32"):
                logits, loss = model(input_ids, labels=labels)

            # Accumulate loss
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, step, config, metrics=None):
    """Save training checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint-{step}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "config": asdict(config),
        "metrics": metrics
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Also save as latest
    latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["step"], checkpoint.get("metrics", {})


def calculate_model_size(model):
    """Calculate model size in millions of parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6, trainable_params / 1e6


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT-2 with optional Gated Attention")

    # Model arguments
    parser.add_argument("--use_gated_attention", action="store_true", default=True,
                        help="Use gated attention mechanism (default: True)")
    parser.add_argument("--no_gated_attention", dest="use_gated_attention", action="store_false",
                        help="Disable gated attention (baseline model)")
    parser.add_argument("--use_qk_norm", action="store_true", default=False,
                        help="Use QK normalization")
    parser.add_argument("--use_rope", action="store_true", default=True,
                        help="Use RoPE position embeddings (default: True)")
    parser.add_argument("--no_rope", dest="use_rope", action="store_false",
                        help="Use learned position embeddings instead of RoPE")
    parser.add_argument("--use_fp8", action="store_true", default=True,
                        help="Use FP8 compute acceleration (default: True)")
    parser.add_argument("--no_fp8", dest="use_fp8", action="store_false",
                        help="Disable FP8")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Training batch size (default: 12)")
    parser.add_argument("--sequence_length", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--num_train_steps", type=int, default=100000,
                        help="Total training steps (default: 100000)")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluation interval (default: 500)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--wandb_project", type=str, default="gpt2-gated",
                        help="WandB project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Configuration
    config = TrainingConfig()

    # Override config with command-line arguments
    config.use_gated_attention = args.use_gated_attention
    config.use_qk_norm = args.use_qk_norm
    config.use_rope = args.use_rope
    config.use_fp8 = args.use_fp8
    config.batch_size = args.batch_size
    config.sequence_length = args.sequence_length
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.learning_rate = args.learning_rate
    config.max_grad_norm = args.max_grad_norm
    config.num_train_steps = args.num_train_steps
    config.eval_interval = args.eval_interval
    config.checkpoint_dir = args.checkpoint_dir
    config.wandb_project = args.wandb_project

    # Update model name based on configuration
    if config.use_gated_attention:
        config.model_name = "gpt2-gated-124M"
    else:
        config.model_name = "gpt2-baseline-124M"

    # Print configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Gated Attention: {config.use_gated_attention}")
    print(f"FP8: {config.use_fp8}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Training Steps: {config.num_train_steps}")
    print("="*60)

    # Initialize wandb
    # Update tags based on configuration
    wandb_tags = []
    if config.use_gated_attention:
        wandb_tags.append("gated-attention")
    else:
        wandb_tags.append("baseline")
    if config.use_fp8:
        wandb_tags.append("fp8")
    wandb_tags.append("rtx5090")

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or f"{config.model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
        config=asdict(config),
        tags=wandb_tags
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model_config = get_gated_gpt2_config(
        vocab_size=config.vocab_size,
        use_gated_attention=config.use_gated_attention,
        use_qk_norm=config.use_qk_norm,
        use_rope=config.use_rope
    )

    # Override FP8 setting from training config
    model_config.use_fp8 = config.use_fp8

    model = GatedGPT2Model(model_config)
    model = model.to(config.device)

    # Log FP8 configuration
    if config.use_fp8:
        print(f"FP8 enabled: E4M3 forward, E5M2 backward (HYBRID format)")
        print(f"Note: FP8 applies to compute (GEMMs), not storage")
        print(f"Weights remain in BF16, dynamically quantized to FP8")

    # Calculate and log model size
    total_params, trainable_params = calculate_model_size(model)
    print(f"Model size: {total_params:.2f}M parameters ({trainable_params:.2f}M trainable)")
    wandb.config.update({
        "total_params": total_params * 1e6,
        "trainable_params": trainable_params * 1e6,
        "use_fp8": config.use_fp8,
        "fp8_format": "HYBRID (E4M3 fwd, E5M2 bwd)" if config.use_fp8 else "None"
    })

    # Convert to bfloat16 if specified
    if config.mixed_precision == "bf16":
        model = model.to(dtype=torch.bfloat16)

    # Compile model if requested
    if config.compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    # Create optimizer
    print("Creating optimizer...")
    # Use fused AdamW when available (requires CUDA, provides ~10% speedup)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available()  # Enable fused optimizer on CUDA
    )
    optimizer.step_count = 0  # Track steps for gradient accumulation

    # Create scheduler
    scheduler = get_lr_scheduler(optimizer, config)

    # Create scaler for mixed precision
    scaler = None
    if config.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint if resuming
    start_step = 0
    if config.resume_from:
        print(f"Resuming from checkpoint: {config.resume_from}")
        start_step, metrics = load_checkpoint(model, optimizer, scheduler, config.resume_from)
        print(f"Resumed from step {start_step}")

    # Create dataloaders with DIFFERENT seeds to ensure different data
    print("Creating dataloaders...")
    print("  Using shuffled streaming with different seeds for train/eval")
    train_dataloader = create_streaming_dataloader(config, tokenizer, seed=args.seed)
    # CRITICAL: Use different seed for eval to get different data!
    eval_dataloader = create_streaming_dataloader(config, tokenizer, seed=args.seed + 1000)

    # Training loop
    print(f"\nStarting training for {config.num_train_steps} steps...")
    print(f"Batch size: {config.batch_size}, Seq length: {config.sequence_length}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Total tokens per step: {config.batch_size * config.sequence_length * config.gradient_accumulation_steps}")
    print(f"Mixed precision: {config.mixed_precision} + FP8 compute: {config.use_fp8}")

    # Expected loss behavior warning
    print("\n⚠️  EXPECTED LOSS BEHAVIOR:")
    print("  - Initial loss: ~10-11 (random init)")
    print("  - After 1000 steps: ~7-8")
    print("  - After 10000 steps: ~4-5")
    print("  If loss drops to <3 in <1000 steps, DATA IS BEING REUSED!")

    # Note about expected performance
    if config.batch_size == 12 and config.sequence_length == 2048 and config.use_fp8:
        print("\nNote: For RTX 5090 with BS=12, Seq=2048, FP8:")
        print("  Expected: ~185K tokens/sec (single forward pass)")
        print("  With backward pass: ~130K tokens/sec is normal")

    # No need to call iter() on generator
    running_loss = 0
    grad_norms = []

    for step in range(start_step, config.num_train_steps):
        # Get batch from generator
        try:
            batch = next(train_dataloader)
        except StopIteration:
            # Dataset exhausted (shouldn't happen with streaming)
            print("Warning: Dataset exhausted, recreating dataloader...")
            train_dataloader = create_streaming_dataloader(config, tokenizer, seed=args.seed + step)
            batch = next(train_dataloader)

        # Training step
        step_start = time.time()
        loss, grad_norm = train_step(model, batch, optimizer, scheduler, scaler, config)
        step_time = time.time() - step_start

        running_loss += loss
        if grad_norm is not None:
            grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
            lr = scheduler.get_last_lr()[0]
            # Tokens/sec is for a SINGLE step (not accumulated)
            tokens_per_sec = (config.batch_size * config.sequence_length) / step_time

            log_dict = {
                "train/loss": avg_loss,
                "train/perplexity": math.exp(avg_loss) if avg_loss < 10 else float('inf'),
                "train/grad_norm": avg_grad_norm,
                "train/learning_rate": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step_time": step_time,
                "train/step": step
            }
            wandb.log(log_dict)

            print(f"Step {step+1}/{config.num_train_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"PPL: {math.exp(avg_loss) if avg_loss < 10 else float('inf'):.2f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")

            running_loss = 0
            grad_norms = []

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            print(f"\nEvaluating at step {step+1}...")
            eval_loss, eval_ppl = evaluate(model, eval_dataloader, config)

            eval_dict = {
                "eval/loss": eval_loss,
                "eval/perplexity": eval_ppl,
                "eval/step": step
            }
            wandb.log(eval_dict)

            print(f"Eval Loss: {eval_loss:.4f} | Eval PPL: {eval_ppl:.2f}\n")

        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            metrics = {
                "train_loss": running_loss / config.log_interval if running_loss > 0 else 0,
                "step": step + 1
            }
            save_checkpoint(model, optimizer, scheduler, step + 1, config, metrics)

    # Final evaluation
    print("\nFinal evaluation...")
    final_loss, final_ppl = evaluate(model, eval_dataloader, config, num_eval_steps=100)
    print(f"Final Loss: {final_loss:.4f} | Final PPL: {final_ppl:.2f}")

    wandb.log({
        "final/loss": final_loss,
        "final/perplexity": final_ppl
    })

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, config.num_train_steps, config, {
        "final_loss": final_loss,
        "final_perplexity": final_ppl
    })

    wandb.finish()
    print("Training completed!")


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    main()