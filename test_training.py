"""
Quick test of the training setup without full training.
"""
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from model_te_gated import GatedGPT2Model, get_gated_gpt2_config
from train_gated import DataCollator, TrainingConfig
import transformer_engine.pytorch as te


def test_model():
    """Test model creation and forward pass."""
    print("="*60)
    print("MODEL TEST")
    print("="*60)

    # Create config
    config = get_gated_gpt2_config(
        vocab_size=32768,
        use_gated_attention=True,
        use_rope=True
    )

    # Create model
    model = GatedGPT2Model(config)
    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Test forward pass
    batch_size = 2
    seq_length = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    model = model.to(device)

    # Forward pass with FP8
    with te.fp8_autocast(enabled=config.use_fp8):
        logits, loss = model(input_ids, labels=input_ids)

    print(f"Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    return model


def test_data_pipeline():
    """Test data loading and tokenization."""
    print("\n" + "="*60)
    print("DATA PIPELINE TEST")
    print("="*60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load dataset (just a few samples)
    print("Loading dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    # Test collator
    collator = DataCollator(tokenizer, max_length=512)
    samples = list(dataset.take(4))
    batch = collator(samples)

    print(f"Batch created successfully!")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Num padding tokens: {(batch['labels'] == -100).sum().item()}")

    return batch


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("TRAINING STEP TEST")
    print("="*60)

    # Config
    config = TrainingConfig(
        batch_size=2,
        sequence_length=128,
        num_train_steps=10,
        wandb_project="test"
    )

    # Create model
    model_config = get_gated_gpt2_config(
        vocab_size=32768,
        use_gated_attention=True
    )
    model = GatedGPT2Model(model_config).to(config.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Create dummy batch
    batch = {
        "input_ids": torch.randint(0, 32768, (2, 128)).to(config.device),
        "attention_mask": torch.ones(2, 128).to(config.device),
        "labels": torch.randint(0, 32768, (2, 128)).to(config.device)
    }

    # Training step
    model.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=config.mixed_precision == "bf16", dtype=torch.bfloat16):
        logits, loss = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"Training step successful!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Grad norm: {grad_norm:.4f}")

    # Check if gating is working (sparse activations)
    with torch.no_grad():
        # Get gate activations from first attention layer
        for name, module in model.named_modules():
            if hasattr(module, 'gate_proj') and hasattr(module, 'gate_activation'):
                x = torch.randn(128, 2, 768).to(config.device)  # Dummy input
                gate_scores = module.gate_proj(x)
                gate_scores = module.gate_activation(gate_scores)
                sparsity = (gate_scores < 0.1).float().mean().item()
                print(f"\nGating sparsity in {name}: {sparsity*100:.1f}% < 0.1")
                break


def test_memory_usage():
    """Test memory usage with different settings."""
    print("\n" + "="*60)
    print("MEMORY TEST")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Test with gating
        config_gated = get_gated_gpt2_config(use_gated_attention=True)
        model_gated = GatedGPT2Model(config_gated).to(device)
        input_ids = torch.randint(0, 32768, (8, 512)).to(device)

        with te.fp8_autocast(enabled=True):
            _, loss = model_gated(input_ids, labels=input_ids)
            loss.backward()

        memory_gated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Memory with gating: {memory_gated:.1f} MB")

        # Clean up
        del model_gated, loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Test without gating
        config_normal = get_gated_gpt2_config(use_gated_attention=False)
        model_normal = GatedGPT2Model(config_normal).to(device)

        with te.fp8_autocast(enabled=True):
            _, loss = model_normal(input_ids, labels=input_ids)
            loss.backward()

        memory_normal = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Memory without gating: {memory_normal:.1f} MB")
        print(f"Difference: {memory_gated - memory_normal:.1f} MB")


def main():
    """Run all tests."""
    print("Testing Gated GPT-2 Training Setup")
    print("="*60)

    # Test model
    model = test_model()

    # Test data pipeline
    batch = test_data_pipeline()

    # Test training step
    test_training_step()

    # Test memory
    test_memory_usage()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nReady to train with:")
    print("  python train_gated.py")
    print("\nOr import and use:")
    print("  from model_te_gated import GatedGPT2Model, get_gated_gpt2_config")
    print("  from train_gated import main as train_main")


if __name__ == "__main__":
    main()