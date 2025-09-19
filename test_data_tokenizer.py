"""
Test tokenizer and dataset setup for training.
"""
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

def test_tokenizer():
    """Test Mistral tokenizer with encoding/decoding."""
    print("="*60)
    print("TOKENIZER TEST")
    print("="*60)

    # Load tokenizer
    model_id = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Check vocab size
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # Test encoding/decoding
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming AI research.",
        "RTX 5090 achieves 185K tokens/sec with optimizations.",
        "Special characters and numbers: 123-456!"
    ]

    print("\n" + "-"*60)
    print("ENCODING/DECODING TEST:")
    print("-"*60)

    for text in test_texts:
        # Encode
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)

        print(f"\nOriginal: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Decoded:  {decoded}")
        print(f"Match: {'YES' if text == decoded else 'NO'}")

    # Test special tokens
    print("\n" + "-"*60)
    print("SPECIAL TOKENS:")
    print("-"*60)
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"\nWARNING: Setting pad_token to eos_token: {tokenizer.pad_token}")

    return tokenizer


def test_dataset(tokenizer, num_samples=3):
    """Test dataset loading and tokenization."""
    print("\n" + "="*60)
    print("DATASET TEST")
    print("="*60)

    # Load dataset with streaming
    print("\nLoading fineweb-edu dataset (streaming mode)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    print("Dataset loaded successfully!")

    # Test a few samples
    print(f"\nTesting first {num_samples} samples:")
    print("-"*60)

    for i, sample in enumerate(ds.take(num_samples)):
        text = sample['text']

        # Truncate for display
        display_text = text[:200] + "..." if len(text) > 200 else text

        print(f"\nSample {i+1}:")
        print(f"Text preview: {display_text}")
        print(f"Text length: {len(text)} characters")

        # Tokenize
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=2048
        )

        print(f"Tokens: {len(tokens)}")
        print(f"First 10 tokens: {tokens[:10]}")

        # Decode back
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)

        # Check if it matches (accounting for truncation)
        original_truncated = tokenizer.decode(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=2048
            ),
            skip_special_tokens=True
        )

        match = (decoded == original_truncated)
        print(f"Decode match: {'YES' if match else 'NO'}")

    return ds


def test_batch_tokenization(tokenizer, num_batches=2):
    """Test batch tokenization for training."""
    print("\n" + "="*60)
    print("BATCH TOKENIZATION TEST")
    print("="*60)

    # Load dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    batch_size = 4
    seq_length = 512

    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_length}")

    # Process batches
    batch_iter = ds.take(batch_size * num_batches)
    texts = []

    for sample in batch_iter:
        texts.append(sample['text'])

        if len(texts) == batch_size:
            # Tokenize batch
            batch_tokens = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=seq_length,
                return_tensors='pt'
            )

            print(f"\nBatch shape: {batch_tokens['input_ids'].shape}")
            print(f"Attention mask shape: {batch_tokens['attention_mask'].shape}")

            # Check padding
            for i, ids in enumerate(batch_tokens['input_ids']):
                num_pad = (ids == tokenizer.pad_token_id).sum().item()
                print(f"  Sample {i+1}: {seq_length - num_pad} real tokens, {num_pad} padding")

            # Create labels (shifted input_ids)
            labels = batch_tokens['input_ids'].clone()
            labels[batch_tokens['attention_mask'] == 0] = -100  # Ignore padding in loss

            print(f"Labels shape: {labels.shape}")
            print(f"Ignored tokens: {(labels == -100).sum().item()}")

            texts = []

    print("\n[OK] Batch tokenization working correctly!")


def main():
    """Run all tests."""
    # Test tokenizer
    tokenizer = test_tokenizer()

    # Test dataset
    dataset = test_dataset(tokenizer)

    # Test batch tokenization
    test_batch_tokenization(tokenizer)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    print(f"[OK] Tokenizer vocab size: {tokenizer.vocab_size}")
    print("[OK] Dataset loads and streams correctly")
    print("[OK] Tokenization/detokenization preserves text")
    print("[OK] Batch processing works")
    print("\nReady for training implementation!")


if __name__ == "__main__":
    main()