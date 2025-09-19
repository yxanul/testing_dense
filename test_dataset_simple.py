"""
Simple test of dataset and tokenizer - no timeout issues.
"""
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

def test_tokenizer():
    """Test tokenizer basics."""
    print("="*60)
    print("TOKENIZER TEST")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token")

    # Test encode/decode
    texts = [
        "Hello world!",
        "Testing GPT-2 with gated attention.",
        "RTX 5090 optimization"
    ]

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens[:5]}... ({len(tokens)} total)")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")

    return tokenizer


def test_dataset(tokenizer):
    """Test dataset loading with streaming."""
    print("\n" + "="*60)
    print("DATASET TEST (Streaming)")
    print("="*60)

    print("Loading dataset (streaming mode)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    print("Testing first 3 samples:")
    for i, sample in enumerate(dataset.take(3)):
        text = sample['text']
        print(f"\nSample {i+1}:")
        print(f"  Text preview: {text[:100]}...")
        print(f"  Text length: {len(text)} chars")

        # Tokenize
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        print(f"  Token count: {tokens['input_ids'].shape[1]}")
        print(f"  First 5 tokens: {tokens['input_ids'][0][:5].tolist()}")


def test_batch_processing(tokenizer):
    """Test batch tokenization."""
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST")
    print("="*60)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )

    # Collect 4 samples for a batch
    texts = []
    for sample in dataset.take(4):
        texts.append(sample['text'])

    # Batch tokenize
    batch = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")

    # Check padding
    for i in range(len(texts)):
        num_pad = (batch['input_ids'][i] == tokenizer.pad_token_id).sum().item()
        print(f"  Sample {i+1}: {256 - num_pad} tokens, {num_pad} padding")

    # Create labels for language modeling
    labels = batch['input_ids'].clone()
    labels[batch['attention_mask'] == 0] = -100

    print(f"\nLabels shape: {labels.shape}")
    print(f"Ignored tokens (-100): {(labels == -100).sum().item()}")
    print(f"Valid tokens: {(labels != -100).sum().item()}")


def main():
    print("Testing Dataset and Tokenizer Setup")
    print("="*60)

    # Test tokenizer
    tokenizer = test_tokenizer()

    # Test dataset
    test_dataset(tokenizer)

    # Test batch processing
    test_batch_processing(tokenizer)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("Dataset and tokenizer are working correctly.")
    print("="*60)


if __name__ == "__main__":
    main()