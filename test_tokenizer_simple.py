"""
Quick test of tokenizer setup.
"""
from transformers import AutoTokenizer

# Load tokenizer
model_id = "mistralai/Mistral-7B-v0.3"
print(f"Loading tokenizer from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token ID: {tokenizer.pad_token_id}")

# Quick test
text = "Hello world! Testing the tokenizer."
tokens = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(tokens)

print(f"\nOriginal: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Match: {text == decoded}")

print("\nTokenizer ready for training!")