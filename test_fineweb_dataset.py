from datasets import load_dataset

print("=== TESTING FINEWEB-EDU DATASET ===")

try:
    print("Loading dataset with streaming=True...")

    # Load the dataset in streaming mode
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        streaming=True,
        split="train"  # Specify train split
    )

    print(f"Dataset loaded successfully!")
    print(f"Dataset type: {type(ds)}")

    # Get the first example
    print("\nGetting first row...")
    first_row = next(iter(ds))

    print(f"Keys in first row: {list(first_row.keys())}")

    # Show basic info about each field
    for key, value in first_row.items():
        if isinstance(value, str):
            # Show length and first 200 chars of text fields
            preview = value[:200] + "..." if len(value) > 200 else value
            print(f"\n{key}:")
            print(f"  Type: {type(value).__name__}")
            print(f"  Length: {len(value)}")
            print(f"  Preview: {repr(preview)}")
        else:
            # Show non-string fields directly (but truncated if long)
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            print(f"\n{key}: {value_str}")

    print(f"\n=== SUCCESS ===")
    print("Dataset streaming works! Ready for tokenization.")

except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e).__name__}")

    # Try to give helpful suggestions
    if "login" in str(e).lower() or "authentication" in str(e).lower():
        print("\nSuggestion: You may need to login with: huggingface-cli login")
    elif "not found" in str(e).lower():
        print("\nSuggestion: Check if the dataset name or config is correct")
    else:
        print(f"\nFull error details: {e}")