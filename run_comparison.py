"""
Script to run comparison experiments between baseline and gated attention models.
This script launches training runs with different configurations for easy comparison.
"""
import subprocess
import time
import os

def run_experiment(name, args):
    """Run a single experiment with given arguments."""
    print("="*60)
    print(f"Starting experiment: {name}")
    print(f"Arguments: {' '.join(args)}")
    print("="*60)

    cmd = ["python", "train_gated_with_args.py"] + args

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Experiment '{name}' completed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{name}' failed with error: {e}\n")

    time.sleep(5)  # Small pause between experiments


def main():
    """Run comparison experiments."""

    # Common arguments for all experiments
    common_args = [
        "--batch_size", "12",
        "--sequence_length", "2048",
        "--num_train_steps", "10000",  # Shorter for comparison
        "--eval_interval", "500",
        "--save_interval", "2000",
        "--log_interval", "10",
        "--learning_rate", "3e-4",
        "--warmup_steps", "500",
        "--wandb_project", "gpt2-gated-comparison"
    ]

    experiments = [
        # Baseline model without gated attention
        {
            "name": "Baseline-NoGating-FP8",
            "args": common_args + [
                "--no_gated_attention",
                "--use_fp8",
                "--wandb_run_name", "baseline_fp8"
            ]
        },

        # Gated attention model with FP8
        {
            "name": "Gated-Attention-FP8",
            "args": common_args + [
                "--use_gated_attention",
                "--use_fp8",
                "--wandb_run_name", "gated_fp8"
            ]
        },

        # Baseline without FP8 (for comparison)
        {
            "name": "Baseline-NoGating-NoFP8",
            "args": common_args + [
                "--no_gated_attention",
                "--no_fp8",
                "--wandb_run_name", "baseline_no_fp8"
            ]
        },

        # Gated attention without FP8
        {
            "name": "Gated-Attention-NoFP8",
            "args": common_args + [
                "--use_gated_attention",
                "--no_fp8",
                "--wandb_run_name", "gated_no_fp8"
            ]
        },

        # Gated attention with QK norm
        {
            "name": "Gated-QKNorm-FP8",
            "args": common_args + [
                "--use_gated_attention",
                "--use_qk_norm",
                "--use_fp8",
                "--wandb_run_name", "gated_qknorm_fp8"
            ]
        }
    ]

    print("\n" + "="*60)
    print("GPT-2 GATED ATTENTION COMPARISON EXPERIMENTS")
    print("="*60)
    print(f"Running {len(experiments)} experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    print("="*60 + "\n")

    # Ask for confirmation
    response = input("Do you want to run all experiments? (y/n): ").lower()
    if response != 'y':
        print("Aborted.")
        return

    # Run experiments
    for exp in experiments:
        run_experiment(exp["name"], exp["args"])

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)
    print("\nCheck WandB for detailed comparisons:")
    print("  - Training loss curves")
    print("  - Perplexity comparisons")
    print("  - Training speed (tokens/sec)")
    print("  - Gradient norms (stability)")
    print("\nCheckpoints saved in:")
    for exp in experiments:
        model_type = "gpt2-124M-gated" if "--use_gated_attention" in exp["args"] else "gpt2-124M-baseline"
        print(f"  ./checkpoints/{model_type}/")


if __name__ == "__main__":
    main()