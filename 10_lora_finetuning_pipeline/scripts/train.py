"""Training script for the LoRA fine-tuning pipeline.

Usage:
    python scripts/train.py --config config/training.yaml --data data/train.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

from src.data_preprocessor import load_jsonl, split_dataset
from src.models import TrainingConfig
from src.training import LoRATrainer
from src.evaluation import evaluate_general_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--data", required=True, help="Path to training data JSONL")
    parser.add_argument("--benchmarks", action="store_true", help="Run general benchmark evaluation after training")
    args = parser.parse_args()

    config_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = TrainingConfig.model_validate(config_raw)

    print(f"Loading dataset from {args.data}")
    items = load_jsonl(args.data)
    print(f"Loaded {len(items)} items")

    split = split_dataset(items)
    print(f"Split: {len(split.train)} train / {len(split.validation)} val / {len(split.test)} test")
    print(f"Leakage check: {'PASSED' if split.leakage_check_passed else 'FAILED'}")

    if not split.leakage_check_passed:
        print("WARNING: Evaluation leakage detected. Review dataset for duplicates.")

    trainer = LoRATrainer(config)
    trainer.init_wandb()

    print(f"Loading base model: {config.base_model}")
    try:
        trainer.load_model()
        trainer.configure_lora()
    except RuntimeError as exc:
        print(f"Model loading skipped: {exc}")

    print("Starting training loop")
    metrics = trainer.train(split)
    print(f"Training completed. {len(metrics)} steps logged.")

    if args.benchmarks:
        print("Running general benchmark evaluation for catastrophic forgetting")
        results = evaluate_general_benchmarks(config.base_model, config.output_dir)
        for r in results:
            status = "FORGETTING DETECTED" if r.forgetting_detected else "OK"
            print(f"  {r.benchmark_name}: base={r.base_model_score:.3f} ft={r.finetuned_model_score:.3f} delta={r.performance_delta:+.3f} [{status}]")


if __name__ == "__main__":
    main()
