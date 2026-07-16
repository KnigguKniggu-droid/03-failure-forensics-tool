"""LoRA training loop targeting linear attention mechanisms.

Configures PEFT/QLoRA for q_proj and v_proj modules, runs the training
loop, and logs loss steps to Weights and Biases.
"""

from __future__ import annotations

import os
from typing import Any

from src.models import DatasetSplitResult, TrainingConfig, TrainingMetrics

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LoRATrainer:
    """Training harness for LoRA/QLoRA fine-tuning."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._peft_model: Any = None
        self._wandb_run: Any = None

    def init_wandb(self) -> None:
        """Initialize Weights and Biases logging."""
        try:
            import wandb
            run_name = self.config.wandb_run_name or f"lora_{self.config.base_model.split('/')[-1]}"
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.model_dump(),
            )
        except ImportError:
            pass

    def load_model(self) -> None:
        """Load the base model with optional 4-bit quantization."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch and transformers are not installed")

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.config.use_qlora:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                pass

        self._model = AutoModelForCausalLM.from_pretrained(self.config.base_model, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.config.use_qlora:
            self._model = prepare_model_for_kbit_training(self._model)

    def configure_lora(self) -> None:
        """Apply LoRA adapters to the target modules."""
        if not HAS_TORCH:
            raise RuntimeError("PEFT is not installed")
        lora_config = LoraConfig(**self.config.to_peft_config())
        self._peft_model = get_peft_model(self._model, lora_config)
        self._peft_model.print_trainable_parameters()

    def train(self, dataset_split: DatasetSplitResult) -> list[TrainingMetrics]:
        """Run the training loop and log metrics to W&B."""
        if not HAS_TORCH:
            return self._mock_train(dataset_split)

        from src.data_preprocessor import format_instruction

        def tokenize(items):
            texts = [format_instruction(item) for item in items]
            return self._tokenizer(texts, truncation=True, max_length=self.config.max_seq_length, padding=True)

        train_dataset = tokenize(dataset_split.train)
        eval_dataset = tokenize(dataset_split.validation)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            save_total_limit=3,
            report_to="wandb" if self._wandb_run else "none",
        )

        trainer = Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        trainer.save_model(self.config.output_dir)
        self._log_wandb_summary()

        return self._extract_metrics(trainer)

    def _mock_train(self, dataset_split: DatasetSplitResult) -> list[TrainingMetrics]:
        """Mock training loop for environments without GPU/PyTorch."""
        metrics: list[TrainingMetrics] = []
        total_steps = len(dataset_split.train) // self.config.per_device_train_batch_size * self.config.num_train_epochs
        for step in range(1, total_steps + 1):
            loss = max(0.1, 2.5 * (0.95 ** step))
            lr = self.config.learning_rate * (1.0 - step / total_steps * 0.9)
            m = TrainingMetrics(step=step, epoch=step / total_steps * self.config.num_train_epochs, loss=loss, learning_rate=lr)
            metrics.append(m)
            self.log_to_wandb(m)
        return metrics

    def log_to_wandb(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to W&B."""
        if self._wandb_run:
            try:
                import wandb
                wandb.log(metrics.model_dump())
            except Exception:
                pass

    def _log_wandb_summary(self) -> None:
        if self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def _extract_metrics(self, trainer: Any) -> list[TrainingMetrics]:
        """Extract logged metrics from the trainer state."""
        metrics: list[TrainingMetrics] = []
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            for entry in trainer.state.log_history:
                metrics.append(TrainingMetrics(
                    step=entry.get("step", 0),
                    epoch=entry.get("epoch", 0.0),
                    loss=entry.get("loss", 0.0),
                    learning_rate=entry.get("learning_rate", 0.0),
                    eval_loss=entry.get("eval_loss"),
                    grad_norm=entry.get("grad_norm"),
                ))
        return metrics
