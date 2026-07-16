"""Typed contracts for the LoRA fine-tuning pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TrainingConfig(BaseModel):
    """Configuration for a LoRA fine-tuning run."""

    base_model: str = Field("meta-llama/Llama-3.1-8B", description="HuggingFace model ID")
    output_dir: str = Field("./output/lora_adapter")
    lora_r: int = Field(16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        ["q_proj", "v_proj"],
        description="Linear attention modules to apply LoRA to",
    )
    bias: str = Field("none", description="none | all | lora_only")
    task_type: str = Field("CAUSAL_LM")
    learning_rate: float = Field(2e-4, gt=0)
    num_train_epochs: int = Field(3, ge=1, le=100)
    per_device_train_batch_size: int = Field(4, ge=1)
    per_device_eval_batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(0.01, ge=0.0)
    max_seq_length: int = Field(2048, ge=128)
    save_steps: int = Field(500, ge=1)
    eval_steps: int = Field(500, ge=1)
    logging_steps: int = Field(10, ge=1)
    use_qlora: bool = Field(True, description="Use 4-bit quantization for QLoRA")
    quantization_config: dict[str, Any] = Field(
        default_factory=lambda: {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_type": "nf4"}
    )
    wandb_project: str = Field("lora-finetuning")
    wandb_run_name: str = Field("")

    def to_peft_config(self) -> dict[str, Any]:
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


class DatasetItem(BaseModel):
    """A single training example."""

    id: str
    text: str = Field(..., min_length=1)
    label: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    split: DatasetSplit | None = None


class DatasetSplitResult(BaseModel):
    """Result of splitting a dataset into train/val/test."""

    train: list[DatasetItem]
    validation: list[DatasetItem]
    test: list[DatasetItem]
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    total_items: int
    leakage_check_passed: bool


class TrainingMetrics(BaseModel):
    """Metrics logged during training."""

    step: int
    epoch: float
    loss: float
    learning_rate: float
    eval_loss: float | None = None
    grad_norm: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BenchmarkResult(BaseModel):
    """Result of a blind LLM-as-judge benchmark."""

    benchmark_name: str
    model_name: str
    scores: list[float] = Field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    num_samples: int = 0
    is_blind: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForgettingCheckResult(BaseModel):
    """Result of catastrophic forgetting evaluation on general benchmarks."""

    benchmark_name: str
    base_model_score: float
    finetuned_model_score: float
    performance_delta: float
    forgetting_detected: bool
    threshold: float = Field(0.05, description="Performance drop threshold for forgetting detection")
