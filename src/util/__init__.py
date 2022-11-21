from typing import Union, List, Optional
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass


from operations import Operations


class Models(Enum):
    CIFAR10 = auto()
    IMAGENET = auto()


@dataclass
class LoggingConfig:
    out_dir: str = "log/default"
    log_every: int = 50

    # Checkpointing
    n_epochs: Optional[int] = None
    selection_metric: str = "val/loss"
    selection_mode: str = "min"


@dataclass
class PLConfig:
    max_epochs: int = 1000
    max_steps: int = -1
    val_every: int = 1

    resume_path: Optional[str] = None
    overfit: float = 0.0


@dataclass
class ModelConfig:
    type: Models = Models.CIFAR10
    operator: Operations = Operations.CONV2D
    init_scheme: str = "flat"
    order: int = 1
    orders: Optional[List[int]] = None
    network: str = "resnet18"
    bit: int = 4


@dataclass
class TrainingConfig:
    lr: float = 0.001
    milestones: Optional[List[int]] = None


@dataclass
class ExperimentConfig:
    name: str = "Default"
    debug: bool = False

    logging: LoggingConfig = LoggingConfig()
    pl: PLConfig = PLConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
