from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class DataConfig:
    class_order: list[str] | None = None
    pretrain_label: str | None = None
    test_samples_per_class: int = 10


@dataclass(slots=True)
class SimCLRConfig:
    temperature: float = 0.3
    positive_weight: float = 0.8
    negative_weight: float = 0.3
    encoder_lr: float = 1e-4
    projector_lr: float = 1e-3
    projection_hidden_dim: int | None = None
    projection_output_dim: int | None = None
    epochs: int = 500
    early_stop_patience: int = 10
    batch_size: int = 32


@dataclass(slots=True)
class DQNConfig:
    min_epoch: int = 15
    episodes: int = 300
    gamma: float = 0.01
    mini_batch: int = 16
    learn_when_capacity: int = 50
    target_update_interval: int = 500
    memory_capacity: int = 80
    epsilon: float | None = None
    initial_epsilon: float = 0.3
    reward_plus: float = 0.0
    learning_rate: float = 1e-3
    hidden_size: int = 32
    layers: int = 1
    autoencoder_epochs: int = 500
    autoencoder_lr: float = 1e-3
    autoencoder_patience: int = 10
    early_stop: int = 10


@dataclass(slots=True)
class RuntimeConfig:
    model_name: str = "ccrl"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_workers: int = 0
    pin_memory: bool = False
    multi_process: bool = False
    label_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    seq_len: int = 0
    feature_dim: int = 0


@dataclass(slots=True)
class CCRLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
