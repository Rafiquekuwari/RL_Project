"""Configuration helpers for RailSched-RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class GeneratorConfig:
    """Configuration for synthetic train generation."""

    episode_minutes: int
    min_trains: int
    max_trains: int
    delay_probability: float
    max_delay_minutes: int
    min_dwell_time: int
    max_dwell_time: int
    compatibility_types: list[int]
    priority_levels: list[int]


@dataclass(slots=True)
class StationConfig:
    """Configuration for the single-station simulator."""

    num_platforms: int
    maintenance_probability: float
    hold_penalty: float
    invalid_action_penalty: float
    assignment_reward: float
    conflict_penalty: float
    wait_penalty_per_minute: float
    propagation_penalty_per_minute: float
    utilization_reward_weight: float


@dataclass(slots=True)
class TrainingConfig:
    """PPO training configuration."""

    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    clip_range: float
    eval_episodes: int


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation configuration."""

    episodes: int
    output_dir: str


@dataclass(slots=True)
class LoggingConfig:
    """Output locations for logs and artifacts."""

    logs_dir: str
    models_dir: str
    plots_dir: str
    outputs_dir: str
    tensorboard_dir: str = "logs/tensorboard"


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int
    generator: GeneratorConfig
    station: StationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig


def _build_config(raw_config: dict[str, Any]) -> ExperimentConfig:
    """Convert a dictionary into a typed experiment configuration."""

    return ExperimentConfig(
        seed=int(raw_config["seed"]),
        generator=GeneratorConfig(**raw_config["generator"]),
        station=StationConfig(**raw_config["station"]),
        training=TrainingConfig(**raw_config["training"]),
        evaluation=EvaluationConfig(**raw_config["evaluation"]),
        logging=LoggingConfig(**raw_config["logging"]),
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return _build_config(raw_config)
