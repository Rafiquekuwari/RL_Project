"""Train PPO for railway platform scheduling with evaluation and plotting."""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from railway_rl.envs import RailwayPlatformEnv
from railsched_rl.config import (
    EvaluationConfig,
    ExperimentConfig,
    GeneratorConfig,
    LoggingConfig,
    StationConfig,
    TrainingConfig,
)


@dataclass(slots=True)
class TrainingArtifacts:
    """Paths produced by the training run."""

    model_path: Path
    best_model_dir: Path
    episode_log_path: Path
    reward_plot_path: Path
    waiting_plot_path: Path
    conflicts_plot_path: Path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(num_platforms: int, timesteps: int, output_dir: Path, seed: int) -> ExperimentConfig:
    """Create a runnable experiment configuration from CLI arguments."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    outputs_dir = output_dir / "outputs"
    for directory in (logs_dir, models_dir, plots_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ExperimentConfig(
        seed=seed,
        generator=GeneratorConfig(
            episode_minutes=360,
            min_trains=18,
            max_trains=30,
            delay_probability=0.45,
            max_delay_minutes=20,
            min_dwell_time=8,
            max_dwell_time=18,
            compatibility_types=[0, 1],
            priority_levels=[1, 2, 3],
        ),
        station=StationConfig(
            num_platforms=num_platforms,
            maintenance_probability=0.10,
            hold_penalty=1.0,
            invalid_action_penalty=4.0,
            assignment_reward=5.0,
            conflict_penalty=3.0,
            wait_penalty_per_minute=0.15,
            propagation_penalty_per_minute=0.25,
            utilization_reward_weight=1.5,
        ),
        training=TrainingConfig(
            total_timesteps=timesteps,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            eval_episodes=5,
        ),
        evaluation=EvaluationConfig(
            episodes=5,
            output_dir=str(outputs_dir),
        ),
        logging=LoggingConfig(
            logs_dir=str(logs_dir),
            models_dir=str(models_dir),
            plots_dir=str(plots_dir),
            outputs_dir=str(outputs_dir),
        ),
    )


def make_env(config: ExperimentConfig, seed: int) -> Monitor:
    """Create a monitored RailwayPlatformEnv instance."""

    env = RailwayPlatformEnv(config=config, episode_seed=seed)
    return Monitor(env)


class EpisodeStatsCallback(BaseCallback):
    """Collect episode-level reward and operational metrics during training."""

    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.rows: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            metrics = info.get("episode_metrics")
            if metrics is None:
                continue
            self.rows.append(
                {
                    "episode_id": int(metrics.episode_id),
                    "reward": float(metrics.total_reward),
                    "waiting_time": float(metrics.total_waiting_time),
                    "conflicts": int(metrics.conflicts),
                    "invalid_actions": int(metrics.invalid_actions),
                    "utilization": float(metrics.average_utilization),
                }
            )
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "episode_id",
                "reward",
                "waiting_time",
                "conflicts",
                "invalid_actions",
                "utilization",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


def save_training_plots(csv_path: Path, plots_dir: Path) -> tuple[Path, Path, Path]:
    """Save training plots for reward, waiting time, and conflicts."""

    plots_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError("Training log is empty; cannot plot training metrics.")

    outputs: list[Path] = []
    for metric, color in (
        ("reward", "#1f77b4"),
        ("waiting_time", "#ff7f0e"),
        ("conflicts", "#d62728"),
    ):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data["episode_id"], data[metric], color=color, linewidth=1.8)
        ax.set_title(f"Training {metric.replace('_', ' ').title()}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(alpha=0.3)
        output_path = plots_dir / f"{metric}_plot.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs.append(output_path)

    return outputs[0], outputs[1], outputs[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train PPO on RailwayPlatformEnv.")
    parser.add_argument("--timesteps", type=int, default=15000, help="Total PPO timesteps.")
    parser.add_argument("--num-platforms", type=int, default=4, help="Number of station platforms.")
    parser.add_argument("--output-dir", type=str, default="railway_outputs", help="Artifact output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per PPO update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=2000,
        help="Evaluation frequency in timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes per evaluation pass.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> TrainingArtifacts:
    """Run PPO training and save artifacts."""

    output_dir = Path(args.output_dir).resolve()
    config = build_config(
        num_platforms=args.num_platforms,
        timesteps=args.timesteps,
        output_dir=output_dir,
        seed=args.seed,
    )
    config.training.learning_rate = args.learning_rate
    config.training.n_steps = args.n_steps
    config.training.batch_size = args.batch_size
    config.training.gamma = args.gamma
    config.training.gae_lambda = args.gae_lambda
    config.training.ent_coef = args.ent_coef
    config.training.clip_range = args.clip_range
    config.training.eval_episodes = args.eval_episodes

    seed_everything(config.seed)

    train_env = DummyVecEnv([lambda: make_env(config, config.seed)])
    eval_env = DummyVecEnv([lambda: make_env(config, config.seed + 10_000)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        seed=config.seed,
        verbose=1,
    )

    logs_dir = Path(config.logging.logs_dir)
    models_dir = Path(config.logging.models_dir)
    plots_dir = Path(config.logging.plots_dir)
    episode_log_path = logs_dir / "training_episode_metrics.csv"
    best_model_dir = models_dir / "best_model"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(logs_dir / "eval_logs"),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    stats_callback = EpisodeStatsCallback(csv_path=episode_log_path)

    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=CallbackList([stats_callback, eval_callback]),
        progress_bar=False,
    )

    model_path = models_dir / "ppo_final_model.zip"
    model.save(str(model_path))
    reward_plot, waiting_plot, conflicts_plot = save_training_plots(
        csv_path=episode_log_path,
        plots_dir=plots_dir,
    )

    train_env.close()
    eval_env.close()

    return TrainingArtifacts(
        model_path=model_path,
        best_model_dir=best_model_dir,
        episode_log_path=episode_log_path,
        reward_plot_path=reward_plot,
        waiting_plot_path=waiting_plot,
        conflicts_plot_path=conflicts_plot,
    )


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    artifacts = train(args)
    print("Training completed.")
    print(f"Final model: {artifacts.model_path}")
    print(f"Best model directory: {artifacts.best_model_dir}")
    print(f"Episode log: {artifacts.episode_log_path}")
    print(f"Reward plot: {artifacts.reward_plot_path}")
    print(f"Waiting plot: {artifacts.waiting_plot_path}")
    print(f"Conflicts plot: {artifacts.conflicts_plot_path}")


if __name__ == "__main__":
    main()
