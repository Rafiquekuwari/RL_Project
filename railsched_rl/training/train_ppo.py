"""Train a PPO agent for railway platform scheduling."""

from __future__ import annotations

import argparse
from pathlib import Path

from railsched_rl import load_config
from railsched_rl.envs.platform_env import RailPlatformEnv
from railsched_rl.utils.io import ensure_dir
from railsched_rl.utils.seeding import set_global_seed
from railsched_rl.utils.training_metrics import TrainingMetricsCallback, save_training_plots


def training_artifact_paths(config_path: str) -> tuple[Path, Path, Path]:
    """Return metric CSV, training plot directory, and TensorBoard directory."""

    config = load_config(config_path)
    metrics_csv = Path(config.logging.outputs_dir) / "training" / "training_metrics.csv"
    plots_dir = Path(config.logging.plots_dir) / "training"
    tensorboard_dir = Path(config.logging.tensorboard_dir)
    return metrics_csv, plots_dir, tensorboard_dir


def train(config_path: str) -> Path:
    """Train PPO, write visual training artifacts, and return the saved model path."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = load_config(config_path)
    set_global_seed(config.seed)
    metrics_csv = ensure_dir(Path(config.logging.outputs_dir) / "training") / "training_metrics.csv"
    training_plots_dir = ensure_dir(Path(config.logging.plots_dir) / "training")
    tensorboard_dir = ensure_dir(config.logging.tensorboard_dir)

    def make_env() -> Monitor:
        env = RailPlatformEnv(config=config, episode_seed=config.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    metrics_callback = TrainingMetricsCallback(csv_path=metrics_csv)
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        ent_coef=config.training.ent_coef,
        clip_range=config.training.clip_range,
        verbose=1,
        seed=config.seed,
        tensorboard_log=str(tensorboard_dir),
    )
    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=metrics_callback,
        progress_bar=False,
        tb_log_name="ppo_railsched",
    )

    models_dir = ensure_dir(config.logging.models_dir)
    model_path = models_dir / "ppo_railsched.zip"
    model.save(str(model_path))
    save_training_plots(metrics_csv, training_plots_dir)
    vec_env.close()
    return model_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Train PPO for RailSched-RL.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    model_path = train(args.config)
    _, training_plots_dir, tensorboard_dir = training_artifact_paths(args.config)
    print(f"Saved PPO model to {model_path}")
    print(f"Training plots saved to {training_plots_dir}")
    print(f"TensorBoard logs saved to {tensorboard_dir}")
    print(f"Launch TensorBoard with: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()
