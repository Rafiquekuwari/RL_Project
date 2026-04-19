"""Evaluate PPO and heuristic schedulers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from railsched_rl import load_config
from railsched_rl.baselines.heuristics import BASELINE_POLICIES, PolicyFn
from railsched_rl.data_models import AssignmentRecord, EpisodeMetrics
from railsched_rl.envs.platform_env import RailPlatformEnv
from railsched_rl.utils.io import ensure_dir
from railsched_rl.utils.logging_utils import write_assignments, write_episode_metrics
from railsched_rl.utils.plotting import save_comparison_plot
from railsched_rl.utils.seeding import set_global_seed


def run_policy_episode(
    env: RailPlatformEnv,
    policy_name: str,
    policy_fn: PolicyFn,
    render: bool = False,
) -> tuple[EpisodeMetrics, list[AssignmentRecord]]:
    """Run one heuristic episode."""

    _, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        if env.current_train is None:
            break
        if render:
            env.render()
        action = policy_fn(
            deepcopy(env.current_train),
            deepcopy(env.waiting_queue[1:]),
            deepcopy(env.simulator.platforms),
            env.current_time,
        )
        _, _, done, truncated, _ = env.step(action)

    metrics = env.get_episode_metrics(policy_name, env.episode_index)
    assignments = deepcopy(env.simulator.assignment_records)
    return metrics, assignments


def run_ppo_episode(
    env: RailPlatformEnv,
    model_path: str | Path,
    render: bool = False,
) -> tuple[EpisodeMetrics, list[AssignmentRecord]]:
    """Run one PPO-controlled episode."""

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    observation, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        if render:
            env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, truncated, _ = env.step(int(action))
    metrics = env.get_episode_metrics("ppo", env.episode_index)
    assignments = deepcopy(env.simulator.assignment_records)
    return metrics, assignments


def evaluate(
    config_path: str,
    model_path: str | None = None,
    episodes: int | None = None,
    render: bool = False,
) -> dict[str, Path]:
    """Run evaluation for heuristics and PPO, then save all artifacts."""

    config = load_config(config_path)
    set_global_seed(config.seed)
    if episodes is not None:
        config.evaluation.episodes = episodes

    outputs_dir = ensure_dir(config.logging.outputs_dir)
    logs_dir = ensure_dir(config.logging.logs_dir)
    plots_dir = ensure_dir(config.logging.plots_dir)
    models_dir = ensure_dir(config.logging.models_dir)

    resolved_model_path = Path(model_path) if model_path else models_dir / "ppo_railsched.zip"

    metrics_rows: list[EpisodeMetrics] = []
    assignment_rows: list[AssignmentRecord] = []

    for episode in range(config.evaluation.episodes):
        episode_seed = config.seed + episode
        for policy_name, policy_fn in BASELINE_POLICIES.items():
            env = RailPlatformEnv(config=config, episode_seed=episode_seed)
            metrics, assignments = run_policy_episode(env, policy_name, policy_fn, render=render)
            metrics_rows.append(metrics)
            assignment_rows.extend(assignments)

        env = RailPlatformEnv(config=config, episode_seed=episode_seed)
        metrics, assignments = run_ppo_episode(env, resolved_model_path, render=render)
        metrics_rows.append(metrics)
        assignment_rows.extend(assignments)

    metrics_csv = outputs_dir / "comparison_metrics.csv"
    assignments_csv = logs_dir / "assignment_records.csv"
    plot_path = plots_dir / "comparison_plot.png"

    write_episode_metrics(metrics_csv, metrics_rows)
    write_assignments(assignments_csv, assignment_rows)
    save_comparison_plot(metrics_csv, plot_path)

    return {
        "metrics_csv": metrics_csv,
        "assignments_csv": assignments_csv,
        "plot_path": plot_path,
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Evaluate RailSched-RL schedulers.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a trained PPO model.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for evaluation.episodes from the YAML config.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation episodes.",
    )
    args = parser.parse_args()
    outputs = evaluate(
        config_path=args.config,
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
