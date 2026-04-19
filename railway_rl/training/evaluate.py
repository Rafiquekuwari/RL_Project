"""Evaluate a trained PPO model against baseline railway schedulers."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
matplotlib.use("Agg")

import pandas as pd
from stable_baselines3 import PPO

from railway_rl.envs import RailwayPlatformEnv
from railway_rl.schedulers import EarliestFreeScheduler, FCFSScheduler, PriorityAwareScheduler
from railway_rl.training.train_ppo import build_config, seed_everything
from railsched_rl.data_models import AssignmentRecord, EpisodeMetrics


@dataclass(slots=True)
class SchedulerRunResult:
    """Flat evaluation row for CSV export."""

    episode_id: int
    scheduler_name: str
    total_reward: float
    total_waiting_time: float
    average_waiting_time: float
    total_propagated_delay: float
    assignments: int
    invalid_actions: int
    conflicts: int
    average_utilization: float
    max_queue_length: int


@dataclass(slots=True)
class AssignmentRow:
    """Assignment record exported for dashboard timelines and tables."""

    episode_id: int
    scheduler_name: str
    train_id: str
    platform_id: int
    arrival_time: int
    service_start: int
    departure_time: int
    waiting_time: int
    propagated_delay: int
    priority: int


@dataclass(slots=True)
class QueueHistoryRow:
    """Queue history exported for interactive dashboard charts."""

    episode_id: int
    scheduler_name: str
    step_index: int
    current_time: int
    queue_length: int
    utilization: float
    conflicts: int


def _assignment_rows(
    episode_id: int,
    scheduler_name: str,
    assignments: list[AssignmentRecord],
) -> list[AssignmentRow]:
    """Convert assignment records into flat CSV rows."""

    return [
        AssignmentRow(
            episode_id=episode_id,
            scheduler_name=scheduler_name,
            train_id=record.train_id,
            platform_id=record.platform_id,
            arrival_time=record.arrival_time,
            service_start=record.service_start,
            departure_time=record.departure_time,
            waiting_time=record.waiting_time,
            propagated_delay=record.propagated_delay,
            priority=record.priority,
        )
        for record in assignments
    ]


def run_scheduler_episode(
    env: RailwayPlatformEnv,
    scheduler_name: str,
    scheduler: object,
) -> tuple[SchedulerRunResult, list[AssignmentRow], list[QueueHistoryRow]]:
    """Run one baseline scheduler episode."""

    _, _ = env.reset()
    terminated = False
    truncated = False
    queue_history: list[QueueHistoryRow] = []
    step_index = 0
    while not terminated and not truncated:
        if env.current_train is None:
            break
        action = scheduler.choose_action(env, env.current_train)
        _, _, terminated, truncated, info = env.step(action)
        queue_history.append(
            QueueHistoryRow(
                episode_id=env.episode_index if env.episode_index > 0 else 1,
                scheduler_name=scheduler_name,
                step_index=step_index,
                current_time=int(info.get("current_time", env.current_time)),
                queue_length=int(info.get("queue_length", len(env.waiting_queue))),
                utilization=float(info.get("utilization", 0.0)),
                conflicts=int(info.get("conflicts", 0)),
            )
        )
        step_index += 1

    metrics = env.get_episode_metrics(scheduler_name=scheduler_name, episode_id=env.episode_index)
    assignment_rows = _assignment_rows(
        episode_id=metrics.episode_id,
        scheduler_name=scheduler_name,
        assignments=list(env.simulator.assignment_records),
    )
    for row in queue_history:
        row.episode_id = metrics.episode_id
    return SchedulerRunResult(**asdict(metrics)), assignment_rows, queue_history


def run_ppo_episode(
    env: RailwayPlatformEnv,
    model: PPO,
) -> tuple[SchedulerRunResult, list[AssignmentRow], list[QueueHistoryRow]]:
    """Run one PPO evaluation episode."""

    observation, _ = env.reset()
    terminated = False
    truncated = False
    queue_history: list[QueueHistoryRow] = []
    step_index = 0
    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))
        queue_history.append(
            QueueHistoryRow(
                episode_id=env.episode_index if env.episode_index > 0 else 1,
                scheduler_name="ppo",
                step_index=step_index,
                current_time=int(info.get("current_time", env.current_time)),
                queue_length=int(info.get("queue_length", len(env.waiting_queue))),
                utilization=float(info.get("utilization", 0.0)),
                conflicts=int(info.get("conflicts", 0)),
            )
        )
        step_index += 1

    metrics: EpisodeMetrics = env.get_episode_metrics(scheduler_name="ppo", episode_id=env.episode_index)
    assignment_rows = _assignment_rows(
        episode_id=metrics.episode_id,
        scheduler_name="ppo",
        assignments=list(env.simulator.assignment_records),
    )
    for row in queue_history:
        row.episode_id = metrics.episode_id
    return SchedulerRunResult(**asdict(metrics)), assignment_rows, queue_history


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by scheduler."""

    return results_df.groupby("scheduler_name", as_index=False).mean(numeric_only=True)


def print_summary(summary_df: pd.DataFrame) -> None:
    """Print a readable evaluation summary."""

    print("\nEvaluation Summary")
    print("-" * 80)
    for row in summary_df.itertuples(index=False):
        print(
            f"{row.scheduler_name:>15} | "
            f"reward={row.total_reward:>8.2f} | "
            f"avg_wait={row.average_waiting_time:>6.2f} | "
            f"delay={row.total_propagated_delay:>7.2f} | "
            f"conflicts={row.conflicts:>5.2f} | "
            f"util={row.average_utilization:>5.2f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate PPO against baseline schedulers.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained PPO model zip.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--num-platforms", type=int, default=4, help="Number of station platforms.")
    parser.add_argument("--output-dir", type=str, default="railway_outputs", help="Directory for CSV outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Path:
    """Run evaluation episodes and export a comparison CSV."""

    output_dir = Path(args.output_dir).resolve()
    config = build_config(
        num_platforms=args.num_platforms,
        timesteps=1,
        output_dir=output_dir,
        seed=args.seed,
    )
    config.evaluation.episodes = args.episodes
    seed_everything(config.seed)

    model = PPO.load(args.model_path)

    schedulers = [
        ("fcfs", FCFSScheduler()),
        ("earliest_free", EarliestFreeScheduler()),
        ("priority_aware", PriorityAwareScheduler()),
    ]

    rows: list[SchedulerRunResult] = []
    assignment_rows: list[AssignmentRow] = []
    queue_history_rows: list[QueueHistoryRow] = []
    for episode in range(args.episodes):
        episode_seed = config.seed + episode
        for scheduler_name, scheduler in schedulers:
            env = RailwayPlatformEnv(config=config, episode_seed=episode_seed)
            result, episode_assignments, episode_queue_history = run_scheduler_episode(
                env=env,
                scheduler_name=scheduler_name,
                scheduler=scheduler,
            )
            rows.append(result)
            assignment_rows.extend(episode_assignments)
            queue_history_rows.extend(episode_queue_history)
            env.close()

        env = RailwayPlatformEnv(config=config, episode_seed=episode_seed)
        result, episode_assignments, episode_queue_history = run_ppo_episode(env=env, model=model)
        rows.append(result)
        assignment_rows.extend(episode_assignments)
        queue_history_rows.extend(episode_queue_history)
        env.close()

    results_df = pd.DataFrame(asdict(row) for row in rows)
    output_csv = output_dir / "outputs" / "comparison_results.csv"
    assignments_csv = output_dir / "outputs" / "assignment_records.csv"
    queue_history_csv = output_dir / "outputs" / "queue_history.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    pd.DataFrame(asdict(row) for row in assignment_rows).to_csv(assignments_csv, index=False)
    pd.DataFrame(asdict(row) for row in queue_history_rows).to_csv(queue_history_csv, index=False)

    summary_df = summarize_results(results_df)
    print_summary(summary_df)
    print(f"\nComparison CSV saved to: {output_csv}")
    print(f"Assignment CSV saved to: {assignments_csv}")
    print(f"Queue history CSV saved to: {queue_history_csv}")
    return output_csv


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
