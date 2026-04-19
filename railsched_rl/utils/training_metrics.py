"""Training metric collection and plotting utilities."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
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
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(slots=True)
class TrainingMetricRow:
    """One episode-level training metric row."""

    timestep: int
    episode: int
    mean_reward: float
    mean_waiting_time: float
    mean_queue_length: float
    mean_utilization: float
    mean_conflicts: float


class TrainingMetricsCallback(BaseCallback):
    """Collect episode-level environment metrics during PPO training."""

    def __init__(self, csv_path: str | Path) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.rows: list[TrainingMetricRow] = []
        self.episode_count = 0
        self._queue_lengths: list[float] = []
        self._utilizations: list[float] = []

    def _on_training_start(self) -> None:
        """Create the destination directory before training begins."""

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv()

    def _on_step(self) -> bool:
        """Collect step-level context and flush a row when an episode ends."""

        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            self._queue_lengths.append(float(info.get("queue_length", 0.0)))
            self._utilizations.append(float(info.get("utilization", 0.0)))
            episode_metrics = info.get("episode_metrics")
            if episode_metrics is None:
                continue

            self.episode_count += 1
            row = TrainingMetricRow(
                timestep=int(self.num_timesteps),
                episode=self.episode_count,
                mean_reward=float(episode_metrics.total_reward),
                mean_waiting_time=float(episode_metrics.average_waiting_time),
                mean_queue_length=_safe_mean(self._queue_lengths),
                mean_utilization=_safe_mean(self._utilizations),
                mean_conflicts=float(episode_metrics.conflicts),
            )
            self.rows.append(row)
            self._queue_lengths.clear()
            self._utilizations.clear()
            self._write_csv()
        return True

    def _on_training_end(self) -> None:
        """Write a final copy of the training metrics CSV."""

        self._write_csv()

    def _write_csv(self) -> None:
        """Persist all collected rows to CSV."""

        fieldnames = list(TrainingMetricRow.__dataclass_fields__.keys())
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))


def _safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean, or zero for empty input."""

    if not values:
        return 0.0
    return float(fmean(values))


def save_training_plots(csv_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate standard training progress plots from the metrics CSV."""

    metrics_path = Path(csv_path)
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        return []

    data = pd.read_csv(metrics_path)
    if data.empty:
        return []

    plot_specs = [
        ("mean_reward", "reward_vs_timestep.png", "Reward vs Timestep", "Mean Reward"),
        (
            "mean_waiting_time",
            "waiting_time_vs_timestep.png",
            "Waiting Time vs Timestep",
            "Mean Waiting Time",
        ),
        (
            "mean_utilization",
            "utilization_vs_timestep.png",
            "Utilization vs Timestep",
            "Mean Utilization",
        ),
        (
            "mean_conflicts",
            "conflicts_vs_timestep.png",
            "Conflicts vs Timestep",
            "Mean Conflicts",
        ),
    ]

    output_paths: list[Path] = []
    for column, filename, title, ylabel in plot_specs:
        if column not in data.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(data["timestep"], data[column], linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths
