"""Plotting utilities for evaluation outputs."""

from __future__ import annotations

import os
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
import matplotlib.pyplot as plt
import pandas as pd


def save_comparison_plot(csv_path: str | Path, output_path: str | Path) -> None:
    """Create a comparison plot for waiting time and delay metrics."""

    data = pd.read_csv(csv_path)
    grouped = data.groupby("scheduler_name", as_index=False).mean(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(grouped["scheduler_name"], grouped["average_waiting_time"], color="#1f77b4")
    axes[0].set_title("Average Waiting Time")
    axes[0].set_ylabel("Minutes")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(grouped["scheduler_name"], grouped["total_propagated_delay"], color="#ff7f0e")
    axes[1].set_title("Total Propagated Delay")
    axes[1].set_ylabel("Minutes")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
