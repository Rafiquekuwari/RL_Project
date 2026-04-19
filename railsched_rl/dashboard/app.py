"""Streamlit dashboard for RailSched-RL outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

ROOT_DIR = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = ROOT_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml


DEFAULT_METRICS_PATH = ROOT_DIR / "outputs" / "comparison_metrics.csv"
DEFAULT_ASSIGNMENTS_PATH = ROOT_DIR / "logs" / "assignment_records.csv"
DEFAULT_PLOT_PATH = ROOT_DIR / "plots" / "comparison_plot.png"
DEFAULT_TRAINING_METRICS_PATH = ROOT_DIR / "outputs" / "training" / "training_metrics.csv"
DEFAULT_TRAINING_PLOTS_DIR = ROOT_DIR / "plots" / "training"
DEFAULT_TENSORBOARD_DIR = ROOT_DIR / "logs" / "tensorboard"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"


@st.cache_data
def load_data(metrics_path: str, assignments_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load evaluation CSVs."""

    return pd.read_csv(metrics_path), pd.read_csv(assignments_path)


@st.cache_data
def load_training_metrics(metrics_path: str) -> pd.DataFrame:
    """Load training metrics CSV."""

    return pd.read_csv(metrics_path)


@st.cache_data
def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load dashboard config summary from YAML."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def render_metrics_panel(metrics_df: pd.DataFrame) -> None:
    """Render top-level evaluation metrics."""

    st.subheader("Metrics Summary")
    grouped = metrics_df.groupby("scheduler_name", as_index=False).mean(numeric_only=True)
    cols = st.columns(max(len(grouped), 1))
    for col, row in zip(cols, grouped.itertuples(index=False), strict=False):
        with col:
            st.metric(f"{row.scheduler_name} wait", f"{row.average_waiting_time:.2f} min")
            st.metric(f"{row.scheduler_name} util", f"{row.average_utilization:.2%}")


def render_timeline(assignments_df: pd.DataFrame) -> None:
    """Render a simple platform timeline chart."""

    st.subheader("Train Schedule Timeline")
    fig, ax = plt.subplots(figsize=(12, 5))
    for _, row in assignments_df.iterrows():
        ax.barh(
            y=f"Platform {int(row['platform_id'])}",
            width=row["departure_time"] - row["service_start"],
            left=row["service_start"],
            alpha=0.7,
            label=row["train_id"],
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Platform")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)
    plt.close(fig)


def render_platform_occupancy(assignments_df: pd.DataFrame) -> None:
    """Render platform occupancy totals."""

    st.subheader("Platform Occupancy Chart")
    occupancy = (
        assignments_df.assign(duration=assignments_df["departure_time"] - assignments_df["service_start"])
        .groupby("platform_id", as_index=False)["duration"]
        .sum()
    )
    st.bar_chart(occupancy.set_index("platform_id"))


def render_queue_chart(metrics_df: pd.DataFrame) -> None:
    """Render queue length comparison."""

    st.subheader("Queue Length Chart")
    queue_df = metrics_df.groupby("scheduler_name", as_index=False)["max_queue_length"].mean()
    st.line_chart(queue_df.set_index("scheduler_name"))


def render_evaluation_tab(metrics_path: Path, assignments_path: Path, plot_path: Path) -> None:
    """Render the existing evaluation dashboard section."""

    if not metrics_path.exists() or not assignments_path.exists():
        st.warning("Run evaluation first to generate comparison and assignment CSV outputs.")
        st.code("python -m railsched_rl.training.evaluate --config configs/default.yaml")
        return

    metrics_df, assignments_df = load_data(str(metrics_path), str(assignments_path))
    render_metrics_panel(metrics_df)
    render_timeline(assignments_df)
    render_platform_occupancy(assignments_df)
    render_queue_chart(metrics_df)

    if plot_path.exists():
        st.subheader("Evaluation Plot")
        st.image(str(plot_path))
    else:
        st.info(f"Evaluation plot not found at {plot_path}")


def render_training_curve_from_csv(metrics_df: pd.DataFrame, column: str, title: str) -> None:
    """Render a fallback line chart directly from the training metrics CSV."""

    if column not in metrics_df.columns or "timestep" not in metrics_df.columns:
        st.info(f"Column `{column}` is not available in the training metrics CSV.")
        return
    chart_df = metrics_df[["timestep", column]].set_index("timestep")
    st.line_chart(chart_df)
    st.caption(title)


def render_training_plot(
    plot_path: Path,
    metrics_df: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    """Display a saved PNG plot, falling back to a CSV line chart."""

    st.subheader(title)
    if plot_path.exists():
        st.image(str(plot_path), use_column_width=True)
        return
    render_training_curve_from_csv(metrics_df, column, title)


def latest_file(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the newest file in a directory matching one of the patterns."""

    if not directory.exists():
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(directory.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def render_config_summary(config_path: Path) -> None:
    """Render a compact training configuration summary."""

    st.subheader("Config Summary")
    if not config_path.exists():
        st.info(f"Config file not found at {config_path}")
        return

    config = load_yaml_config(str(config_path))
    training = config.get("training", {})
    station = config.get("station", {})
    logging = config.get("logging", {})
    summary = {
        "seed": config.get("seed"),
        "training": training,
        "station": station,
        "tensorboard_dir": logging.get("tensorboard_dir", "logs/tensorboard"),
    }
    st.json(summary)


def render_checkpoint_info(models_dir: Path, tensorboard_dir: Path) -> None:
    """Render latest model and TensorBoard checkpoint/event information."""

    st.subheader("Latest Model Checkpoint")
    latest_model = latest_file(models_dir, ("*.zip",))
    latest_event = latest_file(tensorboard_dir, ("events.out.tfevents.*",))

    cols = st.columns(2)
    with cols[0]:
        if latest_model is None:
            st.info(f"No model checkpoint found in {models_dir}")
        else:
            size_mb = latest_model.stat().st_size / (1024 * 1024)
            st.metric("Model File", latest_model.name)
            st.caption(f"Path: {latest_model}")
            st.caption(f"Size: {size_mb:.2f} MB")

    with cols[1]:
        if latest_event is None:
            st.info(f"No TensorBoard event file found in {tensorboard_dir}")
        else:
            st.metric("TensorBoard Event", latest_event.name)
            st.caption(f"Path: {latest_event}")
            st.code(f"tensorboard --logdir {tensorboard_dir}", language="bash")


def render_training_progress_tab(
    metrics_path: Path,
    plots_dir: Path,
    tensorboard_dir: Path,
    models_dir: Path,
    config_path: Path,
) -> None:
    """Render the Training Progress dashboard section."""

    st.subheader("Training Progress")
    if not metrics_path.exists():
        st.warning("Training metrics CSV not found yet.")
        st.code("python -m railsched_rl.training.train_ppo --config configs/default.yaml")
        st.caption(f"Expected metrics path: {metrics_path}")
        render_config_summary(config_path)
        render_checkpoint_info(models_dir, tensorboard_dir)
        return

    metrics_df = load_training_metrics(str(metrics_path))
    if metrics_df.empty:
        st.info("Training metrics CSV exists but does not contain episode rows yet.")
        render_config_summary(config_path)
        render_checkpoint_info(models_dir, tensorboard_dir)
        return

    latest = metrics_df.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Latest Reward", f"{latest.get('mean_reward', 0.0):.2f}")
    cols[1].metric("Latest Waiting", f"{latest.get('mean_waiting_time', 0.0):.2f} min")
    cols[2].metric("Latest Utilization", f"{latest.get('mean_utilization', 0.0):.2%}")
    cols[3].metric("Latest Conflicts", f"{latest.get('mean_conflicts', 0.0):.2f}")

    plot_specs = [
        (
            plots_dir / "reward_vs_timestep.png",
            "mean_reward",
            "Reward Curve",
        ),
        (
            plots_dir / "waiting_time_vs_timestep.png",
            "mean_waiting_time",
            "Waiting Time Curve",
        ),
        (
            plots_dir / "utilization_vs_timestep.png",
            "mean_utilization",
            "Utilization Curve",
        ),
        (
            plots_dir / "conflicts_vs_timestep.png",
            "mean_conflicts",
            "Conflicts Curve",
        ),
    ]

    left, right = st.columns(2)
    for index, (plot_path, column, title) in enumerate(plot_specs):
        container = left if index % 2 == 0 else right
        with container:
            render_training_plot(plot_path, metrics_df, column, title)

    st.subheader("Training Metrics CSV")
    st.dataframe(metrics_df.tail(25), use_container_width=True)
    render_config_summary(config_path)
    render_checkpoint_info(models_dir, tensorboard_dir)


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="RailSched-RL Dashboard", layout="wide")
    st.title("RailSched-RL: Railway Platform Scheduling Dashboard")

    st.sidebar.header("Evaluation Inputs")
    metrics_path = Path(st.sidebar.text_input("Metrics CSV", value=str(DEFAULT_METRICS_PATH)))
    assignments_path = Path(st.sidebar.text_input("Assignments CSV", value=str(DEFAULT_ASSIGNMENTS_PATH)))
    plot_path = Path(st.sidebar.text_input("Comparison Plot", value=str(DEFAULT_PLOT_PATH)))

    st.sidebar.header("Training Inputs")
    training_metrics_path = Path(
        st.sidebar.text_input("Training Metrics CSV", value=str(DEFAULT_TRAINING_METRICS_PATH))
    )
    training_plots_dir = Path(st.sidebar.text_input("Training Plots Dir", value=str(DEFAULT_TRAINING_PLOTS_DIR)))
    tensorboard_dir = Path(st.sidebar.text_input("TensorBoard Dir", value=str(DEFAULT_TENSORBOARD_DIR)))
    models_dir = Path(st.sidebar.text_input("Models Dir", value=str(DEFAULT_MODELS_DIR)))
    config_path = Path(st.sidebar.text_input("Config YAML", value=str(DEFAULT_CONFIG_PATH)))

    evaluation_tab, training_tab = st.tabs(["Evaluation", "Training Progress"])

    with evaluation_tab:
        render_evaluation_tab(metrics_path, assignments_path, plot_path)

    with training_tab:
        render_training_progress_tab(
            metrics_path=training_metrics_path,
            plots_dir=training_plots_dir,
            tensorboard_dir=tensorboard_dir,
            models_dir=models_dir,
            config_path=config_path,
        )


if __name__ == "__main__":
    main()
