"""Streamlit dashboard for railway platform scheduling evaluation outputs."""

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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Railway RL Dashboard",
    page_icon="🚉",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #f6f8fb 0%, #eef3f7 100%);
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d9e2ec;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
</style>
"""


def discover_csv_candidates(root: Path) -> list[Path]:
    """Discover likely evaluation CSV files in the project tree."""

    preferred_names = {
        "comparison_results.csv",
        "comparison_metrics.csv",
    }
    candidates = [
        path
        for path in root.rglob("*.csv")
        if path.name in preferred_names and ".venv" not in path.parts
    ]
    return sorted(candidates)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    """Load a CSV from a filesystem path."""

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV from uploaded bytes."""

    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def resolve_companion_csv(primary_path: Path, filename: str) -> Path | None:
    """Try common locations for companion CSV files."""

    candidates = [
        primary_path.with_name(filename),
        primary_path.parent / filename,
        primary_path.parent.parent / filename,
        primary_path.parent.parent / "outputs" / filename,
        primary_path.parent.parent / "logs" / filename,
        PROJECT_ROOT / "outputs" / filename,
        PROJECT_ROOT / "logs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_scheduler_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply presentation-friendly scheduler labels."""

    display_map = {
        "ppo": "PPO",
        "fcfs": "FCFS",
        "earliest_free": "Earliest-Free",
        "priority_aware": "Priority-Aware",
        "earliestfree": "Earliest-Free",
        "priorityaware": "Priority-Aware",
    }
    normalized = df.copy()
    if "scheduler_name" in normalized.columns:
        normalized["scheduler_name"] = normalized["scheduler_name"].astype(str).str.strip().str.lower()
        normalized["scheduler_label"] = normalized["scheduler_name"].map(display_map).fillna(
            normalized["scheduler_name"].str.replace("_", " ").str.title()
        )
    return normalized


def summary_cards(results_df: pd.DataFrame) -> None:
    """Render headline KPI cards."""

    avg_wait = results_df["average_waiting_time"].mean() if "average_waiting_time" in results_df else 0.0
    avg_delay = results_df["total_propagated_delay"].mean() if "total_propagated_delay" in results_df else 0.0
    conflicts = results_df["conflicts"].sum() if "conflicts" in results_df else 0.0
    utilization = results_df["average_utilization"].mean() if "average_utilization" in results_df else 0.0
    throughput = results_df["assignments"].mean() if "assignments" in results_df else 0.0

    cols = st.columns(5)
    cols[0].metric("Average Waiting Time", f"{avg_wait:.2f} min")
    cols[1].metric("Average Delay", f"{avg_delay:.2f} min")
    cols[2].metric("Conflict Count", f"{conflicts:.0f}")
    cols[3].metric("Utilization", f"{utilization:.2%}")
    cols[4].metric("Throughput", f"{throughput:.2f} trains/episode")


def comparison_chart(results_df: pd.DataFrame) -> None:
    """Render an interactive scheduler comparison chart."""

    grouped = (
        results_df.groupby("scheduler_label", as_index=False)
        .mean(numeric_only=True)
        .sort_values("average_waiting_time", ascending=True)
    )
    fig = px.bar(
        grouped,
        x="scheduler_label",
        y=["average_waiting_time", "total_propagated_delay", "average_utilization", "assignments"],
        barmode="group",
        title="Scheduler Comparison",
        labels={
            "scheduler_label": "Scheduler",
            "value": "Metric Value",
            "variable": "Metric",
        },
        color_discrete_sequence=["#0f766e", "#f97316", "#2563eb", "#7c3aed"],
    )
    fig.update_layout(
        height=420,
        legend_title_text="Metric",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def occupancy_timeline(assignments_df: pd.DataFrame) -> None:
    """Render a Plotly occupancy timeline if assignment data is available."""

    if assignments_df.empty:
        st.info("Platform occupancy timeline becomes available when `assignment_records.csv` is present.")
        return

    timeline_df = assignments_df.copy()
    timeline_df["platform_label"] = timeline_df["platform_id"].apply(lambda value: f"Platform {value}")
    timeline_df["episode_scheduler"] = (
        "Ep " + timeline_df["episode_id"].astype(str) + " • " + timeline_df["scheduler_label"]
    )
    fig = px.timeline(
        timeline_df.sort_values(["scheduler_label", "platform_id", "service_start"]),
        x_start="service_start",
        x_end="departure_time",
        y="platform_label",
        color="scheduler_label",
        hover_data=[
            "train_id",
            "priority",
            "waiting_time",
            "propagated_delay",
            "episode_id",
        ],
        facet_row="scheduler_label",
        title="Platform Occupancy Timeline",
        color_discrete_sequence=["#2563eb", "#0f766e", "#f97316", "#7c3aed"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(500, 220 * timeline_df["scheduler_label"].nunique()), margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def queue_history_chart(queue_df: pd.DataFrame) -> None:
    """Render queue length over time."""

    if queue_df.empty:
        st.info("Queue-length history becomes available when `queue_history.csv` is present.")
        return

    fig = px.line(
        queue_df.sort_values(["scheduler_label", "episode_id", "current_time"]),
        x="current_time",
        y="queue_length",
        color="scheduler_label",
        line_group="episode_id",
        hover_data=["episode_id", "utilization", "conflicts", "step_index"],
        title="Queue Length Over Time",
        color_discrete_sequence=["#2563eb", "#0f766e", "#f97316", "#7c3aed"],
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def results_table(assignments_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """Render a train-level results table, or fall back to episode-level data."""

    st.subheader("Train-Level Results")
    if not assignments_df.empty:
        display_cols = [
            "scheduler_label",
            "episode_id",
            "train_id",
            "platform_id",
            "arrival_time",
            "service_start",
            "departure_time",
            "waiting_time",
            "propagated_delay",
            "priority",
        ]
        st.dataframe(
            assignments_df[display_cols].sort_values(
                ["scheduler_label", "episode_id", "service_start", "platform_id"]
            ),
            use_container_width=True,
            hide_index=True,
        )
        return

    st.caption("Detailed train assignments were not found, so the table below shows episode-level results.")
    episode_cols = [
        "scheduler_label",
        "episode_id",
        "total_reward",
        "total_waiting_time",
        "average_waiting_time",
        "total_propagated_delay",
        "assignments",
        "conflicts",
        "average_utilization",
    ]
    st.dataframe(
        results_df[episode_cols].sort_values(["scheduler_label", "episode_id"]),
        use_container_width=True,
        hide_index=True,
    )


def scheduler_scatter(results_df: pd.DataFrame) -> None:
    """Render a reward-delay scatter comparison."""

    fig = px.scatter(
        results_df,
        x="average_waiting_time",
        y="total_reward",
        color="scheduler_label",
        size="assignments",
        hover_data=["episode_id", "conflicts", "average_utilization", "total_propagated_delay"],
        title="Reward vs Waiting Time",
        color_discrete_sequence=["#2563eb", "#0f766e", "#f97316", "#7c3aed"],
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def load_detail_csvs(primary_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load optional assignment and queue history CSV files."""

    empty_assignments = pd.DataFrame()
    empty_queue = pd.DataFrame()
    if primary_path is None:
        return empty_assignments, empty_queue

    assignment_path = resolve_companion_csv(primary_path, "assignment_records.csv")
    queue_path = resolve_companion_csv(primary_path, "queue_history.csv")

    assignments_df = pd.read_csv(assignment_path) if assignment_path else empty_assignments
    queue_df = pd.read_csv(queue_path) if queue_path else empty_queue
    return assignments_df, queue_df


def sidebar_source_selector() -> tuple[pd.DataFrame | None, Path | None]:
    """Provide upload and local-file selection options."""

    st.sidebar.header("Data Source")
    discovered = discover_csv_candidates(PROJECT_ROOT)
    discovered_labels = {str(path.relative_to(PROJECT_ROOT)): path for path in discovered}

    source_mode = st.sidebar.radio(
        "Choose input method",
        options=["Select local CSV", "Upload CSV", "Path input"],
        index=0,
    )

    if source_mode == "Select local CSV":
        if not discovered_labels:
            st.sidebar.warning("No evaluation CSVs found in the project tree.")
            return None, None
        selected_label = st.sidebar.selectbox("Available evaluation CSVs", list(discovered_labels.keys()))
        selected_path = discovered_labels[selected_label]
        return load_csv_from_path(str(selected_path)), selected_path

    if source_mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload evaluation CSV", type=["csv"])
        if uploaded is None:
            return None, None
        return load_csv_from_upload(uploaded.getvalue()), None

    path_text = st.sidebar.text_input("CSV path", value=str(PROJECT_ROOT / "outputs" / "comparison_results.csv"))
    candidate = Path(path_text).expanduser()
    if not candidate.exists():
        st.sidebar.warning("The provided path does not exist yet.")
        return None, None
    return load_csv_from_path(str(candidate)), candidate


def main() -> None:
    """Run the dashboard app."""

    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Railway Platform Scheduling Dashboard")
    st.caption(
        "Run with `streamlit run railway_rl/dashboard/app.py` and select the "
        "`comparison_results.csv` file generated by the evaluation script."
    )

    results_df, primary_path = sidebar_source_selector()
    if results_df is None:
        st.info("Select or upload an evaluation CSV to begin.")
        return

    results_df = normalize_scheduler_names(results_df)
    assignments_df, queue_df = load_detail_csvs(primary_path)
    if not assignments_df.empty:
        assignments_df = normalize_scheduler_names(assignments_df)
    if not queue_df.empty:
        queue_df = normalize_scheduler_names(queue_df)

    with st.sidebar:
        st.header("Filters")
        available_schedulers = sorted(results_df["scheduler_label"].unique().tolist())
        selected_schedulers = st.multiselect(
            "Schedulers",
            options=available_schedulers,
            default=available_schedulers,
        )
        available_episodes = sorted(results_df["episode_id"].unique().tolist())
        selected_episodes = st.multiselect(
            "Episodes",
            options=available_episodes,
            default=available_episodes,
        )

    filtered_results = results_df[
        results_df["scheduler_label"].isin(selected_schedulers)
        & results_df["episode_id"].isin(selected_episodes)
    ].copy()
    filtered_assignments = assignments_df.copy()
    filtered_queue = queue_df.copy()
    if not filtered_assignments.empty:
        filtered_assignments = filtered_assignments[
            filtered_assignments["scheduler_label"].isin(selected_schedulers)
            & filtered_assignments["episode_id"].isin(selected_episodes)
        ]
    if not filtered_queue.empty:
        filtered_queue = filtered_queue[
            filtered_queue["scheduler_label"].isin(selected_schedulers)
            & filtered_queue["episode_id"].isin(selected_episodes)
        ]

    if filtered_results.empty:
        st.warning("No rows match the current filter selection.")
        return

    summary_cards(filtered_results)

    col_left, col_right = st.columns([1.3, 1.0])
    with col_left:
        comparison_chart(filtered_results)
    with col_right:
        scheduler_scatter(filtered_results)

    occupancy_timeline(filtered_assignments)
    queue_history_chart(filtered_queue)
    results_table(filtered_assignments, filtered_results)

    st.subheader("Evaluation Summary")
    summary = (
        filtered_results.groupby("scheduler_label", as_index=False)
        .mean(numeric_only=True)
        .sort_values("average_waiting_time")
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if primary_path is not None:
        st.caption(f"Loaded primary CSV: {primary_path}")


if __name__ == "__main__":
    main()
