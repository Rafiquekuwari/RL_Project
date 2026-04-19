"""CSV logging helpers for experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from railsched_rl.data_models import AssignmentRecord, EpisodeMetrics


def write_episode_metrics(path: str | Path, metrics: Iterable[EpisodeMetrics]) -> None:
    """Persist aggregated episode metrics to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(metrics)
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_assignments(path: str | Path, assignments: Iterable[AssignmentRecord]) -> None:
    """Persist assignment records to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(assignments)
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
