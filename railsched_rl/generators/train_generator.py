"""Synthetic train arrival generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from railsched_rl.config import GeneratorConfig
from railsched_rl.data_models import Train


def generate_trains(config: GeneratorConfig, seed: int) -> list[Train]:
    """Create a sorted list of synthetic trains for one episode."""

    rng = np.random.default_rng(seed)
    num_trains = int(rng.integers(config.min_trains, config.max_trains + 1))
    arrival_candidates = np.sort(rng.integers(0, config.episode_minutes, size=num_trains))

    trains: list[Train] = []
    for index, arrival_time in enumerate(arrival_candidates):
        dwell_time = int(rng.integers(config.min_dwell_time, config.max_dwell_time + 1))
        delay_minutes = 0
        if rng.random() < config.delay_probability:
            delay_minutes = int(rng.integers(1, config.max_delay_minutes + 1))

        compatibility = int(rng.choice(np.array(config.compatibility_types, dtype=int)))
        priority = int(rng.choice(np.array(config.priority_levels, dtype=int)))
        scheduled_departure = int(arrival_time + dwell_time)
        trains.append(
            Train(
                train_id=f"T{index:03d}",
                scheduled_arrival=int(arrival_time),
                scheduled_departure=scheduled_departure,
                dwell_time=dwell_time,
                priority=priority,
                platform_compatibility=compatibility,
                delay_minutes=delay_minutes,
            )
        )
    trains.sort(key=lambda train: train.actual_arrival)
    return trains


def queue_summary(waiting_queue: Sequence[Train], max_items: int = 3) -> list[float]:
    """Create a compact fixed-length summary of the waiting queue."""

    summary: list[float] = []
    for train in waiting_queue[:max_items]:
        summary.extend(
            [
                float(train.actual_arrival),
                float(train.dwell_time),
                float(train.priority),
                float(train.platform_compatibility),
            ]
        )

    while len(summary) < max_items * 4:
        summary.append(0.0)
    return summary
