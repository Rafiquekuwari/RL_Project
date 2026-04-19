"""Heuristic baselines for comparison against PPO."""

from __future__ import annotations

from typing import Callable

from railsched_rl.data_models import Platform, Train

PolicyFn = Callable[[Train, list[Train], list[Platform], int], int]


def fcfs_scheduler(train: Train, _: list[Train], platforms: list[Platform], current_time: int) -> int:
    """Assign the first compatible free platform, otherwise hold."""

    for platform in platforms:
        if platform.maintenance_block:
            continue
        if platform.compatibility_type != train.platform_compatibility:
            continue
        if platform.occupied_until <= current_time:
            return platform.platform_id
    return len(platforms)


def earliest_free_scheduler(
    train: Train,
    _: list[Train],
    platforms: list[Platform],
    current_time: int,
) -> int:
    """Prefer the free compatible platform with the smallest remaining time."""

    candidates = [
        platform
        for platform in platforms
        if (not platform.maintenance_block)
        and platform.compatibility_type == train.platform_compatibility
        and platform.occupied_until <= current_time
    ]
    if not candidates:
        return len(platforms)
    return min(candidates, key=lambda platform: (platform.occupied_until, platform.platform_id)).platform_id


def priority_aware_scheduler(
    train: Train,
    waiting_queue: list[Train],
    platforms: list[Platform],
    current_time: int,
) -> int:
    """Hold low-priority trains briefly when a higher-priority train is already waiting."""

    higher_priority_waiting = any(candidate.priority > train.priority for candidate in waiting_queue)
    if higher_priority_waiting and train.priority <= 1:
        return len(platforms)
    return earliest_free_scheduler(train, waiting_queue, platforms, current_time)


BASELINE_POLICIES: dict[str, PolicyFn] = {
    "fcfs": fcfs_scheduler,
    "earliest_free": earliest_free_scheduler,
    "priority_aware": priority_aware_scheduler,
}
