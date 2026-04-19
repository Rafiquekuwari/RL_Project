"""Class-based baseline schedulers for railway platform assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from railsched_rl.data_models import Platform, Train


class HasPlatformState(Protocol):
    """Protocol for the environment state used by scheduler baselines."""

    current_time: int
    waiting_queue: list[Train]
    action_space: object

    @property
    def simulator(self) -> object: ...


def env_num_platforms(env: HasPlatformState) -> int:
    """Infer the number of platform actions from the environment."""

    return int(env.action_space.n - 1)  # type: ignore[attr-defined]


def hold_action(env: HasPlatformState) -> int:
    """Return the discrete HOLD action."""

    return env_num_platforms(env)


def compatible_platforms(
    train: Train,
    platforms: list[Platform],
    current_time: int,
) -> list[Platform]:
    """Filter to compatible platforms that are not blocked by maintenance."""

    return [
        platform
        for platform in platforms
        if (not platform.maintenance_block)
        and platform.compatibility_type == train.platform_compatibility
        and platform.occupied_until <= current_time
    ]


def downstream_blocking_score(
    platform: Platform,
    waiting_queue: list[Train],
    current_time: int,
) -> tuple[int, int, int]:
    """Compute a simple blocking score for platform selection.

    Lower is better:
    - fewer waiting trains require this platform type
    - smaller busy time extension
    - lower platform id for stable tie-breaking
    """

    matching_trains = sum(
        1 for train in waiting_queue if train.platform_compatibility == platform.compatibility_type
    )
    remaining_busy = max(platform.occupied_until - current_time, 0)
    return (matching_trains, remaining_busy, platform.platform_id)


@dataclass(slots=True)
class FCFSScheduler:
    """Assign to the first available compatible platform, otherwise HOLD."""

    name: str = "fcfs"

    def choose_action(self, env: HasPlatformState, current_train: Train) -> int:
        """Return the action for the current train."""

        platforms = compatible_platforms(
            train=current_train,
            platforms=env.simulator.platforms,  # type: ignore[attr-defined]
            current_time=env.current_time,
        )
        if not platforms:
            return hold_action(env)
        return platforms[0].platform_id


@dataclass(slots=True)
class EarliestFreeScheduler:
    """Assign to the compatible platform with the smallest occupied-until value."""

    name: str = "earliest_free"

    def choose_action(self, env: HasPlatformState, current_train: Train) -> int:
        """Return the action for the current train."""

        platforms = compatible_platforms(
            train=current_train,
            platforms=env.simulator.platforms,  # type: ignore[attr-defined]
            current_time=env.current_time,
        )
        if not platforms:
            return hold_action(env)
        return min(
            platforms,
            key=lambda platform: (platform.occupied_until, platform.platform_id),
        ).platform_id


@dataclass(slots=True)
class PriorityAwareScheduler:
    """Prioritize high-priority trains and minimize downstream blocking."""

    high_priority_threshold: int = 3
    name: str = "priority_aware"

    def choose_action(self, env: HasPlatformState, current_train: Train) -> int:
        """Return the action for the current train."""

        higher_priority_waiting = any(
            train.priority > current_train.priority for train in env.waiting_queue[1:]
        )
        if higher_priority_waiting and current_train.priority < self.high_priority_threshold:
            return hold_action(env)

        platforms = compatible_platforms(
            train=current_train,
            platforms=env.simulator.platforms,  # type: ignore[attr-defined]
            current_time=env.current_time,
        )
        if not platforms:
            return hold_action(env)

        return min(
            platforms,
            key=lambda platform: downstream_blocking_score(
                platform=platform,
                waiting_queue=env.waiting_queue[1:],
                current_time=env.current_time,
            ),
        ).platform_id
