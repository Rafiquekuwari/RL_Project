"""Core data structures for RailSched-RL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Train:
    """Represents one train arrival and its operating requirements."""

    train_id: str
    scheduled_arrival: int
    scheduled_departure: int
    dwell_time: int
    priority: int
    platform_compatibility: int
    delay_minutes: int = 0
    actual_arrival: int = field(init=False)
    assigned_platform: int | None = None
    actual_start_service: int | None = None
    actual_departure: int | None = None
    waiting_time: int = 0
    propagated_delay: int = 0

    def __post_init__(self) -> None:
        self.actual_arrival = self.scheduled_arrival + self.delay_minutes


@dataclass(slots=True)
class Platform:
    """Represents a station platform."""

    platform_id: int
    occupied_until: int = 0
    compatibility_type: int = 0
    maintenance_block: bool = False
    current_train_id: str | None = None


@dataclass(slots=True)
class AssignmentRecord:
    """One assignment event for dashboard visualization."""

    train_id: str
    platform_id: int
    arrival_time: int
    service_start: int
    departure_time: int
    waiting_time: int
    propagated_delay: int
    priority: int


@dataclass(slots=True)
class EpisodeMetrics:
    """Aggregated results for one episode."""

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
