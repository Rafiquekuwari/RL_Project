"""Single-station railway scheduling simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from railsched_rl.config import StationConfig
from railsched_rl.data_models import AssignmentRecord, EpisodeMetrics, Platform, Train


@dataclass(slots=True)
class StepResult:
    """State transition outcome for one scheduling decision."""

    reward: float
    invalid_action: bool
    assigned: bool
    conflict: bool


class StationSimulator:
    """Maintains platform state and computes scheduling outcomes."""

    def __init__(self, config: StationConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.platforms: list[Platform] = []
        self.assignment_records: list[AssignmentRecord] = []
        self.invalid_actions = 0
        self.conflicts = 0
        self.max_queue_length = 0
        self.total_platform_busy_time = 0
        self.horizon_end = 1
        self.reset()

    def reset(self) -> None:
        """Reset all platform and metric state."""

        self.assignment_records = []
        self.invalid_actions = 0
        self.conflicts = 0
        self.max_queue_length = 0
        self.total_platform_busy_time = 0
        self.platforms = []
        for platform_id in range(self.config.num_platforms):
            compatibility_type = platform_id % 2
            maintenance_block = bool(self.rng.random() < self.config.maintenance_probability)
            self.platforms.append(
                Platform(
                    platform_id=platform_id,
                    occupied_until=0,
                    compatibility_type=compatibility_type,
                    maintenance_block=maintenance_block,
                    current_train_id=None,
                )
            )
        self._ensure_operational_coverage()

    def _ensure_operational_coverage(self) -> None:
        """Guarantee at least one active platform per compatibility type."""

        by_type: dict[int, list[Platform]] = {}
        for platform in self.platforms:
            by_type.setdefault(platform.compatibility_type, []).append(platform)

        for platforms in by_type.values():
            if any(not platform.maintenance_block for platform in platforms):
                continue
            platforms[0].maintenance_block = False

    def clone(self) -> "StationSimulator":
        """Create a deep copy for heuristic rollouts."""

        clone = StationSimulator(config=self.config, seed=self.seed)
        clone.rng = np.random.default_rng(self.seed)
        clone.platforms = deepcopy(self.platforms)
        clone.assignment_records = deepcopy(self.assignment_records)
        clone.invalid_actions = self.invalid_actions
        clone.conflicts = self.conflicts
        clone.max_queue_length = self.max_queue_length
        clone.total_platform_busy_time = self.total_platform_busy_time
        clone.horizon_end = self.horizon_end
        return clone

    def available_platform_ids(self, train: Train, current_time: int) -> list[int]:
        """Return all compatible platforms currently free for the train."""

        available: list[int] = []
        for platform in self.platforms:
            if platform.maintenance_block:
                continue
            if platform.compatibility_type != train.platform_compatibility:
                continue
            if platform.occupied_until > current_time:
                continue
            available.append(platform.platform_id)
        return available

    def platform_state_vector(self, current_time: int) -> list[float]:
        """Return a flattened platform feature vector."""

        state: list[float] = []
        for platform in self.platforms:
            remaining_busy = max(platform.occupied_until - current_time, 0)
            state.extend(
                [
                    float(platform.platform_id),
                    float(remaining_busy),
                    float(platform.compatibility_type),
                    float(int(platform.maintenance_block)),
                ]
            )
        return state

    def assign_or_hold(
        self,
        train: Train,
        action: int,
        current_time: int,
        queue_length: int,
    ) -> StepResult:
        """Apply one action for a train and return reward details."""

        self.max_queue_length = max(self.max_queue_length, queue_length)
        hold_action = len(self.platforms)
        if action == hold_action:
            reward = -self.config.hold_penalty
            return StepResult(reward=reward, invalid_action=False, assigned=False, conflict=False)

        if action < 0 or action >= len(self.platforms):
            self.invalid_actions += 1
            return StepResult(
                reward=-self.config.invalid_action_penalty,
                invalid_action=True,
                assigned=False,
                conflict=True,
            )

        platform = self.platforms[action]
        is_compatible = platform.compatibility_type == train.platform_compatibility
        is_available = platform.occupied_until <= current_time
        is_maintained = platform.maintenance_block

        if (not is_compatible) or (not is_available) or is_maintained:
            self.invalid_actions += 1
            self.conflicts += 1
            return StepResult(
                reward=-(self.config.invalid_action_penalty + self.config.conflict_penalty),
                invalid_action=True,
                assigned=False,
                conflict=True,
            )

        service_start = max(current_time, train.actual_arrival)
        waiting_time = max(service_start - train.actual_arrival, 0)
        actual_departure = service_start + train.dwell_time
        propagated_delay = max(actual_departure - train.scheduled_departure, 0)

        platform.occupied_until = actual_departure
        platform.current_train_id = train.train_id
        train.assigned_platform = platform.platform_id
        train.actual_start_service = service_start
        train.actual_departure = actual_departure
        train.waiting_time = waiting_time
        train.propagated_delay = propagated_delay

        self.total_platform_busy_time += train.dwell_time
        self.horizon_end = max(self.horizon_end, actual_departure)
        utilization_reward = self.config.utilization_reward_weight * self.current_utilization()

        reward = (
            self.config.assignment_reward
            - self.config.wait_penalty_per_minute * waiting_time
            - self.config.propagation_penalty_per_minute * propagated_delay
            + utilization_reward
            + (0.3 * train.priority)
        )

        self.assignment_records.append(
            AssignmentRecord(
                train_id=train.train_id,
                platform_id=platform.platform_id,
                arrival_time=train.actual_arrival,
                service_start=service_start,
                departure_time=actual_departure,
                waiting_time=waiting_time,
                propagated_delay=propagated_delay,
                priority=train.priority,
            )
        )
        return StepResult(reward=reward, invalid_action=False, assigned=True, conflict=False)

    def current_utilization(self) -> float:
        """Compute station utilization over the observed horizon."""

        capacity = max(self.horizon_end * len(self.platforms), 1)
        return min(self.total_platform_busy_time / capacity, 1.0)

    def build_metrics(
        self,
        episode_id: int,
        scheduler_name: str,
        total_reward: float,
        trains: Iterable[Train],
    ) -> EpisodeMetrics:
        """Aggregate station and train metrics into one record."""

        assigned_trains = [train for train in trains if train.assigned_platform is not None]
        total_waiting = float(sum(train.waiting_time for train in assigned_trains))
        total_delay = float(sum(train.propagated_delay for train in assigned_trains))
        num_assignments = len(assigned_trains)
        avg_waiting = total_waiting / max(num_assignments, 1)
        return EpisodeMetrics(
            episode_id=episode_id,
            scheduler_name=scheduler_name,
            total_reward=total_reward,
            total_waiting_time=total_waiting,
            average_waiting_time=avg_waiting,
            total_propagated_delay=total_delay,
            assignments=num_assignments,
            invalid_actions=self.invalid_actions,
            conflicts=self.conflicts,
            average_utilization=self.current_utilization(),
            max_queue_length=self.max_queue_length,
        )
