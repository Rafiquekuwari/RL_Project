"""Gymnasium environment for railway platform scheduling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from railsched_rl.config import ExperimentConfig
from railsched_rl.data_models import EpisodeMetrics, Train
from railsched_rl.generators.train_generator import generate_trains, queue_summary
from railsched_rl.simulator.station import StationSimulator


class RailPlatformEnv(gym.Env[np.ndarray, int]):
    """A custom environment where an agent assigns trains to platforms."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: ExperimentConfig, episode_seed: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.base_seed = episode_seed if episode_seed is not None else config.seed
        self.episode_index = 0
        self.simulator = StationSimulator(config.station, seed=self.base_seed)
        self.max_queue_size = 3
        self.max_trains = config.generator.max_trains
        self.max_steps = max(config.generator.episode_minutes * 3, config.generator.max_trains * 20)

        obs_size = 1 + (config.station.num_platforms * 4) + (self.max_queue_size * 4) + 4 + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(config.generator.episode_minutes + config.generator.max_delay_minutes + 120),
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.station.num_platforms + 1)

        self.current_time = 0
        self.upcoming_trains: list[Train] = []
        self.waiting_queue: list[Train] = []
        self.current_train: Train | None = None
        self.completed_trains: list[Train] = []
        self.total_reward = 0.0
        self.step_count = 0

    def _episode_seed(self) -> int:
        return self.base_seed + self.episode_index

    def _advance_time(self) -> None:
        if self.waiting_queue:
            self.current_time = max(self.current_time, min(train.actual_arrival for train in self.waiting_queue))
            return
        if self.upcoming_trains:
            self.current_time = max(self.current_time, self.upcoming_trains[0].actual_arrival)

    def _fill_waiting_queue(self) -> None:
        while self.upcoming_trains and self.upcoming_trains[0].actual_arrival <= self.current_time:
            self.waiting_queue.append(self.upcoming_trains.pop(0))
        self.waiting_queue.sort(key=lambda train: (train.actual_arrival, -train.priority))

    def _select_current_train(self) -> Train | None:
        if not self.waiting_queue:
            return None
        return self.waiting_queue[0]

    def _build_observation(self) -> np.ndarray:
        current_train = self.current_train
        train_features = [0.0, 0.0, 0.0, 0.0]
        if current_train is not None:
            train_features = [
                float(current_train.actual_arrival),
                float(current_train.dwell_time),
                float(current_train.priority),
                float(current_train.platform_compatibility),
            ]

        queue_features = queue_summary(self.waiting_queue[1:], max_items=self.max_queue_size)
        platform_features = self.simulator.platform_state_vector(self.current_time)
        summary_features = [
            float(len(self.waiting_queue)),
            float(len(self.upcoming_trains)),
            float(self.simulator.current_utilization()),
        ]
        observation = np.array(
            [float(self.current_time)] + platform_features + queue_features + train_features + summary_features,
            dtype=np.float32,
        )
        return observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""

        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = seed
            self.episode_index = 0

        episode_seed = self._episode_seed()
        self.simulator = StationSimulator(self.config.station, seed=episode_seed)
        self.upcoming_trains = deepcopy(generate_trains(self.config.generator, seed=episode_seed))
        self.waiting_queue = []
        self.completed_trains = []
        self.total_reward = 0.0
        self.step_count = 0
        self.current_time = 0
        self._advance_time()
        self._fill_waiting_queue()
        self.current_train = self._select_current_train()
        observation = self._build_observation()
        info = {"episode_seed": episode_seed}
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one scheduling action."""

        if self.current_train is None:
            observation = self._build_observation()
            return observation, 0.0, True, False, {}

        train = self.current_train
        self.step_count += 1
        result = self.simulator.assign_or_hold(
            train=train,
            action=int(action),
            current_time=self.current_time,
            queue_length=len(self.waiting_queue),
        )
        reward = result.reward

        if result.assigned:
            self.waiting_queue.pop(0)
            self.completed_trains.append(train)
        else:
            self.current_time += 1

        self._advance_time()
        self._fill_waiting_queue()
        self.current_train = self._select_current_train()

        terminated = self.current_train is None and (not self.upcoming_trains) and (not self.waiting_queue)
        truncated = self.step_count >= self.max_steps
        self.total_reward += reward
        if truncated:
            reward -= self.config.station.invalid_action_penalty
            self.total_reward -= self.config.station.invalid_action_penalty

        info = {
            "current_time": self.current_time,
            "queue_length": len(self.waiting_queue),
            "utilization": self.simulator.current_utilization(),
            "invalid_actions": self.simulator.invalid_actions,
            "conflicts": self.simulator.conflicts,
            "step_count": self.step_count,
        }
        observation = self._build_observation()
        if terminated or truncated:
            self.episode_index += 1
            info["episode_metrics"] = self.get_episode_metrics("ppo", self.episode_index)
        return observation, reward, terminated, truncated, info

    def get_episode_metrics(self, scheduler_name: str, episode_id: int) -> EpisodeMetrics:
        """Return metrics for the finished episode."""

        all_trains = self.completed_trains + self.waiting_queue + self.upcoming_trains
        return self.simulator.build_metrics(
            episode_id=episode_id,
            scheduler_name=scheduler_name,
            total_reward=self.total_reward,
            trains=all_trains,
        )

    def render(self) -> None:
        """Print a compact state summary."""

        current_train_id = self.current_train.train_id if self.current_train else "None"
        print(
            f"time={self.current_time} current_train={current_train_id} "
            f"queue={len(self.waiting_queue)} utilization={self.simulator.current_utilization():.2f}"
        )
