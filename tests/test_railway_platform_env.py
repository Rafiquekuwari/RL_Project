"""Pytest coverage for the railway platform scheduling environment."""

from __future__ import annotations

import math

import numpy as np

from railway_rl.envs import RailwayPlatformEnv
from railsched_rl.config import (
    EvaluationConfig,
    ExperimentConfig,
    GeneratorConfig,
    LoggingConfig,
    StationConfig,
    TrainingConfig,
)
from railsched_rl.data_models import Train


def make_config(num_platforms: int = 4) -> ExperimentConfig:
    """Build a compact deterministic config for tests."""

    return ExperimentConfig(
        seed=123,
        generator=GeneratorConfig(
            episode_minutes=120,
            min_trains=6,
            max_trains=6,
            delay_probability=0.0,
            max_delay_minutes=10,
            min_dwell_time=5,
            max_dwell_time=8,
            compatibility_types=[0, 1],
            priority_levels=[1, 2, 3],
        ),
        station=StationConfig(
            num_platforms=num_platforms,
            maintenance_probability=0.0,
            hold_penalty=1.0,
            invalid_action_penalty=4.0,
            assignment_reward=5.0,
            conflict_penalty=3.0,
            wait_penalty_per_minute=0.15,
            propagation_penalty_per_minute=0.25,
            utilization_reward_weight=1.5,
        ),
        training=TrainingConfig(
            total_timesteps=256,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            eval_episodes=2,
        ),
        evaluation=EvaluationConfig(
            episodes=2,
            output_dir="outputs",
        ),
        logging=LoggingConfig(
            logs_dir="logs",
            models_dir="models",
            plots_dir="plots",
            outputs_dir="outputs",
        ),
    )


def make_env(num_platforms: int = 4) -> RailwayPlatformEnv:
    """Create and reset an environment for testing."""

    env = RailwayPlatformEnv(config=make_config(num_platforms=num_platforms), episode_seed=123)
    env.reset()
    return env


def make_train(
    compatibility: int,
    arrival: int = 0,
    dwell: int = 6,
    priority: int = 2,
) -> Train:
    """Create a deterministic train instance for direct state setup."""

    return Train(
        train_id="TEST",
        scheduled_arrival=arrival,
        scheduled_departure=arrival + dwell,
        dwell_time=dwell,
        priority=priority,
        platform_compatibility=compatibility,
        delay_minutes=0,
    )


def first_platform_with_compatibility(env: RailwayPlatformEnv, compatibility: int) -> int:
    """Return the first platform matching the requested compatibility."""

    for platform in env.simulator.platforms:
        if platform.compatibility_type == compatibility:
            return platform.platform_id
    raise AssertionError(f"No platform found for compatibility type {compatibility}")


def setup_single_train_state(env: RailwayPlatformEnv, train: Train) -> int:
    """Replace runtime state with one deterministic train and one free station."""

    env.current_time = train.actual_arrival
    env.waiting_queue = [train]
    env.current_train = train
    env.upcoming_trains = []
    env.completed_trains = []
    env.step_count = 0
    env.total_reward = 0.0

    for platform in env.simulator.platforms:
        platform.occupied_until = 0
        platform.maintenance_block = False
        platform.current_train_id = None

    return first_platform_with_compatibility(env, train.platform_compatibility)


def test_reset_returns_valid_observation_and_info() -> None:
    """Environment reset should return a valid observation/info pair."""

    env = make_env()
    observation, info = env.reset()

    assert isinstance(info, dict)
    assert "episode_seed" in info
    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)

    env.close()


def test_action_space_and_observation_space_are_valid() -> None:
    """Spaces should match the configured station size and observation shape."""

    env = make_env(num_platforms=5)

    assert env.action_space.n == 6
    assert env.observation_space.shape is not None
    assert len(env.observation_space.shape) == 1
    assert env.observation_space.shape[0] > 0
    assert env.observation_space.dtype == np.float32

    env.close()


def test_assigning_to_free_compatible_platform_succeeds() -> None:
    """A train assigned to a free compatible platform should be accepted."""

    env = make_env()
    train = make_train(compatibility=0)
    valid_action = setup_single_train_state(env, train)

    observation, reward, terminated, truncated, _ = env.step(valid_action)

    assert isinstance(observation, np.ndarray)
    assert reward > 0
    assert not truncated
    assert train.assigned_platform == valid_action
    assert len(env.completed_trains) == 1
    assert env.simulator.invalid_actions == 0
    assert terminated

    env.close()


def test_assigning_to_occupied_or_incompatible_platform_gets_penalized() -> None:
    """Invalid platform assignments should receive a negative penalty."""

    env = make_env()
    train = make_train(compatibility=0)
    valid_action = setup_single_train_state(env, train)

    occupied_platform = env.simulator.platforms[valid_action]
    occupied_platform.occupied_until = env.current_time + 20

    incompatible_action = first_platform_with_compatibility(env, compatibility=1)

    _, occupied_reward, _, _, _ = env.step(valid_action)

    assert occupied_reward < 0
    assert env.simulator.invalid_actions == 1
    assert env.simulator.conflicts == 1

    env.close()

    env = make_env()
    train = make_train(compatibility=0)
    setup_single_train_state(env, train)

    _, incompatible_reward, _, _, _ = env.step(incompatible_action)

    assert incompatible_reward < 0
    assert env.simulator.invalid_actions == 1
    assert env.simulator.conflicts == 1

    env.close()


def test_hold_action_increases_waiting_time() -> None:
    """The HOLD action should increase waiting time by keeping the train queued."""

    env = make_env()
    train = make_train(compatibility=0, arrival=0)
    setup_single_train_state(env, train)
    hold_action = env.action_space.n - 1

    _, reward, terminated, truncated, _ = env.step(hold_action)

    assert reward < 0
    assert not terminated
    assert not truncated
    assert train.waiting_time == 0
    assert env.current_time == 1

    env.close()


def test_episode_terminates_correctly() -> None:
    """An episode should terminate after the final train is assigned."""

    env = make_env()
    train = make_train(compatibility=0)
    valid_action = setup_single_train_state(env, train)

    _, _, terminated, truncated, _ = env.step(valid_action)

    assert terminated
    assert not truncated
    assert env.current_train is None
    assert not env.waiting_queue
    assert not env.upcoming_trains

    env.close()


def test_reward_stays_numeric_and_finite() -> None:
    """Rewards returned by the environment should be finite numeric values."""

    env = make_env()
    observation, _ = env.reset()

    assert env.observation_space.contains(observation)

    for _ in range(10):
        if env.current_train is None:
            break
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(reward, (int, float))
        assert math.isfinite(float(reward))
        if terminated or truncated:
            break

    env.close()


def test_platform_occupancy_updates_correctly_after_assignment() -> None:
    """Platform occupied_until should update to the assigned train departure time."""

    env = make_env()
    train = make_train(compatibility=0, arrival=10, dwell=7)
    valid_action = setup_single_train_state(env, train)
    platform = env.simulator.platforms[valid_action]

    _, _, terminated, _, _ = env.step(valid_action)

    assert terminated
    assert train.actual_start_service == 10
    assert train.actual_departure == 17
    assert platform.occupied_until == 17
    assert platform.current_train_id == train.train_id

    env.close()
