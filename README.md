# RailSched-RL

RailSched-RL is a final-year project starter that models railway platform scheduling as a reinforcement learning problem. A PPO agent learns to assign delayed and priority-sensitive trains to a limited set of platforms while minimizing waiting time, propagated delays, and conflicts.

## Project Structure

```text
RailSched-RL/
├── configs/
│   └── default.yaml
├── logs/
├── models/
├── outputs/
├── plots/
├── railsched_rl/
│   ├── baselines/
│   ├── dashboard/
│   ├── envs/
│   ├── generators/
│   ├── simulator/
│   ├── training/
│   └── utils/
├── README.md
└── requirements.txt
```

## Features

- Synthetic train arrival generation with delays, priorities, and compatibility classes
- Single-station platform simulator with maintenance blocks
- Custom Gymnasium environment with a discrete action space
- PPO training using Stable-Baselines3
- TensorBoard logging for PPO training diagnostics
- Baseline schedulers: FCFS, earliest-free, and priority-aware
- CSV logging for training and evaluation metrics
- Matplotlib training and evaluation plots
- Streamlit dashboard for interactive visualization

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the PPO Agent

```bash
python -m railsched_rl.training.train_ppo --config configs/default.yaml
```

This saves the trained model to `models/ppo_railsched.zip`.

Training also generates visual diagnostics:

- TensorBoard logs: `logs/tensorboard/`
- Training metrics CSV: `outputs/training/training_metrics.csv`
- Training plots: `plots/training/`

The generated training plots are:

- `plots/training/reward_vs_timestep.png`
- `plots/training/waiting_time_vs_timestep.png`
- `plots/training/utilization_vs_timestep.png`
- `plots/training/conflicts_vs_timestep.png`

## View TensorBoard

TensorBoard reads the PPO event logs written during training and provides interactive reward, loss, policy, and value-function diagnostics.

Start it with:

```bash
tensorboard --logdir logs/tensorboard
```

Then open the local URL printed by TensorBoard, usually `http://localhost:6006`.

## Evaluate PPO Against Baselines

```bash
python -m railsched_rl.training.evaluate --config configs/default.yaml
```

This generates:

- `outputs/comparison_metrics.csv`
- `logs/assignment_records.csv`
- `plots/comparison_plot.png`

## Launch the Dashboard

```bash
streamlit run railsched_rl/dashboard/app.py
```

The dashboard has two tabs:

- `Evaluation`: comparison metrics, assignment timeline, platform occupancy, queue length, and evaluation plots
- `Training Progress`: reward, waiting time, utilization, and conflicts curves from `outputs/training/training_metrics.csv` and `plots/training/`

If training files are missing, the dashboard shows a helpful message instead of failing.

## Notes on the Environment

- Observation includes current time, platform states, waiting queue summary, next train features, and utilization context
- Action `0..N-1` maps to platform indices and action `N` means hold the train in the queue
- Rewards encourage valid assignments, lower waiting time, lower propagated delay, fewer conflicts, and higher utilization

## Reproducibility

- Seeds are loaded from `configs/default.yaml`
- Synthetic episodes are deterministic for a given seed
- PPO training also uses the same base seed

## Suggested Final-Year Project Extensions

- Multi-station dispatch coordination
- Stochastic disruptions and maintenance windows
- Richer reward shaping with passenger delay weighting
- Comparison with optimization-based schedulers
# RL_Project
