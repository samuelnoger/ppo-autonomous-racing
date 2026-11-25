# ppo-autonomous-racing

A PyTorch reinforcement learning project where an Actor-Critic agent learns to drive around 2D race tracks using vectorized kinematics, ray-cast perception, and Proximal Policy Optimization (PPO).

## Features

- **Vectorized 2D Physics:** Batched PyTorch car dynamics (`CarState`) featuring Ackermann steering, speed-dependent grip/drift hysteresis, and wall-drag collision handling.
- **PPO Multi-Track Training:** Mixes rollouts from multiple tracks into every PPO update (with GAE and `CosineAnnealingLR`) so the policy generalizes across wide sweepers and tight hairpins without forgetting.
- **Ray-Cast Perception:** 1D CNN encoder over distance rays combined with speed and upcoming track curvature/lookahead features.
- **Real-Time Viewer:** Tkinter simulation (`simulation.py`) that loads `config.json` and trained checkpoints for zero-noise evaluation.
- **Built-in & Custom Tracks:** Supports built-in layouts (`simple`, `square`, `triangle`, `square_narrow`, `us_track`, `redbull_ring`) as well as custom GeoJSON tracks.

## Installation

Requires Python 3.11+ and Tkinter.

```bash
pip install torch numpy matplotlib opencv-python geopandas tqdm

```

## Quick Start

### 1. Train a model

Train across multiple tracks simultaneously (CPU is recommended, as vectorized rollouts avoid GPU/MPS sync overhead):

```bash
python -m train.train_rl \
  --tracks square triangle square_narrow \
  --n_epochs 150 \
  --n_steps 400 \
  --n_cars 128 \
  --device cpu \
  --save checkpoints/base.pth

```

Training automatically saves `config.json`, periodic checkpoints (`checkpoints/base.pth`), the best all-around model (`checkpoints/base_best.pth`), and a reward curve (`reward_plot.png`).

To resume or fine-tune from an existing checkpoint:

```bash
python -m train.train \
  --load checkpoints/base_best.pth \
  --tracks square triangle square_narrow \
  --n_epochs 100 \
  --lr 1e-4

```

### 2. Run the simulation

```bash
python simulation.py

```

Update the checkpoint path or track inside `simulate_nn.py` (or pass `--track <name>`) to test different models and layouts.

## Architecture

`CarNet` (`model/model_CarNet.py`) is an Actor-Critic network:

* **Ray encoder:** 2-layer 1D CNN over normalized wall-distance rays.
* **State encoder:** MLP over normalized speed, upcoming curvatures, and lookahead distances.
* **Heads:** Shared MLP trunk feeding a continuous Gaussian policy (`steering`, `accel` in `[-1, 1]` with learnable `log_std`) and a scalar value head `V(s)`.

## Development Methodology

Core PyTorch boilerplate and network scaffolding were developed with the assistance of AI coding tools. My primary contributions focus on the conceptual design, 2D physics and drift modeling, reward/PPO tuning, and end-to-end training and evaluation workflows.