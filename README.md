# Solar Panel Optimization — Predictive AI-Controlled RL Environment

> **Meta PyTorch Hackathon** · OpenEnv track  
> A reinforcement learning environment that simulates a real-world dual-axis solar panel system. The agent must orient the panel using predictive irradiance signals to maximise energy output over a full simulated day.

---

## Demo

| Easy (clear sky) | Hard (heavy clouds + noisy forecast) |
|:---:|:---:|
| Panel tracks sun perfectly, near-maximum energy | Agent must handle uncertainty and movement costs |

Run the live demo locally:
```bash
python app/app.py
# Open http://127.0.0.1:7860
```

---

## Project Structure

```
solar-rl-env/
├── env/
│   ├── models.py        # Pydantic typed models (SolarState, SolarAction, StepResult …)
│   ├── solar_env.py     # Core environment — reset(), step(), state()
│   ├── tasks.py         # Factory functions: make_easy_env(), make_medium_env(), make_hard_env()
│   └── __init__.py
├── configs/
│   └── openenv.yaml     # OpenEnv specification file
├── baseline/
│   └── baseline.py      # Greedy sun-tracking baseline agent
├── app/
│   └── app.py           # Gradio interactive demo (HF Spaces compatible)
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Environment Overview

### State Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `time_of_day` | float | [0, 24] | Current simulated hour |
| `sun_azimuth` | float | [0, 360) | Compass bearing of the sun |
| `sun_elevation` | float | [-90, 90] | Sun angle above horizon; negative = night |
| `predicted_irradiance` | float | [0, 1000] | Forecast irradiance W/m² (noisy on hard tasks) |
| `panel_tilt` | float | [0, 90] | Panel tilt from horizontal |
| `panel_rotation` | float | [0, 360) | Panel compass orientation |

### Action Space

| Field | Range | Effect |
|-------|-------|--------|
| `tilt_change` | [-1, 1] | Scaled by ±5°/step, clamped to [0°, 90°] |
| `rotation_change` | [-1, 1] | Scaled by ±10°/step, wraps [0°, 360°) |

### Reward Function

```
reward = energy_reward − movement_cost − misalignment_penalty

energy_reward        = energy_wh × scale
movement_cost        = weight × (|Δtilt| + |Δrotation|) / max_possible_move
misalignment_penalty = weight × angle_diff / 90°   (suppressed at night)
```

### Energy Formula

```
energy_wh = irradiance × cos(angle_difference) × step_duration_hours
angle_difference = √(Δtilt² + Δrotation²), clamped to [0°, 90°]
```

---

## Task Tiers

| Parameter | Easy | Medium | Hard |
|-----------|:----:|:------:|:----:|
| Cloud noise σ | 0.0 | 0.2 | 0.5 |
| Movement penalty weight | 0.0 | 0.1 | 0.3 |
| Forecast noise σ | 0.0 | 0.1 | 0.3 |
| Steps per episode | 96 | 96 | 96 |
| Step duration | 15 min | 15 min | 15 min |

- **Easy** — Clear sky, free movement, perfect forecast. Use to verify the environment and set a performance ceiling.
- **Medium** — Realistic rooftop conditions. Primary training target.
- **Hard** — Heavy clouds, strong movement cost, noisy forecast. Stress-tests robust agents.

---

## Quick Start

### Install

```bash
git clone https://github.com/your-username/solar-rl-env
cd solar-rl-env
pip install -r requirements.txt
```

### Run the Gradio demo

```bash
python app/app.py
# Open http://127.0.0.1:7860
```

### Run the baseline agent

```bash
# All three tasks:
python baseline/baseline.py

# Single task with verbose step output:
python baseline/baseline.py --task hard --verbose

# Custom seed:
python baseline/baseline.py --task medium --seed 7
```

### Use the environment in your own code

```python
from env.tasks import make_easy_env, make_medium_env, make_hard_env
from env.models import SolarAction

# Create and reset
env = make_medium_env(seed=42)
state = env.reset()

# Agent loop
while True:
    action = SolarAction(tilt_change=0.5, rotation_change=-0.3)
    result = env.step(action)

    print(f"t={result.state.time_of_day:.2f}h  "
          f"reward={result.reward:.4f}  "
          f"energy={result.state.energy_this_step:.4f} Wh")

    if result.done:
        print(f"Episode done. Total energy: {result.info.episode_cumulative_energy:.3f} Wh")
        break

# Read current state without stepping
current = env.state()
```

### Use the task registry

```python
from env.tasks import TASK_REGISTRY

for task_name in ["easy", "medium", "hard"]:
    env = TASK_REGISTRY[task_name](seed=42)
    state = env.reset()
    print(f"{task_name}: {env}")
```

---

## Docker

```bash
# Build
docker build -t solar-rl-env .

# Run
docker run -p 7860:7860 solar-rl-env

# Override command (e.g. run baseline inside container)
docker run solar-rl-env python baseline/baseline.py --task hard
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Set **SDK** to `Docker`
3. Push this repository:

```bash
git remote add space https://huggingface.co/spaces/your-username/solar-rl-env
git push space main
```

HF Spaces will automatically build the Docker image and serve the Gradio UI at `https://your-username-solar-rl-env.hf.space`.

> **Runtime:** CPU Basic (free tier). No GPU required — the environment is pure Python + NumPy.

---

## OpenEnv Specification

The full environment contract is defined in [`configs/openenv.yaml`](configs/openenv.yaml), including:
- Complete state and action space definitions with ranges and units
- Reward function formulas for all three terms
- Per-task configuration values
- Episode mechanics (step duration, termination conditions)
- Solar model parameters (latitude, peak irradiance, sunrise/sunset)

---

## Implementation Notes

### Solar model
A simplified sinusoidal approximation — no astronomy library needed:
- **Elevation**: half-sine arc peaking at solar noon (~61° at 28.6°N latitude)
- **Azimuth**: linear sweep East (90°) → South (180°) → West (270°)
- **Irradiance**: `PEAK × sin(elevation)`, modelling the air-mass effect

### Reward design
The `misalignment_penalty` is the most important term for learning. Without it the agent gets zero gradient signal at night (no energy = no reward regardless of panel position). With it, the agent learns to **pre-position** toward sunrise — exactly the predictive behaviour the environment is designed to reward.

### Reproducibility
All randomness (cloud noise, forecast noise) flows through a seeded `random.Random` instance per environment. Two environments with the same seed produce identical episodes regardless of other system state.

---

## Requirements

```
pydantic>=2.0.0,<3.0.0
gradio==4.44.1
huggingface_hub==0.24.6
numpy>=1.24.0,<2.0.0
matplotlib  # installed as gradio dependency
```

Python 3.10+ recommended.

---

## License

MIT