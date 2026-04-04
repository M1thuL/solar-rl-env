---
title: Solar Panel Optimization RL Environment
emoji: ☀️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Solar Panel Optimization — Predictive AI-Controlled RL Environment

A reinforcement learning environment that simulates a real-world dual-axis solar panel system. The agent must orient the panel using predictive irradiance signals to maximise energy output over a full simulated day.

Live demo: [huggingface.co/spaces/M1thuL/solar-rl-env](https://huggingface.co/spaces/M1thuL/solar-rl-env)

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
```

### Energy Formula

```
energy_wh = irradiance × cos(angle_difference) × step_duration_hours
```

---

## Task Tiers

| Parameter | Easy | Medium | Hard |
|-----------|:----:|:------:|:----:|
| Cloud noise σ | 0.0 | 0.2 | 0.5 |
| Movement penalty | 0.0 | 0.1 | 0.3 |
| Forecast noise σ | 0.0 | 0.1 | 0.3 |
| Steps per episode | 96 | 96 | 96 |

---

## Project Structure

```
solar-rl-env/
├── env/
│   ├── models.py        # Pydantic typed models
│   ├── solar_env.py     # Core environment — reset(), step(), state()
│   ├── tasks.py         # easy / medium / hard factory functions
│   └── __init__.py
├── configs/
│   └── openenv.yaml     # OpenEnv specification
├── baseline/
│   └── baseline.py      # Greedy sun-tracking baseline agent
├── app/
│   └── app.py           # Gradio demo
├── server/
│   └── app.py           # OpenEnv server entry point
├── inference.py         # Submission inference script
├── openenv.yaml         # OpenEnv spec (root copy for validator)
├── pyproject.toml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/M1thuL/solar-rl-env
cd solar-rl-env
pip install -r requirements.txt

# Run Gradio demo
python app/app.py

# Run baseline agent
python baseline/baseline.py

# Run inference script
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
```

---

## Docker

```bash
docker build -t solar-rl-env .
docker run -p 7860:7860 solar-rl-env
```

---

## License

MIT


<3