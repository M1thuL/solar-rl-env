"""
tasks.py — Task configurations and factory functions for the Solar RL Environment.

Three difficulty tiers, each returning a ready-to-use SolarEnv:

    make_easy_env()   → clear sky, free movement, perfect forecast
    make_medium_env() → mild clouds, small movement cost, slight forecast noise
    make_hard_env()   → heavy clouds, strong movement cost, noisy forecast

Usage
-----
    from env.tasks import make_easy_env, make_medium_env, make_hard_env

    env = make_easy_env(seed=42)
    state = env.reset()
    result = env.step(action)
"""

from __future__ import annotations

from env.models import EpisodeConfig, TaskDifficulty
from env.solar_env import SolarEnv

# ---------------------------------------------------------------------------
# Raw parameter tables
# ---------------------------------------------------------------------------
# Keeping the numbers in one plain dict makes them easy to audit, tweak,
# and reference from openenv.yaml without digging through class definitions.
#
# Field mapping to EpisodeConfig:
#   cloud_noise_std          ← how variable the cloud cover is each step
#   movement_penalty_weight  ← cost multiplier per unit of panel movement
#   prediction_noise_std     ← std-dev of noise added to the irradiance forecast
#   misalignment_penalty_weight ← cost per unit of angular misalignment (fixed)
#   max_steps                ← 96 steps × 0.25 h = 24 h per episode for all tiers
# ---------------------------------------------------------------------------

TASK_PARAMS: dict[TaskDifficulty, dict] = {

    TaskDifficulty.EASY: {
        # Perfect conditions — the agent's only job is to learn sun-tracking.
        # No clouds means irradiance is deterministic.
        # No movement cost means the agent can explore freely.
        # Perfect forecast means predicted == true every step.
        "cloud_noise_std":           0.0,
        "movement_penalty_weight":   0.0,
        "prediction_noise_std":      0.0,
        "misalignment_penalty_weight": 0.1,   # light nudge to face the sun
        "energy_reward_scale":       1.0,
        "max_steps":                 96,       # 15-min intervals × 96 = 24 h
        "max_tilt_step":             5.0,      # max 5° tilt change per step
        "max_rotation_step":         10.0,     # max 10° rotation change per step
    },

    TaskDifficulty.MEDIUM: {
        # Realistic rooftop conditions.
        # Clouds attenuate irradiance unpredictably.
        # Small movement cost discourages wasteful thrashing.
        # Mild forecast noise tests whether the agent can handle uncertainty.
        "cloud_noise_std":           0.2,
        "movement_penalty_weight":   0.1,
        "prediction_noise_std":      0.1,
        "misalignment_penalty_weight": 0.1,
        "energy_reward_scale":       1.0,
        "max_steps":                 96,
        "max_tilt_step":             5.0,
        "max_rotation_step":         10.0,
    },

    TaskDifficulty.HARD: {
        # Challenging grid-scale conditions.
        # Heavy clouds create large, unpredictable irradiance drops.
        # Strong movement cost forces the agent to commit to positions.
        # Noisy forecast means the agent must act on imperfect information.
        "cloud_noise_std":           0.5,
        "movement_penalty_weight":   0.3,
        "prediction_noise_std":      0.3,
        "misalignment_penalty_weight": 0.1,
        "energy_reward_scale":       1.0,
        "max_steps":                 96,
        "max_tilt_step":             5.0,
        "max_rotation_step":         10.0,
    },
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
# A thin helper that turns the flat dict above into a typed EpisodeConfig.
# Centralising this conversion means factory functions stay one-liners.
# ---------------------------------------------------------------------------

def _make_config(task: TaskDifficulty) -> EpisodeConfig:
    """
    Build a typed EpisodeConfig from the TASK_PARAMS table.

    Parameters
    ----------
    task : TaskDifficulty
        Which tier to look up.

    Returns
    -------
    EpisodeConfig
        Fully validated Pydantic config ready to pass into SolarEnv.
    """
    params = TASK_PARAMS[task]
    return EpisodeConfig(task=task, **params)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------
# Each function:
#   1. Builds the EpisodeConfig for its tier
#   2. Constructs SolarEnv with that config and the caller's seed
#   3. Returns the env — caller must still call env.reset() to start
#
# The `seed` parameter is threaded through so experiments are reproducible.
# Default seed=42 is arbitrary; override it in training loops.
# ---------------------------------------------------------------------------

def make_easy_env(seed: int = 42) -> SolarEnv:
    """
    Create an Easy-tier solar environment.

    Conditions: clear sky, free movement, perfect irradiance forecast.
    Use this to verify the environment works and to establish a performance
    ceiling — any agent should score near-optimal here.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility. Default 42.

    Returns
    -------
    SolarEnv
        Initialised but not yet reset. Call env.reset() before stepping.
    """
    config = _make_config(TaskDifficulty.EASY)
    return SolarEnv(config=config, seed=seed)


def make_medium_env(seed: int = 42) -> SolarEnv:
    """
    Create a Medium-tier solar environment.

    Conditions: mild cloud noise, small movement penalty, slight forecast noise.
    The primary training target — realistic enough to be useful, tractable
    enough for standard RL algorithms.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility. Default 42.

    Returns
    -------
    SolarEnv
        Initialised but not yet reset. Call env.reset() before stepping.
    """
    config = _make_config(TaskDifficulty.MEDIUM)
    return SolarEnv(config=config, seed=seed)


def make_hard_env(seed: int = 42) -> SolarEnv:
    """
    Create a Hard-tier solar environment.

    Conditions: heavy cloud noise, strong movement penalty, noisy forecast.
    Designed to stress-test agents trained on Medium — the agent must be
    efficient, predictive, and robust to uncertainty.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility. Default 42.

    Returns
    -------
    SolarEnv
        Initialised but not yet reset. Call env.reset() before stepping.
    """
    config = _make_config(TaskDifficulty.HARD)
    return SolarEnv(config=config, seed=seed)


# ---------------------------------------------------------------------------
# Convenience: task registry
# ---------------------------------------------------------------------------
# A single dict mapping strings → factory functions.
# Used by baseline.py and app.py to select a task by name without
# importing all three factory functions individually.
#
# Example:
#   from env.tasks import TASK_REGISTRY
#   env = TASK_REGISTRY["hard"](seed=7)
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, callable] = {
    "easy":   make_easy_env,
    "medium": make_medium_env,
    "hard":   make_hard_env,
}