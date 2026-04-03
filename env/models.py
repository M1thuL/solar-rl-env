"""
models.py — Typed Pydantic models for the Solar RL Environment.

These models define the core data contracts for:
  - SolarState      : everything the agent observes at each timestep
  - SolarAction     : what the agent can do (tilt + rotation delta)
  - StepResult      : what the environment returns after each step
  - RewardBreakdown : transparent sub-components of the reward signal
  - EpisodeConfig   : per-task configuration knobs
  - EpisodeInfo     : metadata returned inside StepResult
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 1. Task difficulty enum
# ---------------------------------------------------------------------------

class TaskDifficulty(str, Enum):
    """The three supported OpenEnv task tiers."""
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# 2. Core observation — what the agent *sees* at every timestep
# ---------------------------------------------------------------------------

class SolarState(BaseModel):
    """
    The full observation returned by reset() and step().

    All angular values are in degrees for human-readability;
    the environment converts to radians internally where needed.
    """

    # --- Time ---
    time_of_day: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Current hour of the day (0 = midnight, 12 = noon, 24 = midnight).",
    )

    # --- Sun geometry (ground truth, computed from a simple solar model) ---
    sun_azimuth: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description=(
            "True compass bearing of the sun (degrees). "
            "0 = North, 90 = East, 180 = South, 270 = West."
        ),
    )
    sun_elevation: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description=(
            "True angle of the sun above the horizon (degrees). "
            "Negative values mean the sun is below the horizon."
        ),
    )

    # --- Irradiance signal ---
    predicted_irradiance: float = Field(
        ...,
        ge=0.0,
        description=(
            "Model-predicted solar irradiance at the *next* timestep (W/m²). "
            "On easy tasks this equals the true value; harder tasks add noise."
        ),
    )

    # --- Panel state (what the agent controls) ---
    panel_tilt: float = Field(
        ...,
        ge=0.0,
        le=90.0,
        description=(
            "Current panel tilt from horizontal (degrees). "
            "0 = flat, 90 = vertical."
        ),
    )
    panel_rotation: float = Field(
        ...,
        ge=0.0,
        lt=360.0,
        description=(
            "Current panel compass orientation (degrees). "
            "Follows the same convention as sun_azimuth."
        ),
    )

    # --- Derived convenience fields (populated by the env, not the agent) ---
    true_irradiance: float = Field(
        default=0.0,
        ge=0.0,
        description="Ground-truth irradiance at the current timestep (W/m²).",
    )
    energy_this_step: float = Field(
        default=0.0,
        ge=0.0,
        description="Energy harvested this timestep (Wh, after cloud / angle effects).",
    )
    step_index: int = Field(
        default=0,
        ge=0,
        description="How many steps have elapsed in the current episode.",
    )

    # Convenience: is the sun up?
    @property
    def sun_is_up(self) -> bool:
        return self.sun_elevation > 0.0

    class Config:
        # Allow extra fields so subclasses can extend without breaking validation
        extra = "allow"


# ---------------------------------------------------------------------------
# 3. Action — what the agent sends back to the environment
# ---------------------------------------------------------------------------

class SolarAction(BaseModel):
    """
    Continuous action space: deltas applied to panel tilt and rotation.

    Both values are clipped to [-1, 1] before use.  The environment scales
    them by `max_tilt_step` and `max_rotation_step` from EpisodeConfig.
    """

    tilt_change: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Fractional change to apply to panel_tilt. "
            "-1 = max tilt down, +1 = max tilt up."
        ),
    )
    rotation_change: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Fractional change to apply to panel_rotation. "
            "-1 = max counter-clockwise, +1 = max clockwise."
        ),
    )

    @field_validator("tilt_change", "rotation_change", mode="before")
    @classmethod
    def clip_to_valid_range(cls, v: float) -> float:
        """Silently clip out-of-range actions rather than raising."""
        return float(max(-1.0, min(1.0, v)))

    @classmethod
    def no_op(cls) -> "SolarAction":
        """Convenience: an action that does nothing."""
        return cls(tilt_change=0.0, rotation_change=0.0)


# ---------------------------------------------------------------------------
# 4. Reward breakdown — transparent sub-components
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Decomposed reward so researchers can introspect each term.

    total = energy_reward - movement_cost - misalignment_penalty
    """

    energy_reward: float = Field(
        default=0.0,
        description="Positive reward proportional to energy harvested this step.",
    )
    movement_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Penalty for physically moving the panel (task-dependent).",
    )
    misalignment_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Extra penalty when the panel is far from optimal sun-facing angle. "
            "Encourages the agent to track the sun proactively."
        ),
    )

    @property
    def total(self) -> float:
        return self.energy_reward - self.movement_cost - self.misalignment_penalty

    def as_dict(self) -> dict:
        return {
            "energy_reward":       self.energy_reward,
            "movement_cost":       self.movement_cost,
            "misalignment_penalty": self.misalignment_penalty,
            "total":               self.total,
        }


# ---------------------------------------------------------------------------
# 5. Episode info — extra metadata carried inside StepResult
# ---------------------------------------------------------------------------

class EpisodeInfo(BaseModel):
    """
    Diagnostic metadata attached to every StepResult.
    Mirrors the 'info' dict convention from Gymnasium.
    """

    angle_difference_deg: float = Field(
        default=0.0,
        description="Angle (degrees) between the panel normal and the sun vector.",
    )
    optimal_tilt: float = Field(
        default=0.0,
        description="Tilt the panel *should* have to face the sun exactly.",
    )
    optimal_rotation: float = Field(
        default=0.0,
        description="Rotation the panel *should* have to face the sun exactly.",
    )
    cloud_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Multiplicative cloud attenuation applied this step (1 = clear sky).",
    )
    task: TaskDifficulty = Field(
        default=TaskDifficulty.EASY,
        description="Which task tier this episode is running.",
    )
    episode_cumulative_energy: float = Field(
        default=0.0,
        ge=0.0,
        description="Total energy harvested so far in this episode (Wh).",
    )


# ---------------------------------------------------------------------------
# 6. Step result — the full return value of env.step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Everything the environment returns after a single step.

    Compatible with the (obs, reward, terminated, truncated, info) tuple
    convention used by Gymnasium 0.26+.
    """

    state:      SolarState     = Field(..., description="New observation after the action.")
    reward:     float          = Field(..., description="Scalar reward for this step.")
    terminated: bool           = Field(..., description="Episode ended due to a terminal condition (night).")
    truncated:  bool           = Field(..., description="Episode ended due to max step limit.")
    breakdown:  RewardBreakdown = Field(..., description="Decomposed reward components.")
    info:       EpisodeInfo    = Field(..., description="Diagnostic metadata.")

    @property
    def done(self) -> bool:
        """Convenience alias: is the episode over for any reason?"""
        return self.terminated or self.truncated

    def to_gym_tuple(self):
        """
        Returns a classic Gymnasium-style 5-tuple:
        (observation_dict, reward, terminated, truncated, info_dict)
        """
        return (
            self.state.model_dump(),
            self.reward,
            self.terminated,
            self.truncated,
            {**self.info.model_dump(), **self.breakdown.as_dict()},
        )


# ---------------------------------------------------------------------------
# 7. Episode configuration — one per task tier
# ---------------------------------------------------------------------------

class EpisodeConfig(BaseModel):
    """
    All tunable knobs for a single episode / task tier.

    The `tasks.py` module constructs one EpisodeConfig per difficulty level
    and passes it into SolarEnv.__init__().
    """

    # Identity
    task: TaskDifficulty = Field(..., description="Which tier this config represents.")
    max_steps: int = Field(
        default=96,
        gt=0,
        description=(
            "Maximum steps per episode. Default 96 = 15-min intervals over 24 hours."
        ),
    )

    # Panel mechanics
    max_tilt_step: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum degrees the panel can tilt per step (scales tilt_change).",
    )
    max_rotation_step: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum degrees the panel can rotate per step (scales rotation_change).",
    )

    # Cloud / irradiance noise
    cloud_noise_std: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Standard deviation of Gaussian noise added to the cloud attenuation factor. "
            "0 = clear sky always."
        ),
    )
    prediction_noise_std: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Noise added to the predicted_irradiance signal seen by the agent. "
            "0 = perfect forecast."
        ),
    )

    # Reward shaping weights
    movement_penalty_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Multiplier on the movement cost term in the reward.",
    )
    misalignment_penalty_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Multiplier on the misalignment penalty term.",
    )
    energy_reward_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Global scale factor on the energy reward term.",
    )

    # Derived: step duration (hours) — episode covers one full day
    @property
    def step_duration_hours(self) -> float:
        return 24.0 / self.max_steps

    @model_validator(mode="after")
    def validate_noise_matches_difficulty(self) -> "EpisodeConfig":
        """
        Soft sanity check: warn if noise params seem inconsistent with the
        declared difficulty (doesn't raise, just enforces logical ordering).
        """
        if self.task == TaskDifficulty.EASY:
            assert self.cloud_noise_std == 0.0, (
                "Easy task must have cloud_noise_std=0 (clear sky)."
            )
        return self

    class Config:
        extra = "allow"