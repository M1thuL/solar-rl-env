"""
solar_env.py — Core RL environment for the Solar Panel Optimization System.

Implements the OpenEnv specification:
  - reset()  : start a new episode, return initial SolarState
  - step()   : apply an action, advance time, return StepResult
  - state()  : return the current SolarState without advancing time

Built step-by-step:
  Step 1 — Class skeleton + __init__
  Step 2 — reset()
  Step 3 — Helper functions (sun position, irradiance, energy)
  Step 4 — step()
  Step 5 — state()
"""

from __future__ import annotations

import math
import random
from typing import Optional

from env.models import (
    EpisodeConfig,
    EpisodeInfo,
    RewardBreakdown,
    SolarAction,
    SolarState,
    StepResult,
    TaskDifficulty,
)


# ---------------------------------------------------------------------------
# STEP 1 — Class skeleton + __init__
# ---------------------------------------------------------------------------
# Design goals:
#   • Accept an EpisodeConfig so the same class drives all three task tiers.
#   • Keep ALL mutable episode state as plain Python floats / ints so it is
#     easy to reset, inspect, and serialise.
#   • Expose a `rng` (random.Random) seeded deterministically so experiments
#     are reproducible without touching the global random state.
# ---------------------------------------------------------------------------

class SolarEnv:
    """
    Simulates a single-axis + rotation solar panel tracking system.

    The agent observes the sun's position and a (possibly noisy) irradiance
    forecast, then adjusts panel tilt and rotation to maximise energy output
    over one simulated day (24 hours).

    Parameters
    ----------
    config : EpisodeConfig
        Task-specific configuration (noise levels, penalty weights, etc.).
        Use the factory helpers in tasks.py to obtain pre-built configs.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    # Geographic latitude used for the simplified solar model (degrees).
    # 28.6°N ≈ New Delhi — a sun-rich location, good for a solar demo.
    LATITUDE_DEG: float = 28.6

    # Peak irradiance on a perfectly clear day at solar noon (W/m²).
    # ~1000 W/m² is the standard test condition for solar panels.
    PEAK_IRRADIANCE: float = 1000.0

    # Panel physical limits
    MIN_TILT: float = 0.0    # flat (horizontal)
    MAX_TILT: float = 90.0   # vertical
    MIN_ROTATION: float = 0.0
    MAX_ROTATION: float = 360.0  # full compass rose (wraps around)

    def __init__(
        self,
        config: EpisodeConfig,
        seed: int = 42,
    ) -> None:
        """
        Store config and initialise all mutable episode-state variables.

        Note: __init__ does NOT start an episode. Call reset() first.
        """

        # ── Task configuration ──────────────────────────────────────────
        # The config encapsulates everything that differs between easy /
        # medium / hard: noise levels, penalty weights, step limits, etc.
        self.config: EpisodeConfig = config

        # ── Reproducible randomness ─────────────────────────────────────
        # Using an instance-level RNG (not the global random module) means
        # two envs with different seeds don't interfere with each other.
        self.rng: random.Random = random.Random(seed)

        # ── Episode state (all reset in reset()) ────────────────────────
        # We declare them here with sentinel values so type-checkers know
        # these attributes always exist after __init__.

        self._time_of_day: float  = 0.0   # hours, 0–24
        self._panel_tilt: float   = 0.0   # degrees, 0–90
        self._panel_rotation: float = 0.0 # degrees, 0–360

        # Cached sun geometry (recomputed each step)
        self._sun_azimuth: float    = 0.0
        self._sun_elevation: float  = 0.0

        # Cached irradiance values
        self._true_irradiance: float      = 0.0
        self._predicted_irradiance: float = 0.0

        # Step counter and cumulative energy tracker
        self._step_index: int              = 0
        self._cumulative_energy: float     = 0.0

        # Cloud attenuation factor applied to the CURRENT step (0–1)
        self._cloud_factor: float = 1.0

        # Energy generated in the most recent step (stored for state())
        self._energy_this_step: float = 0.0

        # Flag: has reset() been called at least once?
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Properties — read-only access to internal state
    # ------------------------------------------------------------------

    @property
    def task(self) -> TaskDifficulty:
        """Shortcut to the task difficulty of the current config."""
        return self.config.task

    @property
    def max_steps(self) -> int:
        """Total steps allowed per episode (from config)."""
        return self.config.max_steps

    @property
    def step_duration_hours(self) -> float:
        """Duration of each timestep in hours (e.g. 0.25 for 15-min steps)."""
        return self.config.step_duration_hours

    def __repr__(self) -> str:
        return (
            f"SolarEnv(task={self.task.value!r}, "
            f"step={self._step_index}/{self.max_steps}, "
            f"time={self._time_of_day:.2f}h)"
        )

    # ------------------------------------------------------------------
    # STEP 2 — reset()
    # ------------------------------------------------------------------
    # Design goals:
    #   • Fully deterministic — no randomness here, ever.
    #     Randomness only enters during step() via cloud / prediction noise.
    #     This guarantees every episode starts from the same known state,
    #     making debugging and reproducibility straightforward.
    #
    #   • Single source of truth — all internal state variables are set
    #     here in one place. If you need to understand "what does the env
    #     look like at t=0?", read this method.
    #
    #   • Calls compute_sun_position() — even though the sun is below the
    #     horizon at midnight, we still compute real geometry values so the
    #     first SolarState returned is physically consistent.
    # ------------------------------------------------------------------

    def reset(self) -> SolarState:
        """
        Start a new episode from a clean, deterministic initial state.

        Always call this before the first step() of an episode.
        Safe to call again mid-episode to restart.

        Returns
        -------
        SolarState
            The observation at t = 0 (midnight, panel flat and facing North).
        """

        # ── 1. Reset time ───────────────────────────────────────────────
        # Start at midnight (0.0 hours).
        # The sun is well below the horizon here — that's intentional.
        # The agent will watch the sun rise, peak, and set during the episode.
        self._time_of_day = 0.0

        # ── 2. Reset panel geometry ─────────────────────────────────────
        # Flat (tilt = 0°) and facing North (rotation = 0°).
        # This is a neutral, physically safe starting pose — a real panel
        # would park here overnight to avoid wind load.
        self._panel_tilt     = 0.0   # degrees from horizontal
        self._panel_rotation = 0.0   # degrees (0 = North)

        # ── 3. Reset episode counters ───────────────────────────────────
        self._step_index         = 0
        self._cumulative_energy  = 0.0
        self._energy_this_step   = 0.0

        # ── 4. Reset cloud factor to clear sky ──────────────────────────
        # No cloud noise at t=0; step() will sample noise on each advance.
        self._cloud_factor = 1.0

        # ── 5. Compute the sun's position at t = 0 ──────────────────────
        # compute_sun_position() returns (azimuth_deg, elevation_deg).
        # At midnight the sun is ~28° below the horizon (for our latitude),
        # so sun_elevation will be negative — this is correct behaviour.
        #
        # NOTE: compute_sun_position() is defined in Step 3 below.
        #       We call it here via self so the method is self-contained.
        self._sun_azimuth, self._sun_elevation = self._compute_sun_position(
            self._time_of_day
        )

        # ── 6. Compute ground-truth irradiance at t = 0 ─────────────────
        # At midnight the sun is down, so irradiance = 0.
        # We still call the function so the logic path is always exercised.
        # cloud_variance = 0.0 because reset is deterministic (no noise).
        self._true_irradiance = self._compute_irradiance(
            sun_elevation=self._sun_elevation,
            cloud_variance=0.0,          # clear sky, no randomness
        )
        self._cloud_factor = 1.0         # already set above; explicit for clarity

        # ── 7. Set predicted irradiance = true irradiance ───────────────
        # At reset there is no forecasting involved — the agent simply sees
        # the current true value as its "prediction" for the first step.
        # Noise is added during step() for medium/hard tasks.
        self._predicted_irradiance = self._true_irradiance

        # ── 8. Mark environment as initialised ──────────────────────────
        self._initialised = True

        # ── 9. Build and return the initial SolarState ──────────────────
        return self._build_state()

    # ------------------------------------------------------------------
    # Internal helper: build a SolarState snapshot from current fields
    # ------------------------------------------------------------------
    # Extracted into its own method so both reset() and step() can call it
    # without duplicating the SolarState(...) constructor arguments.
    # ------------------------------------------------------------------

    def _build_state(self) -> SolarState:
        """
        Package all current internal fields into a SolarState object.

        This is a pure read — it does not modify any internal state.
        """
        return SolarState(
            time_of_day           = self._time_of_day,
            sun_azimuth           = self._sun_azimuth,
            sun_elevation         = self._sun_elevation,
            predicted_irradiance  = self._predicted_irradiance,
            panel_tilt            = self._panel_tilt,
            panel_rotation        = self._panel_rotation,
            true_irradiance       = self._true_irradiance,
            energy_this_step      = self._energy_this_step,
            step_index            = self._step_index,
        )

    # ------------------------------------------------------------------
    # STEP 3 — Helper functions: sun position, irradiance, energy
    # ------------------------------------------------------------------
    # These three functions are the entire "physics engine" of the env.
    # Each is pure (no side-effects) and operates only on its arguments,
    # making them trivial to unit-test in isolation.
    #
    # Complexity budget: sin() and cos() only — no astronomy libraries,
    # no Julian dates, no atmospheric refraction models.
    # ------------------------------------------------------------------

    def _compute_sun_position(
        self,
        time_of_day: float,
    ) -> tuple[float, float]:
        """
        Approximate the sun's azimuth and elevation for a given hour.

        Model
        -----
        We treat the sun as moving along a simple sinusoidal arc:

          • Elevation follows a half-sine wave that peaks at solar noon
            (12:00) and touches zero at sunrise (6:00) and sunset (18:00).
            Values outside [6, 18] are negative — sun below horizon.

          • Azimuth sweeps linearly from 90° (East) at sunrise through
            180° (South, solar noon) to 270° (West) at sunset.
            Outside daylight hours it continues the same linear sweep —
            this is intentionally simple and gives a stable, continuous
            signal for the agent to observe even at night.

        Both outputs are in degrees.

        Parameters
        ----------
        time_of_day : float
            Current hour in [0, 24].

        Returns
        -------
        (azimuth_deg, elevation_deg) : tuple[float, float]
        """

        # ── Elevation ────────────────────────────────────────────────────
        #
        # We want:
        #   elevation = 0   at t = SUNRISE_HOUR  (6:00)
        #   elevation = MAX at t = NOON_HOUR      (12:00)   ← sin peaks here
        #   elevation = 0   at t = SUNSET_HOUR   (18:00)
        #   elevation < 0   outside [6, 18]      (sun below horizon)
        #
        # Mapping time → angle for sin():
        #   At t=6  → sin(0)     = 0
        #   At t=12 → sin(π/2)   = 1   (peak)
        #   At t=18 → sin(π)     = 0
        #
        #   angle = π * (time - SUNRISE) / DAY_LENGTH
        #         = π * (t - 6) / 12
        #
        # We then subtract a small bias equal to sin at midnight so the
        # curve dips slightly below zero at night — gives the agent a
        # meaningful negative signal that the sun is genuinely absent.

        SUNRISE_HOUR: float = 6.0
        DAY_LENGTH:   float = 12.0   # hours of daylight (6:00 → 18:00)
        MAX_ELEVATION: float = 90.0 - self.LATITUDE_DEG   # ~61.4° at 28.6°N

        # sin argument: 0 at sunrise, π/2 at noon, π at sunset, wraps outside
        elevation_angle_rad = math.pi * (time_of_day - SUNRISE_HOUR) / DAY_LENGTH

        # Raw elevation: positive during day, negative at night
        elevation_deg = MAX_ELEVATION * math.sin(elevation_angle_rad)

        # ── Azimuth ──────────────────────────────────────────────────────
        #
        # We want:
        #   azimuth = 90°  at sunrise (6:00)  → sun rises in the East
        #   azimuth = 180° at noon    (12:00) → sun due South (Northern hemisphere)
        #   azimuth = 270° at sunset  (18:00) → sun sets in the West
        #
        # Linear interpolation across the full day:
        #   azimuth = 90 + 180 * (time - SUNRISE) / DAY_LENGTH
        #
        # At t=6:  90 + 180*0   = 90°  ✓
        # At t=12: 90 + 180*0.5 = 180° ✓
        # At t=18: 90 + 180*1   = 270° ✓
        #
        # Outside [6,18] the formula extrapolates naturally:
        #   t=0:  90 + 180*(-0.5) = 0°  (North, sun at nadir — symbolic)
        #   t=24: 90 + 180*(1.5)  = 360° = 0° (wraps back to North)
        # This is continuous and never causes sudden jumps.

        azimuth_deg = 90.0 + 180.0 * (time_of_day - SUNRISE_HOUR) / DAY_LENGTH

        # Normalise azimuth to [0, 360) using modulo
        azimuth_deg = azimuth_deg % 360.0

        return azimuth_deg, elevation_deg

    # ------------------------------------------------------------------

    def _compute_irradiance(
        self,
        sun_elevation: float,
        cloud_variance: float,
    ) -> float:
        """
        Compute ground-truth solar irradiance (W/m²) at the current step.

        Formula
        -------
        When the sun is above the horizon, irradiance scales with the sine
        of the elevation angle — this captures the "air mass" effect: a
        low sun has to push through more atmosphere than a high sun.

            base_irradiance = PEAK * sin(elevation_rad)   if elevation > 0
                            = 0                            otherwise

        A cloud attenuation factor is then sampled and applied:

            cloud_factor    = clamp(1 - |N(0, cloud_variance)|, 0, 1)
            true_irradiance = base_irradiance * cloud_factor

        The cloud_factor is stored in self._cloud_factor so step() can
        include it in the EpisodeInfo for diagnostics.

        Parameters
        ----------
        sun_elevation : float
            Current sun elevation in degrees.  Negative = below horizon.
        cloud_variance : float
            Std-dev of Gaussian cloud noise.  0 = perfectly clear sky.
            Sourced from config.cloud_noise_std; passed in explicitly so
            this function remains pure (no direct config access).

        Returns
        -------
        float
            Ground-truth irradiance in W/m², always ≥ 0.
        """

        # ── Night guard ──────────────────────────────────────────────────
        # If the sun hasn't risen (or has set), there is nothing to harvest.
        # Returning zero immediately avoids a negative sin() result leaking
        # through — belt-and-suspenders even though sin of small negatives
        # is also small and negative.
        if sun_elevation <= 0.0:
            self._cloud_factor = 1.0   # no clouds needed at night
            return 0.0

        # ── Base irradiance: scales with sin(elevation) ──────────────────
        #
        # sin(90°) = 1.0  → full irradiance at zenith
        # sin(10°) ≈ 0.17 → only 17% of peak at a low sun angle
        #
        # This models the longer atmospheric path at low angles without
        # needing an actual air-mass formula.
        elevation_rad   = math.radians(sun_elevation)
        base_irradiance = self.PEAK_IRRADIANCE * math.sin(elevation_rad)

        # ── Cloud attenuation ────────────────────────────────────────────
        #
        # Sample a non-negative noise value from a half-normal distribution
        # (we take the absolute value so attenuation is always ≥ 0).
        # Then subtract from 1 to get a multiplicative factor in [0, 1].
        #
        # cloud_variance = 0  → cloud_factor = 1.0  (clear sky, no change)
        # cloud_variance = 0.3 → cloud_factor ≈ 0.7 on average (heavy clouds)
        #
        # We clamp to [0.05, 1.0]:
        #   upper 1.0  → can never amplify beyond clear sky
        #   lower 0.05 → even on the cloudiest step, 5% leaks through
        #                (avoids complete zero-reward cliffs)
        if cloud_variance > 0.0:
            noise         = abs(self.rng.gauss(mu=0.0, sigma=cloud_variance))
            cloud_factor  = max(0.05, min(1.0, 1.0 - noise))
        else:
            cloud_factor  = 1.0   # easy task: always clear sky

        # Cache so step() can put it in EpisodeInfo
        self._cloud_factor = cloud_factor

        # ── Final irradiance ─────────────────────────────────────────────
        true_irradiance = base_irradiance * cloud_factor

        # Hard floor: irradiance is a physical quantity, never negative
        return max(0.0, true_irradiance)

    # ------------------------------------------------------------------

    def _compute_energy(
        self,
        panel_tilt: float,
        panel_rotation: float,
        sun_azimuth: float,
        sun_elevation: float,
    ) -> tuple[float, float]:
        """
        Compute energy harvested and the alignment angle for this step.

        The core formula
        ----------------
        A solar panel generates power proportional to how directly it faces
        the sun.  "Directly facing" means the panel's surface normal points
        straight at the sun — the angle between them is 0°.

        We approximate the 3-D alignment using two independent angle deltas:

            Δtilt     = panel_tilt     − sun_elevation   (vertical mismatch)
            Δrotation = panel_rotation − sun_azimuth     (horizontal mismatch)

        The combined angle difference uses the 2-D Euclidean distance in
        angle space (degrees), clamped to [0°, 90°] so the cosine never
        goes negative (a panel facing away from the sun yields 0, not debt):

            angle_diff = clamp( sqrt(Δtilt² + Δrotation²), 0, 90 )

        Energy for this timestep:

            energy = irradiance * cos(angle_diff_rad) * step_duration_hours

        Multiplying by step_duration_hours converts W/m² → Wh/m².

        Parameters
        ----------
        panel_tilt, panel_rotation : float   degrees
        sun_azimuth, sun_elevation : float   degrees

        Returns
        -------
        (energy_wh, angle_diff_deg) : tuple[float, float]
            energy_wh      : Wh harvested this step  (≥ 0)
            angle_diff_deg : alignment error in degrees (0 = perfect)
        """

        # ── Night guard ──────────────────────────────────────────────────
        # No sun = no energy.  Skip all trig.
        if sun_elevation <= 0.0:
            return 0.0, 90.0   # 90° = maximum misalignment (symbolic)

        # ── Angle deltas ─────────────────────────────────────────────────
        #
        # tilt delta: how far off vertically is the panel from the sun?
        delta_tilt = panel_tilt - sun_elevation

        # rotation delta: shortest angular distance around the compass.
        # A naïve subtraction can give -350° when the answer is 10°, so we
        # normalise to [-180, +180] first using the ((x+180) % 360 - 180) trick.
        raw_rotation_diff = panel_rotation - sun_azimuth
        delta_rotation    = (raw_rotation_diff + 180.0) % 360.0 - 180.0

        # ── Combined angle difference ─────────────────────────────────────
        #
        # Euclidean distance in (tilt, rotation) space.
        # Think of it as the hypotenuse: if the panel is 20° off in tilt
        # and 30° off in rotation, the combined error is √(400+900) ≈ 36°.
        angle_diff_deg = math.sqrt(delta_tilt ** 2 + delta_rotation ** 2)

        # Clamp to [0°, 90°]:
        #   • 0°  → cos(0)      = 1.0  → perfect alignment, full energy
        #   • 90° → cos(π/2)    = 0.0  → panel edge-on to sun, zero energy
        #   • >90° would give negative cos → we never want negative energy
        angle_diff_deg = max(0.0, min(90.0, angle_diff_deg))
        angle_diff_rad = math.radians(angle_diff_deg)

        # ── Energy this step ─────────────────────────────────────────────
        #
        # self._true_irradiance is already set by _compute_irradiance()
        # before _compute_energy() is called in step().
        #
        # step_duration_hours converts instantaneous power (W/m²) to
        # energy (Wh/m²) over the timestep interval.
        energy_wh = (
            self._true_irradiance
            * math.cos(angle_diff_rad)
            * self.step_duration_hours
        )

        # Final safety clamp — can't harvest negative energy
        energy_wh = max(0.0, energy_wh)

        return energy_wh, angle_diff_deg

    # ------------------------------------------------------------------
    # STEP 4 — step(action)
    # ------------------------------------------------------------------
    # This is the heart of the environment.  Every call to step() does
    # exactly ONE simulated timestep in this order:
    #
    #   1. Validate the environment is ready
    #   2. Apply the action  → update panel angles
    #   3. Advance time      → increment time_of_day
    #   4. Recompute sun     → new azimuth + elevation
    #   5. Recompute irradiance (with cloud noise from config)
    #   6. Build prediction  → add forecast noise for medium/hard
    #   7. Compute energy    → physics formula
    #   8. Compute reward    → energy − movement_cost − misalignment_penalty
    #   9. Check termination → time >= 24 or step limit reached
    #  10. Pack and return   → StepResult
    #
    # Each stage is a clearly labelled block so you can jump straight to
    # the part you want to debug or modify.
    # ------------------------------------------------------------------

    def step(self, action: SolarAction) -> StepResult:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        action : SolarAction
            The agent's chosen tilt_change and rotation_change, each in
            [-1, 1].  Values are scaled by max_tilt_step / max_rotation_step
            from the EpisodeConfig before being applied.

        Returns
        -------
        StepResult
            Contains the new SolarState, scalar reward, terminated/truncated
            flags, RewardBreakdown, and EpisodeInfo.

        Raises
        ------
        RuntimeError
            If reset() has not been called before the first step().
        """

        # ── Guard ────────────────────────────────────────────────────────
        if not self._initialised:
            raise RuntimeError(
                "reset() must be called before step(). "
                "The environment has not been initialised yet."
            )

        # =================================================================
        # STAGE 1 — Apply the action to the panel
        # =================================================================
        #
        # The action holds FRACTIONAL deltas in [-1, 1].
        # We scale them by the config's max step sizes to get real degrees:
        #
        #   tilt_change = 1.0  →  +max_tilt_step  degrees (e.g. +5°)
        #   tilt_change = -0.5 →  -2.5 degrees
        #
        # After applying we CLAMP to the physical limits of the panel.
        # Tilt is bounded [0, 90] — can't tilt below flat or past vertical.
        # Rotation wraps around [0, 360) — 361° is the same as 1°.

        # --- Compute raw degree deltas ---
        tilt_delta     = action.tilt_change     * self.config.max_tilt_step
        rotation_delta = action.rotation_change * self.config.max_rotation_step

        # --- Apply to panel state ---
        new_tilt     = self._panel_tilt     + tilt_delta
        new_rotation = self._panel_rotation + rotation_delta

        # --- Clamp tilt to [MIN_TILT, MAX_TILT] ---
        # If the agent tries to tilt past 90°, the panel just stops at 90°.
        # The clamped movement is still penalised (agent should learn limits).
        new_tilt = max(self.MIN_TILT, min(self.MAX_TILT, new_tilt))

        # --- Wrap rotation into [0, 360) ---
        # Using modulo so 361° → 1° and -10° → 350°, no discontinuity.
        new_rotation = new_rotation % self.MAX_ROTATION

        # --- Compute how much the panel actually moved (for movement cost) ---
        # We measure the true movement after clamping so the penalty reflects
        # real physical work done, not what the agent asked for.
        actual_tilt_move     = abs(new_tilt     - self._panel_tilt)
        actual_rotation_move = abs(new_rotation - self._panel_rotation)

        # Handle the wrap-around edge case for rotation:
        # If the panel crossed 0°/360°, the naive difference would be ~360°
        # when the true movement was tiny.  We take the shorter arc.
        if actual_rotation_move > 180.0:
            actual_rotation_move = 360.0 - actual_rotation_move

        # Commit the new angles
        self._panel_tilt     = new_tilt
        self._panel_rotation = new_rotation

        # =================================================================
        # STAGE 2 — Advance time
        # =================================================================
        #
        # Each step advances by `step_duration_hours` (e.g. 0.25h = 15 min
        # for a 96-step episode).  Time is bounded at 24 — we don't wrap
        # back to 0 because the episode ends when day is done.

        self._time_of_day = min(
            24.0,
            self._time_of_day + self.step_duration_hours
        )
        self._step_index += 1

        # =================================================================
        # STAGE 3 — Recompute sun position
        # =================================================================
        #
        # Now that time has advanced, the sun has moved.  We call the same
        # helper used in reset() — this keeps sun geometry logic in one place.

        self._sun_azimuth, self._sun_elevation = self._compute_sun_position(
            self._time_of_day
        )

        # =================================================================
        # STAGE 4 — Compute true irradiance (with cloud noise)
        # =================================================================
        #
        # This is the GROUND TRUTH irradiance — what the sun actually
        # delivers.  We pass in config.cloud_noise_std; the helper samples
        # the noise and stores the cloud_factor internally.
        #
        # easy   → cloud_noise_std = 0.0  → always clear sky
        # medium → cloud_noise_std = 0.2  → mild clouds
        # hard   → cloud_noise_std = 0.4  → heavy, variable clouds

        self._true_irradiance = self._compute_irradiance(
            sun_elevation  = self._sun_elevation,
            cloud_variance = self.config.cloud_noise_std,
        )

        # =================================================================
        # STAGE 5 — Build the predicted irradiance (agent's forecast)
        # =================================================================
        #
        # The agent sees `predicted_irradiance`, NOT the true value.
        # On easy tasks the prediction equals truth.
        # On medium/hard tasks we add Gaussian noise to simulate an
        # imperfect forecast model.
        #
        # The noise is zero-mean so predictions are unbiased on average —
        # the agent needs to learn to track despite uncertainty, not to
        # compensate for a systematic bias.
        #
        # We clamp the prediction to [0, PEAK_IRRADIANCE] so the agent
        # never sees a physically impossible negative or super-solar value.

        if self.config.prediction_noise_std > 0.0:
            forecast_noise = self.rng.gauss(
                mu    = 0.0,
                sigma = self.config.prediction_noise_std,
            )
        else:
            forecast_noise = 0.0

        self._predicted_irradiance = max(
            0.0,
            min(
                self.PEAK_IRRADIANCE,
                self._true_irradiance + forecast_noise,
            )
        )

        # =================================================================
        # STAGE 6 — Compute energy harvested this step
        # =================================================================
        #
        # _compute_energy() uses self._true_irradiance (set in Stage 4)
        # and the current panel angles to calculate:
        #
        #   energy_wh      = irradiance × cos(angle_diff) × step_hours
        #   angle_diff_deg = √(Δtilt² + Δrotation²), clamped to [0,90]
        #
        # angle_diff_deg is also returned so we can use it in the reward.

        self._energy_this_step, angle_diff_deg = self._compute_energy(
            panel_tilt     = self._panel_tilt,
            panel_rotation = self._panel_rotation,
            sun_azimuth    = self._sun_azimuth,
            sun_elevation  = self._sun_elevation,
        )

        # Accumulate total episode energy
        self._cumulative_energy += self._energy_this_step

        # =================================================================
        # STAGE 7 — Compute reward
        # =================================================================
        #
        # The reward has three terms:
        #
        #   reward = energy_reward
        #          − movement_cost
        #          − misalignment_penalty
        #
        # ── Term 1: energy_reward ────────────────────────────────────────
        # Directly proportional to energy harvested.
        # Scaled by energy_reward_scale (default 1.0) so you can tune the
        # magnitude relative to the penalties without touching the physics.
        #
        # ── Term 2: movement_cost ────────────────────────────────────────
        # Penalises physically moving the panel.
        # = weight × (|Δtilt| + |Δrotation|) / (max_tilt + max_rot)
        #
        # We normalise by the maximum possible movement per step so the
        # penalty is always in [0, weight] regardless of max_step sizes.
        # This makes the weight hyperparameter directly interpretable:
        #   weight = 0.1  →  maximum possible cost per step = 0.1
        #
        # easy   → weight = 0.0  → free to move
        # medium → weight = 0.05 → small nudge to avoid thrashing
        # hard   → weight = 0.2  → strong incentive to move efficiently
        #
        # ── Term 3: misalignment_penalty ─────────────────────────────────
        # Penalises the panel for being far from the optimal sun angle.
        # = weight × (angle_diff_deg / 90°)
        #
        # Normalising by 90° keeps the penalty in [0, weight].
        # This term is crucial for partial progress: even if the panel
        # hasn't perfectly aligned, reducing angle_diff earns positive signal.
        # Without it, the agent would get zero gradient at night (when energy
        # is already 0) and fail to learn to pre-position for sunrise.

        # --- energy_reward ---
        energy_reward = self._energy_this_step * self.config.energy_reward_scale

        # --- movement_cost ---
        # Normalise by the max possible movement per step
        max_possible_move = (
            self.config.max_tilt_step + self.config.max_rotation_step
        )
        normalised_move = (actual_tilt_move + actual_rotation_move) / max_possible_move
        movement_cost   = self.config.movement_penalty_weight * normalised_move

        # --- misalignment_penalty ---
        # Normalise angle_diff by 90° → gives a fraction in [0, 1]
        normalised_misalignment = angle_diff_deg / 90.0
        misalignment_penalty    = (
            self.config.misalignment_penalty_weight * normalised_misalignment
        )

        # --- At night, zero out the misalignment penalty ---
        # When the sun is below the horizon there is no correct alignment —
        # penalising the agent for not facing an underground sun is unfair.
        # We keep movement_cost so the agent still learns not to thrash at night.
        if self._sun_elevation <= 0.0:
            misalignment_penalty = 0.0

        # --- Assemble the breakdown ---
        breakdown = RewardBreakdown(
            energy_reward        = energy_reward,
            movement_cost        = movement_cost,
            misalignment_penalty = misalignment_penalty,
        )

        # Total scalar reward (used by the RL algorithm)
        reward = breakdown.total

        # =================================================================
        # STAGE 8 — Termination and truncation
        # =================================================================
        #
        # terminated: the episode reached a *natural* end — the sun has
        #             completed its arc (time >= 24h).
        #
        # truncated:  the episode was cut short by hitting the step limit.
        #             This should rarely happen because max_steps is
        #             calibrated so step_duration × max_steps = 24h exactly,
        #             but we include it for safety if configs are edited.
        #
        # The distinction matters for advantage estimation in some RL
        # algorithms (e.g. PPO bootstraps from truncated but not terminated).

        terminated = self._time_of_day >= 24.0
        truncated  = (self._step_index >= self.max_steps) and not terminated

        # =================================================================
        # STAGE 9 — Build info dict (EpisodeInfo)
        # =================================================================
        #
        # Compute optimal angles for this sun position so the info dict
        # can report how far off the panel was from perfect alignment.
        # These are the tilt/rotation values that would give angle_diff = 0.
        #
        # Optimal tilt    = sun_elevation  (panel tilted to face the sun)
        # Optimal rotation = sun_azimuth   (panel rotated to face the sun)

        optimal_tilt     = max(self.MIN_TILT, self._sun_elevation)
        optimal_rotation = self._sun_azimuth

        info = EpisodeInfo(
            angle_difference_deg      = angle_diff_deg,
            optimal_tilt              = optimal_tilt,
            optimal_rotation          = optimal_rotation,
            cloud_factor              = self._cloud_factor,
            task                      = self.task,
            episode_cumulative_energy = self._cumulative_energy,
        )

        # =================================================================
        # STAGE 10 — Build and return StepResult
        # =================================================================

        return StepResult(
            state      = self._build_state(),
            reward     = reward,
            terminated = terminated,
            truncated  = truncated,
            breakdown  = breakdown,
            info       = info,
        )

    # ------------------------------------------------------------------
    # STEP 5 — state()
    # ------------------------------------------------------------------
    # Design goals:
    #   * Pure read — zero side-effects, zero recomputation.
    #   * Always valid — guards against calls before reset().
    #   * Delegates entirely to _build_state() which is the single
    #     canonical place that constructs a SolarState from internal fields.
    #
    # This method exists separately from reset() / step() because some
    # OpenEnv-compatible runners and evaluation harnesses call state()
    # independently to snapshot the environment mid-episode without
    # advancing the simulation.  It must be a no-op in terms of state.
    # ------------------------------------------------------------------

    def state(self) -> SolarState:
        """
        Return a snapshot of the current environment state.

        This is a pure read — it does not advance time, resample noise,
        or modify any internal variable.  Calling it twice in a row
        returns identical values (assuming no step() call in between).

        Returns
        -------
        SolarState
            The current observation, identical to what the last reset()
            or step() returned.

        Raises
        ------
        RuntimeError
            If called before reset() — there is no valid state yet.
        """

        # Guard: catch the common mistake of calling state() before reset()
        if not self._initialised:
            raise RuntimeError(
                "state() called before reset(). "
                "Call reset() first to initialise the episode."
            )

        # Delegate to the shared builder — single source of truth for
        # how internal fields map to a SolarState object.
        return self._build_state()